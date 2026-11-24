"""A linear ACE energy as the whole score model of RSGM's noising process, fitted in closed form (variant B of the ACE
line, docs/experiments: the ACE report). On the CPU, from the root of the repository:

    python explore/ace_energy/fit.py --check                        # the self-checks, a few seconds
    python explore/ace_energy/fit.py --out ~/data/experiments/ace/energy_b.pt
    python explore/ace_energy/fit.py --wigner 1 2 3 --pairs 1 2 3 --device cuda:0 --out ...   # on a GPU server

Model. E_t(x) = sum_p c_p(t) B_p(x), with B the descriptors of hp.models.ace_features (distances, angles, twists of
the pose; smooth=True, polynomials with bounded gradients, unless --plain) and the score s(x, t) = -grad E_t(x), the derivative along the tangent directions x_j @ exp(eps e_a) (the
frame of RSGM's score). The loss of RSGM (std weighting) is || std(t) s(x_t, t) - target ||^2 with
target = std(t) grad log p_t(x_t | x_0); with w(t) = std(t) c(t) the scaled score is -G(x_t) w(t), G [3J, P] the
tangent Jacobian of the descriptors, linear in w. So for every bin of the noise level (equal widths in log s) the best
w is a least-squares solution, w = -(G^T G + ridge)^-1 G^T target, summed over the samples of the bin; between bin
centres w is interpolated linearly in log s.

Samples: training poses (c10 of PoseD-Flow's AMASS), times uniform within each bin (the std weighting within a bin),
the IGSO(3) noise and target drawn by the noising of the reference RSGM run (--reference; its s(t), std(t) and angle
sampler), exactly as RSGM.loss draws them.

Evaluation on the validation poses, one draw per pose with a fixed generator, the same draws for every model: the loss
of the energy, of the zero score (the scale: E ||target||^2, about 3J) and of the trained networks (--compare), as a
mean and per bin of log s. The networks' training losses are the same quantity (their history.json "valid_loss" uses
other draws of the same distribution).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.func import jacfwd, vmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # the code of this checkout, before an editable install of another one

from hp.evaluation.data import load_reference  # noqa: E402
from hp.models import igso3  # noqa: E402
from hp.models.ace_features import AceFeatures  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.rsgm import exp_frame, global_rotations  # noqa: E402
from hp.models.so3_harmonics import WIGNER_PAIRS, pair_wigner_features, wigner_features  # noqa: E402

ROTATIONS, SPLIT = "~/data/AMASS_rotations_posedflow_c10", "posedflow"
REFERENCE = "~/data/experiments/factorial/rsgm_posedflow_fact_RDW_s10/run_0001"
DEVICE = torch.device("cpu")  # --device: where the Jacobians and the sums of the fit are computed
COMPARE = ["~/data/experiments/factorial/rsgm_posedflow_fact_RDW_s10/run_0001",
           "~/data/experiments/factorial/rsgm_posedflow_fact_N_s10/run_0001"]


class WithOneBody(torch.nn.Module):
    """The descriptors and one-body terms of every joint's own rotation: its 9 matrix entries (an energy linear in
    them is a matrix Fisher distribution per joint) and, with degree 2, their 45 products x_ja x_jb (a <= b), which
    span the l <= 2 harmonics D^l(x_j) on SO(3)."""

    def __init__(self, ace: AceFeatures, degree: int):
        super().__init__()
        self.ace, self.degree = ace, degree
        a, b = torch.triu_indices(9, 9)
        self.register_buffer("a", a, persistent=False)
        self.register_buffer("b", b, persistent=False)
        joints = len(ace.slot) - ace.slot.count(-1)
        self.dim = ace.dim + joints * (9 + (45 if degree >= 2 else 0))

    def forward(self, x):
        m = x.flatten(-2)  # [..., J, 9]
        out = [self.ace(x), m.flatten(-2)]
        if self.degree >= 2:
            out.append((m[..., self.a] * m[..., self.b]).flatten(-2))
        return torch.cat(out, dim=-1)


class WithHarmonics(torch.nn.Module):
    """The descriptors and harmonics of SO(3) (hp.models.so3_harmonics): D^l(x_j) of every joint's own rotation for l
    in `wigner` (one-body), and D^l(G_j), D^l(G_i^T G_k) of the joints with the root and the pairs of WIGNER_PAIRS for
    l in `pairs` (two-body orientations; G relative to the root)."""

    def __init__(self, ace, wigner: Sequence[int], pairs: Sequence[int]):
        super().__init__()
        self.ace, self.wigner, self.pairs = ace, tuple(wigner), tuple(pairs)
        size = lambda degrees: sum((2 * l + 1) ** 2 for l in degrees)  # noqa: E731
        self.dim = ace.dim + 19 * size(self.wigner) + (19 + len(WIGNER_PAIRS)) * size(self.pairs)

    def forward(self, x):
        out = [self.ace(x)]
        if self.wigner:
            out.append(wigner_features(x, self.wigner))
        if self.pairs:
            out.append(pair_wigner_features(global_rotations(x), self.pairs))
        return torch.cat(out, dim=-1)


def draw(ref, x0: torch.Tensor, t: torch.Tensor, generator=None):
    """(x_t, target, std) [B, J, 3, 3], [B, J, 3], [B, J] in float64 for poses x0 at times t [B] (float64), as
    RSGM.loss: s and std of every joint, an IGSO(3) angle and a uniform axis."""
    s, std = ref.joint_s(t), ref.joint_std(t)
    omega = ref.angles.sample(s, generator)
    axis = igso3.random_axes(s.shape, generator)
    xt = exp_frame(x0.double(), omega[..., None] * axis)
    target = std[..., None] * igso3.d_log_density(omega, s)[..., None] * axis
    return xt, target, std


def jacobian(ace, x: torch.Tensor) -> torch.Tensor:
    """G [B, 3J, P]: d B_p(x_j @ exp(eps e_a)) / d eps at 0, in float32 on the device of the descriptors."""
    device = next((b.device for b in ace.buffers()), torch.device("cpu"))
    x = x.float().to(device)

    def one(delta, xi):
        return ace(exp_frame(xi, delta)[None])[0]

    jac = vmap(jacfwd(one))(torch.zeros(x.shape[0], x.shape[1], 3, device=device), x)  # [B, P, J, 3]
    return jac.flatten(2).transpose(1, 2)


class Bins:
    """Bins of equal width in log s over [s(t_min), s(1)]; w per bin, interpolated linearly in log s between centres."""

    def __init__(self, ref, t_min: float, n: int):
        self.ref = ref
        lo, hi = ref.s(torch.tensor(t_min, dtype=torch.float64)).log(), ref.s(torch.tensor(1.0, dtype=torch.float64)).log()
        self.edges = torch.linspace(float(lo), float(hi), n + 1, dtype=torch.float64)  # in log s
        self.centres = (self.edges[1:] + self.edges[:-1]) / 2

    def t_range(self, k: int):
        return [float(self.ref.t_of_s(self.edges[i].exp())) for i in (k, k + 1)]

    def weights(self, w: torch.Tensor, t: torch.Tensor, interp: str = "linear") -> torch.Tensor:
        """w(t) [B, P] from w [n, P] at times t [B]: linear in log s between bin centres, or that of the bin."""
        if interp == "nearest":
            return w[self.index(t)]
        u = self.ref.s(t).log()
        pos = ((u - self.centres[0]) / (self.centres[1] - self.centres[0])).clamp(0, len(self.centres) - 1)
        k = pos.floor().long().clamp(max=len(self.centres) - 2)
        a = (pos - k)[:, None]
        return (1 - a) * w[k] + a * w[k + 1]

    def index(self, t: torch.Tensor) -> torch.Tensor:
        u = self.ref.s(t).log()
        return torch.bucketize(u, self.edges[1:-1])


def fit(ace, ref, bins: Bins, train: torch.Tensor, per_bin: int, batch: int, ridges, seed: int) -> dict:
    """{ridge: dict(w [bins, P], fit_loss)}: one accumulation of G^T G per bin, one solution per ridge."""
    g = torch.Generator().manual_seed(seed)  # the draws on the CPU: the same samples on every device
    p, device = ace.dim, DEVICE
    w = {r: [] for r in ridges}
    fit_loss = {r: [] for r in ridges}
    for k in range(len(bins.centres)):
        t0 = time.perf_counter()
        a = torch.zeros(p, p, dtype=torch.float64, device=device)
        b = torch.zeros(p, dtype=torch.float64, device=device)
        yy, rows = 0.0, 0
        lo, hi = bins.t_range(k)
        for _ in range(0, per_bin, batch):
            x0 = train[torch.randint(len(train), (batch,), generator=g)]
            t = lo + (hi - lo) * torch.rand(batch, dtype=torch.float64, generator=g)
            xt, target, _ = draw(ref, x0, t, g)
            gm = jacobian(ace, xt).double().reshape(-1, p)  # [B 3J, P]
            y = target.reshape(-1).to(device)
            a += gm.T @ gm
            b += gm.T @ y
            yy += float(y @ y)
            rows += batch
        for r in ridges:
            lam = r * float(a.diagonal().mean())
            wk = -torch.linalg.solve(a + lam * torch.eye(p, dtype=torch.float64, device=device), b)
            w[r].append(wk.cpu())
            fit_loss[r].append(float((wk @ a @ wk + 2 * wk @ b + yy) / rows))
        losses = ", ".join(f"{fit_loss[r][-1]:.3f}" for r in ridges)
        print(f"bin {k + 1:2d}/{len(bins.centres)}: t in [{lo:.4f}, {hi:.4f}], s in [{bins.edges[k].exp():.2e}, "
              f"{bins.edges[k + 1].exp():.2e}]: fit loss {losses} (ridge {', '.join(f'{r:g}' for r in ridges)}; "
              f"zero score {yy / rows:.3f}), {time.perf_counter() - t0:.0f} s", flush=True)
    return {r: dict(w=torch.stack(w[r]), fit_loss=fit_loss[r]) for r in ridges}


def evaluate(ace, ref, bins: Bins, w: torch.Tensor, valid: torch.Tensor, compare, batch: int, t_min: float,
             interp: str = "linear") -> dict:
    """Losses [N] per model on one fixed draw per validation pose, and the times."""
    g = torch.Generator().manual_seed(0)
    t = t_min + (1 - t_min) * torch.rand(len(valid), dtype=torch.float64, generator=g)
    xt, target, std = draw(ref, valid, t, g)
    out = {"t": t, "zero": (target ** 2).sum((1, 2))}
    energy = []
    for i in range(0, len(valid), batch):
        gm = jacobian(ace, xt[i:i + batch]).double()  # [b, 3J, P]
        pred = -torch.einsum("bkp,bp->bk", gm, bins.weights(w, t[i:i + batch], interp).to(gm.device)).cpu()
        energy.append(((pred - target[i:i + batch].reshape(len(pred), -1)) ** 2).sum(1))
    out["ACE energy"] = torch.cat(energy)
    for name, model in compare.items():
        losses = []
        with torch.no_grad():
            for i in range(0, len(valid), batch):
                sl = slice(i, i + batch)
                scaled = model.score(xt[sl].float().to(DEVICE), t[sl].float().to(DEVICE)).double().cpu() * std[sl][..., None]
                losses.append(((scaled - target[sl]) ** 2).sum((1, 2)))
        out[name] = torch.cat(losses)
    return out


def check(ace, ref, valid) -> None:
    """The draws reproduce RSGM.loss, and the tangent Jacobian matches finite differences (on the CPU)."""
    ace = ace.cpu()
    x0 = valid[:300]
    ref.eval()
    g = torch.Generator().manual_seed(0)  # RSGM.loss in eval mode: t, then the angles, then the axes, seed 0
    t = ref.loss_times(torch.rand(len(x0), dtype=torch.float64, generator=g))
    xt, target, std = draw(ref, x0, t, g)
    with torch.no_grad():
        mine = ((ref.score(xt.float(), t.float()).double() * std[..., None] - target) ** 2).sum((1, 2)).mean()
        theirs = ref.loss(x0)
    print(f"check: loss by these draws {float(mine):.5f}, RSGM.loss {float(theirs):.5f}")
    assert abs(float(mine) - float(theirs)) < 1e-3 * float(theirs)
    x = xt[:2].double()
    gm = jacobian(ace.double(), x).double()
    v = torch.randn(2, x.shape[1], 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    h = 1e-6
    fd = (ace(exp_frame(x, h * v)) - ace(exp_frame(x, -h * v))) / (2 * h)
    jv = torch.einsum("bkp,bk->bp", gm, v.reshape(2, -1))
    err = float((fd - jv).abs().max() / fd.abs().max())
    print(f"check: tangent Jacobian (float32) against finite differences (float64), max error {err:.2e} of the "
          f"largest derivative")
    assert err < 1e-4
    ace.float()
    print("checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default=REFERENCE, help="the RSGM run whose noising is used")
    parser.add_argument("--compare", nargs="*", default=COMPARE, help="trained RSGM runs evaluated on the same draws")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--per-bin", type=int, default=8000, help="training samples per bin")
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--ridge", type=float, nargs="+", default=[1e-6],
                        help="relative to the mean diagonal of G^T G; several: one fit each from the same sums, "
                             "saved as <out>_r<ridge>")
    parser.add_argument("--valid", type=int, default=5000, help="validation poses of the evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N (float64 sums; mps has no float64)")
    parser.add_argument("--out", default=None, help="a .pt file for w, the bins and the losses")
    parser.add_argument("--check", action="store_true", help="only the self-checks")
    parser.add_argument("--plain", action="store_true",
                        help="the plain descriptors of the input (1 / r gradients) instead of the smooth ones")
    parser.add_argument("--one-body", type=int, choices=(0, 1, 2), default=0,
                        help="also terms of each joint's own rotation: 1 its matrix entries, 2 also their products")
    parser.add_argument("--wigner", type=int, nargs="*", default=[], choices=(1, 2, 3),
                        help="one-body harmonics D^l(x_j) of these degrees (instead of --one-body)")
    parser.add_argument("--pairs", type=int, nargs="*", default=[], choices=(1, 2, 3),
                        help="two-body harmonics D^l(G_j), D^l(G_i^T G_k) of WIGNER_PAIRS of these degrees")
    parser.add_argument("--load", default=None, help="a .pt file of an earlier fit: evaluate it, fit nothing")
    parser.add_argument("--interp", choices=("linear", "nearest"), default="linear", help="w between bins")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    global DEVICE
    DEVICE = torch.device(args.device)

    ref = load_model(Path(args.reference).expanduser(), device="cpu", weights="weights_ema").eval()
    ace = AceFeatures(smooth=not args.plain)
    if args.one_body:
        ace = WithOneBody(ace, args.one_body)
    if args.wigner or args.pairs:
        ace = WithHarmonics(ace, args.wigner, args.pairs)
    ace = ace.to(DEVICE)
    valid = load_reference("valid", ROTATIONS, SPLIT)
    valid = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(0))[:args.valid]]
    if args.check:
        check(ace, ref, valid)
        return
    bins = Bins(ref, ref.loss_t_min, args.bins)
    if args.load:
        saved = torch.load(Path(args.load).expanduser(), map_location="cpu", weights_only=True)
        assert torch.equal(saved["edges"], bins.edges), "the bins of the loaded fit differ (--bins)"
        results = {saved["settings"].get("ridge", args.ridge[0]): dict(w=saved["w"], fit_loss=saved["fit_loss"])}
    else:
        results = None
    train = None if results else load_reference("train", ROTATIONS, SPLIT)
    print(f"{ace.dim} descriptors, {args.bins} bins x {args.per_bin} samples ({args.per_bin * 57:,} rows per bin), "
          f"{len(train) if train is not None else 0:,} training poses, {len(valid)} validation poses", flush=True)
    if results is None:
        results = fit(ace, ref, bins, train, args.per_bin, args.batch, args.ridge, args.seed)
    compare = {Path(r).expanduser().parent.name: load_model(Path(r).expanduser(), device=args.device,
                                                             weights="weights_ema").eval() for r in args.compare}
    for n_ridge, (ridge, result) in enumerate(results.items()):
        report(ace, ref, bins, ridge, result, valid, compare if n_ridge == 0 else {}, args, len(results) > 1)


def report(ace, ref, bins, ridge, result, valid, compare, args, several: bool) -> None:
    """Evaluate one fit on the validation draws, print its table (and the networks' with the first), save it."""
    losses = evaluate(ace, ref, bins, result["w"], valid, compare, args.batch, ref.loss_t_min, args.interp)
    losses[f"ACE energy, ridge {ridge:g}"] = losses.pop("ACE energy")
    k = bins.index(losses["t"])
    names = [n for n in losses if n != "t"]
    print("\nvalidation loss (std weighting, one draw per pose, the same draws for every model)")
    print("| model | mean | " + " | ".join(f"s {bins.edges[i].exp():.0e}" for i in range(len(bins.centres))) + " |")
    print("|---|---|" + "---|" * len(bins.centres))
    table = {}
    for n in names:
        per = [float(losses[n][k == i].mean()) for i in range(len(bins.centres))]
        table[n] = dict(mean=float(losses[n].mean()), per_bin=per)
        print(f"| {n} | {losses[n].mean():.3f} | " + " | ".join(f"{v:.2f}" for v in per) + " |")
    print("| median: |")
    for n in names:
        med = [float(losses[n][k == i].median()) for i in range(len(bins.centres))]
        table[n]["median_per_bin"] = med
        print(f"| {n} | {losses[n].median():.3f} | " + " | ".join(f"{v:.2f}" for v in med) + " |")
    e = losses[f"ACE energy, ridge {ridge:g}"]
    print(f"tails of the energy's loss (ridge {ridge:g}): 99th percentile {float(e.quantile(0.99)):.1f}, "
          f"max {float(e.max()):.1f}, poses above 100: {int((e > 100).sum())}")
    table["ACE energy"] = table[f"ACE energy, ridge {ridge:g}"]  # the name the loaders read
    if args.out and not args.load:
        out = Path(args.out).expanduser()
        if several:
            out = out.with_name(f"{out.stem}_r{ridge:g}{out.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        settings = dict(vars(args), ridge=ridge)
        torch.save(dict(w=result["w"], edges=bins.edges, fit_loss=result["fit_loss"], table=table,
                        settings=settings, descriptors=ace.dim), out)
        out.with_suffix(".json").write_text(json.dumps(dict(table=table, fit_loss=result["fit_loss"],
                                                            settings=settings), indent=1))
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
