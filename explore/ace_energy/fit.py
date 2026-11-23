"""A linear ACE energy as the whole score model of RSGM's noising process, fitted in closed form (variant B of the ACE
line, docs/experiments: the ACE report). On the CPU, from the root of the repository:

    python explore/ace_energy/fit.py --check                        # the self-checks, a few seconds
    python explore/ace_energy/fit.py --out ~/data/experiments/ace/energy_b.pt

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

import torch
from torch.func import jacfwd, vmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # the code of this checkout, before an editable install of another one

from hp.evaluation.data import load_reference  # noqa: E402
from hp.models import igso3  # noqa: E402
from hp.models.ace_features import AceFeatures  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.rsgm import exp_frame  # noqa: E402

ROTATIONS, SPLIT = "~/data/AMASS_rotations_posedflow_c10", "posedflow"
REFERENCE = "~/data/experiments/factorial/rsgm_posedflow_fact_RDW_s10/run_0001"
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
    """G [B, 3J, P]: d B_p(x_j @ exp(eps e_a)) / d eps at 0, in float32."""
    x = x.float()

    def one(delta, xi):
        return ace(exp_frame(xi, delta)[None])[0]

    jac = vmap(jacfwd(one))(torch.zeros(x.shape[0], x.shape[1], 3), x)  # [B, P, J, 3]
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


def fit(ace, ref, bins: Bins, train: torch.Tensor, per_bin: int, batch: int, ridge: float, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    p = ace.dim
    w, fit_loss = [], []
    for k in range(len(bins.centres)):
        t0 = time.perf_counter()
        a = torch.zeros(p, p, dtype=torch.float64)
        b = torch.zeros(p, dtype=torch.float64)
        yy, rows = 0.0, 0
        lo, hi = bins.t_range(k)
        for _ in range(0, per_bin, batch):
            x0 = train[torch.randint(len(train), (batch,), generator=g)]
            t = lo + (hi - lo) * torch.rand(batch, dtype=torch.float64, generator=g)
            xt, target, _ = draw(ref, x0, t, g)
            gm = jacobian(ace, xt).double().reshape(-1, p)  # [B 3J, P]
            y = target.reshape(-1)
            a += gm.T @ gm
            b += gm.T @ y
            yy += float(y @ y)
            rows += batch
        lam = ridge * float(a.diagonal().mean())
        wk = -torch.linalg.solve(a + lam * torch.eye(p, dtype=torch.float64), b)
        loss = (wk @ a @ wk + 2 * wk @ b + yy) / rows
        w.append(wk)
        fit_loss.append(float(loss))
        print(f"bin {k + 1:2d}/{len(bins.centres)}: t in [{lo:.4f}, {hi:.4f}], s in [{bins.edges[k].exp():.2e}, "
              f"{bins.edges[k + 1].exp():.2e}]: fit loss {loss:.3f} (zero score {yy / rows:.3f}), "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    return dict(w=torch.stack(w), fit_loss=fit_loss)


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
        pred = -torch.einsum("bkp,bp->bk", gm, bins.weights(w, t[i:i + batch], interp))
        energy.append(((pred - target[i:i + batch].reshape(len(gm), -1)) ** 2).sum(1))
    out["ACE energy"] = torch.cat(energy)
    for name, model in compare.items():
        losses = []
        with torch.no_grad():
            for i in range(0, len(valid), batch):
                sl = slice(i, i + batch)
                scaled = model.score(xt[sl].float(), t[sl].float()).double() * std[sl][..., None]
                losses.append(((scaled - target[sl]) ** 2).sum((1, 2)))
        out[name] = torch.cat(losses)
    return out


def check(ace, ref, valid) -> None:
    """The draws reproduce RSGM.loss, and the tangent Jacobian matches finite differences."""
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
    parser.add_argument("--ridge", type=float, default=1e-6, help="relative to the mean diagonal of G^T G")
    parser.add_argument("--valid", type=int, default=5000, help="validation poses of the evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", default=None, help="a .pt file for w, the bins and the losses")
    parser.add_argument("--check", action="store_true", help="only the self-checks")
    parser.add_argument("--plain", action="store_true",
                        help="the plain descriptors of the input (1 / r gradients) instead of the smooth ones")
    parser.add_argument("--one-body", type=int, choices=(0, 1, 2), default=0,
                        help="also terms of each joint's own rotation: 1 its matrix entries, 2 also their products")
    parser.add_argument("--load", default=None, help="a .pt file of an earlier fit: evaluate it, fit nothing")
    parser.add_argument("--interp", choices=("linear", "nearest"), default="linear", help="w between bins")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    ref = load_model(Path(args.reference).expanduser(), device="cpu", weights="weights_ema").eval()
    ace = AceFeatures(smooth=not args.plain)
    if args.one_body:
        ace = WithOneBody(ace, args.one_body)
    valid = load_reference("valid", ROTATIONS, SPLIT)
    valid = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(0))[:args.valid]]
    if args.check:
        check(ace, ref, valid)
        return
    bins = Bins(ref, ref.loss_t_min, args.bins)
    if args.load:
        saved = torch.load(Path(args.load).expanduser(), weights_only=False)
        assert torch.equal(saved["edges"], bins.edges), "the bins of the loaded fit differ (--bins)"
        result = dict(w=saved["w"], fit_loss=saved["fit_loss"])
    else:
        result = None
    train = None if result else load_reference("train", ROTATIONS, SPLIT)
    print(f"{ace.dim} descriptors, {args.bins} bins x {args.per_bin} samples ({args.per_bin * 57:,} rows per bin), "
          f"{len(train) if train is not None else 0:,} training poses, {len(valid)} validation poses", flush=True)
    if result is None:
        result = fit(ace, ref, bins, train, args.per_bin, args.batch, args.ridge, args.seed)
    compare = {Path(r).expanduser().parent.name: load_model(Path(r).expanduser(), device="cpu",
                                                             weights="weights_ema").eval() for r in args.compare}
    losses = evaluate(ace, ref, bins, result["w"], valid, compare, args.batch, ref.loss_t_min, args.interp)
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
    if args.out and not args.load:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(w=result["w"], edges=bins.edges, fit_loss=result["fit_loss"], table=table,
                        settings=vars(args), descriptors=ace.dim), out)
        out.with_suffix(".json").write_text(json.dumps(dict(table=table, fit_loss=result["fit_loss"],
                                                            settings=vars(args)), indent=1))
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
