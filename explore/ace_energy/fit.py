"""A linear ACE energy as the whole score model of RSGM's noising process, fitted in closed form (variant B of the ACE
line, docs/experiments: the ACE report). On the CPU, from the root of the repository:

    python explore/ace_energy/fit.py --check                        # the self-checks, a few seconds
    python explore/ace_energy/fit.py --out ~/data/experiments/ace/energy_b.pt
    python explore/ace_energy/fit.py --wigner 1 2 3 --pairs 1 2 3 --device cuda:0 --out ...   # on a GPU server
    python explore/ace_energy/fit.py ... --all-pairs 1 --products --single --batch 100        # an enriched basis

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from joint_density import JointDensity  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models import igso3  # noqa: E402
from hp.models.ace_features import AceFeatures  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.rsgm import PARENTS19, exp_frame, global_rotations  # noqa: E402
from hp.models.so3_harmonics import (PAIR_INDEX, WIGNER_PAIRS, pair_wigner_features, wigner_d,  # noqa: E402
                                     wigner_features)

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


JOINT_PAIRS = tuple((i, k) for i in range(19) for k in range(i + 1, 19))  # all 171
# the pairs that `pairs` does not have: not those of WIGNER_PAIRS, not parent and child (there G_i^T G_k = x_k)
OTHER_PAIRS = tuple((i, k) for i, k in JOINT_PAIRS
                    if (i, k) not in PAIR_INDEX and (k, i) not in PAIR_INDEX and PARENTS19[k] != i)


class WithHarmonics(torch.nn.Module):
    """The descriptors and harmonics of SO(3) (hp.models.so3_harmonics): D^l(x_j) of every joint's own rotation for l
    in `wigner` (one-body), and D^l(G_j), D^l(G_i^T G_k) of the joints with the root and the pairs of WIGNER_PAIRS for
    l in `pairs` (two-body orientations; G relative to the root). More two-body terms: D^l(G_i^T G_k) of OTHER_PAIRS
    for l in `all_pairs`, and with `products` the 81 products (x_i)_a (x_k)_b of the entries of the two joints' own
    rotations, of all 171 pairs."""

    def __init__(self, ace, wigner: Sequence[int], pairs: Sequence[int], all_pairs: Sequence[int] = (),
                 products: bool = False):
        super().__init__()
        self.ace, self.wigner, self.pairs = ace, tuple(wigner), tuple(pairs)
        self.all_pairs, self.products = tuple(all_pairs), bool(products)
        size = lambda degrees: sum((2 * l + 1) ** 2 for l in degrees)  # noqa: E731
        self.dim = (ace.dim + 19 * size(self.wigner) + (19 + len(WIGNER_PAIRS)) * size(self.pairs)
                    + len(OTHER_PAIRS) * size(self.all_pairs) + 81 * len(JOINT_PAIRS) * self.products)

    def forward(self, x):
        out = [self.ace(x)]
        if self.wigner:
            out.append(wigner_features(x, self.wigner))
        if self.pairs:
            out.append(pair_wigner_features(global_rotations(x), self.pairs))
        if self.all_pairs:
            i, k = (list(v) for v in zip(*OTHER_PAIRS))
            g = global_rotations(x)
            rel = g[:, i].transpose(-1, -2) @ g[:, k]
            out += [wigner_d(rel, l).flatten(1) for l in self.all_pairs]
        if self.products:
            i, k = (list(v) for v in zip(*JOINT_PAIRS))
            m = x.flatten(-2)  # [B, J, 9]
            out.append((m[:, i][..., :, None] * m[:, k][..., None, :]).flatten(1))
        return torch.cat(out, dim=-1)


def harmonic_basis(settings: dict) -> WithHarmonics:
    """The basis of a fit from its settings (the arguments of this script)."""
    return WithHarmonics(AceFeatures(smooth=not settings.get("plain", False)), settings.get("wigner", []),
                         settings.get("pairs", []), settings.get("all_pairs", []), settings.get("products", False))


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


HAT = torch.zeros(3, 3, 3)  # hat(e_c): x exp(eps hat(e_c)) = x + eps x HAT[c] + ...
for _c, _i, _j in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    HAT[_c, _i, _j], HAT[_c, _j, _i] = -1.0, 1.0


class Gram:
    """The sums G^T G and G^T y of a basis over the batches of a bin, in float64, with the structure of the products
    (x_i)_a (x_k)_b: the rows of joint j of G have only the 18 pairs of j (1458 of the 13851 columns), so the sums are
    taken per joint over the other features (dense, P0 columns) and these, and put together at the end. The same sums
    as of the full G (without products: the plain sums)."""

    def __init__(self, ace, device, single: bool):
        self.ace, self.device, self.single = ace, device, single
        self.products = bool(getattr(ace, "products", False))
        self.p = ace.dim
        self.p0 = self.p - 81 * len(JOINT_PAIRS) * self.products
        double = lambda *shape: torch.zeros(*shape, dtype=torch.float64, device=device)  # noqa: E731
        self.dense, self.b0 = double(self.p0, self.p0), double(self.p0)
        if self.products:
            self.base = WithHarmonics(ace.ace, ace.wigner, ace.pairs, ace.all_pairs)
            self.cross, self.block = double(19, self.p0, 18 * 81), double(19, 18 * 81, 18 * 81)
            self.by = double(19, 18 * 81)
            # the pairs of joint j, whether j is their first joint, and their columns among the products
            pairs = [[n for n, pair in enumerate(JOINT_PAIRS) if j in pair] for j in range(19)]
            self.other = torch.tensor([[JOINT_PAIRS[n][1] if JOINT_PAIRS[n][0] == j else JOINT_PAIRS[n][0] for n in row]
                                       for j, row in enumerate(pairs)])
            self.first = torch.tensor([[JOINT_PAIRS[n][0] == j for n in row] for j, row in enumerate(pairs)])
            self.columns = torch.tensor([[81 * n + q for n in row for q in range(81)] for row in pairs]) + self.p0
        else:
            self.base = ace

    def product_rows(self, x: torch.Tensor) -> torch.Tensor:
        """[19, 3B, 1458]: for every joint j the rows of its products, d (x_i)_a (x_k)_b / d delta_j."""
        b = x.shape[0]
        m = x.flatten(-2)  # [B, 19, 9]
        dm = torch.einsum("bjpq,cqr->bjcpr", x, HAT.to(x)).flatten(-2)  # [B, 19, 3, 9]
        other = m[:, self.other.to(x.device)]  # [B, 19, 18, 9]
        # the product of the derivative (index a if j is the first joint of the pair, else b) and the other's entries
        as_first = torch.einsum("bjca,bjnq->bjcnaq", dm, other)
        as_second = torch.einsum("bjcq,bjna->bjcnaq", dm, other)
        first = self.first.to(x.device)[None, :, None, :, None, None]
        rows = torch.where(first, as_first, as_second).reshape(b, 19, 3, -1)  # [B, 19, 3, 1458]
        return rows.permute(1, 0, 2, 3).reshape(19, 3 * b, -1)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """The batch x [B, 19, 3, 3] with the targets y [B, 57]."""
        gm = jacobian(self.base, x)  # [B, 57, P0]
        gm = gm if self.single else gm.double()
        flat, yy = gm.reshape(-1, self.p0), y.reshape(-1).to(gm)
        self.dense += (flat.T @ flat).to(self.device)
        self.b0 += (flat.T @ yy).to(self.device)
        if not self.products:
            return
        dtype = gm.dtype
        rows = self.product_rows(x.float().to(gm.device)).to(dtype)  # [19, 3B, 1458]
        g = gm.reshape(len(x), 19, 3, self.p0).permute(1, 0, 2, 3).reshape(19, -1, self.p0)  # [19, 3B, P0]
        yj = y.reshape(len(x), 19, 3).permute(1, 0, 2).reshape(19, -1).to(gm)
        self.cross += torch.bmm(g.transpose(1, 2), rows).to(self.device)
        self.block += torch.bmm(rows.transpose(1, 2), rows).to(self.device)
        self.by += torch.bmm(rows.transpose(1, 2), yj[..., None])[..., 0].to(self.device)

    def total(self):
        """(G^T G [P, P], G^T y [P])."""
        if not self.products:
            return self.dense, self.b0
        a = torch.zeros(self.p, self.p, dtype=torch.float64, device=self.device)
        b = torch.zeros(self.p, dtype=torch.float64, device=self.device)
        a[:self.p0, :self.p0], b[:self.p0] = self.dense, self.b0
        for j in range(19):
            c = self.columns[j].to(self.device)
            a[:self.p0].index_add_(1, c, self.cross[j])
            a[c[:, None], c[None, :]] += self.block[j]
            b.index_add_(0, c, self.by[j])
        a[self.p0:, :self.p0] = a[:self.p0, self.p0:].T
        return a, b


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


def fit(ace, ref, bins: Bins, train: torch.Tensor, per_bin: int, batch: int, ridges, seed: int,
        single: bool = False, prior: JointDensity = None) -> dict:
    """{ridge: dict(w [bins, P], fit_loss)}: one accumulation of G^T G per bin, one solution per ridge. With `single`
    the products of a batch in float32 (their sum and the solution stay in float64). With `prior` (a reference energy)
    the target is what it leaves."""
    g = torch.Generator().manual_seed(seed)  # the draws on the CPU: the same samples on every device
    device = torch.device("cpu") if DEVICE.type == "mps" else DEVICE  # the sums in float64: not on mps
    w = {r: [] for r in ridges}
    fit_loss = {r: [] for r in ridges}
    for k in range(len(bins.centres)):
        t0 = time.perf_counter()
        gram = Gram(ace, device, single)
        yy, rows = 0.0, 0
        lo, hi = bins.t_range(k)
        for _ in range(0, per_bin, batch):
            x0 = train[torch.randint(len(train), (batch,), generator=g)]
            t = lo + (hi - lo) * torch.rand(batch, dtype=torch.float64, generator=g)
            xt, target, _ = draw(ref, x0, t, g)
            if prior is not None:
                target = target - prior.scaled_score(xt.float(), ref.s(t)).double()
            gram.add(xt, target)
            yy += float((target ** 2).sum())
            rows += batch
        a, b = gram.total()
        del gram  # room for the solution
        for r in ridges:
            m = a.clone()
            m.diagonal().add_(r * float(a.diagonal().mean()))
            wk = -torch.cholesky_solve(b[:, None], torch.linalg.cholesky(m))[:, 0]
            del m
            w[r].append(wk.cpu())
            fit_loss[r].append(float((wk @ a @ wk + 2 * wk @ b + yy) / rows))
        losses = ", ".join(f"{fit_loss[r][-1]:.3f}" for r in ridges)
        print(f"bin {k + 1:2d}/{len(bins.centres)}: t in [{lo:.4f}, {hi:.4f}], s in [{bins.edges[k].exp():.2e}, "
              f"{bins.edges[k + 1].exp():.2e}]: fit loss {losses} (ridge {', '.join(f'{r:g}' for r in ridges)}; "
              f"zero score {yy / rows:.3f}), {time.perf_counter() - t0:.0f} s", flush=True)
    return {r: dict(w=torch.stack(w[r]), fit_loss=fit_loss[r]) for r in ridges}


def evaluate(ace, ref, bins: Bins, w: torch.Tensor, valid: torch.Tensor, compare, batch: int, t_min: float,
             interp: str = "linear", prior: JointDensity = None) -> dict:
    """Losses [N] per model on one fixed draw per validation pose, and the times."""
    g = torch.Generator().manual_seed(0)
    t = t_min + (1 - t_min) * torch.rand(len(valid), dtype=torch.float64, generator=g)
    xt, target, std = draw(ref, valid, t, g)
    out = {"t": t, "zero": (target ** 2).sum((1, 2))}
    energy = []
    for i in range(0, len(valid), batch):
        gm = jacobian(ace, xt[i:i + batch])  # [b, 3J, P]
        gm = (gm.cpu() if gm.device.type == "mps" else gm).double()
        pred = -torch.einsum("bkp,bp->bk", gm, bins.weights(w, t[i:i + batch], interp).to(gm.device)).cpu()
        if prior is not None:
            pred = pred + prior.scaled_score(xt[i:i + batch].float(), ref.s(t[i:i + batch])).double().flatten(1)
        energy.append(((pred - target[i:i + batch].reshape(len(pred), -1)) ** 2).sum(1))
    out["ACE energy"] = torch.cat(energy)
    for name, model in compare.items():
        losses = []
        with torch.no_grad():
            for i in range(0, len(valid), batch):
                sl = slice(i, i + batch)
                scaled = model.score(xt[sl].float().to(DEVICE), t[sl].float().to(DEVICE)).cpu().double()
                scaled = scaled * std[sl][..., None]
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
    parser.add_argument("--device", default="cpu",
                        help="cpu, cuda:N or mps (mps needs --single; its sums are on the CPU in float64)")
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
    parser.add_argument("--all-pairs", type=int, nargs="*", default=[], choices=(1, 2, 3),
                        help="with --wigner or --pairs: D^l(G_i^T G_k) of the other pairs of joints, of these degrees")
    parser.add_argument("--products", action="store_true",
                        help="with --wigner or --pairs: the products of the entries of x_i and x_k, all pairs")
    parser.add_argument("--single", action="store_true",
                        help="the products of every batch in float32 (a large basis on a GPU with slow float64)")
    parser.add_argument("--joint-density", type=float, nargs="+", default=None, metavar="N S0",
                        help="the density of every joint's own rotation as a reference energy (joint_density.py): N "
                             "training poses, then the bandwidth S0 (a Brownian time), or one per joint")
    parser.add_argument("--load", default=None, help="a .pt file of an earlier fit: evaluate it, fit nothing")
    parser.add_argument("--interp", choices=("linear", "nearest"), default="linear", help="w between bins")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device.startswith("mps") and not args.single:
        parser.error("--device mps needs --single (no float64 there).")
    global DEVICE
    DEVICE = torch.device(args.device)

    ref = load_model(Path(args.reference).expanduser(), device="cpu", weights="weights_ema").eval()
    ace = AceFeatures(smooth=not args.plain)
    if args.one_body:
        ace = WithOneBody(ace, args.one_body)
    if args.wigner or args.pairs:
        ace = harmonic_basis(vars(args))
    elif args.all_pairs or args.products:
        parser.error("--all-pairs and --products need --wigner or --pairs.")
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
    prior = None
    if args.load and "prior" in saved:
        prior = JointDensity(saved["prior"]["centres"], saved["prior"]["s0"])
    elif args.joint_density:
        prior = JointDensity.from_poses(train, int(args.joint_density[0]), args.joint_density[1:], args.seed)
    if results is None:
        results = fit(ace, ref, bins, train, args.per_bin, args.batch, args.ridge, args.seed, args.single, prior)
    compare = {Path(r).expanduser().parent.name: load_model(Path(r).expanduser(), device=args.device,
                                                             weights="weights_ema").eval() for r in args.compare}
    for n_ridge, (ridge, result) in enumerate(results.items()):
        report(ace, ref, bins, ridge, result, valid, compare if n_ridge == 0 else {}, args, len(results) > 1, prior)


def report(ace, ref, bins, ridge, result, valid, compare, args, several: bool, prior=None) -> None:
    """Evaluate one fit on the validation draws, print its table (and the networks' with the first), save it: the
    weights before the evaluation, so that an error there does not lose the fit."""
    out = Path(args.out).expanduser() if args.out and not args.load else None
    if out is not None:
        out = out.with_name(f"{out.stem}_r{ridge:g}{out.suffix}") if several else out
        out.parent.mkdir(parents=True, exist_ok=True)
    settings = dict(vars(args), ridge=ridge)

    def save(table):
        extra = {} if prior is None else dict(prior=dict(centres=prior.centres.cpu(), s0=prior.s0.cpu()))
        torch.save(dict(w=result["w"], edges=bins.edges, fit_loss=result["fit_loss"], table=table,
                        settings=settings, descriptors=ace.dim, **extra), out)

    if out is not None:
        save({})
    losses = evaluate(ace, ref, bins, result["w"], valid, compare, args.batch, ref.loss_t_min, args.interp, prior)
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
    if out is not None:
        save(table)
        out.with_suffix(".json").write_text(json.dumps(dict(table=table, fit_loss=result["fit_loss"],
                                                            settings=settings), indent=1))
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
