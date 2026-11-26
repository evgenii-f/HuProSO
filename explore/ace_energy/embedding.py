"""A per-joint ACE energy with a Finnis-Sinclair embedding, trained by denoising score matching (the step after the
linear energy of fit.py; docs/experiments/2026-11-23_ace_descriptors.md). From the root of the repository:

    python explore/ace_energy/embedding.py --check                     # the energy at init is the linear fit
    python explore/ace_energy/embedding.py --init ~/data/experiments/ace/energy_b_smooth_w123_p123.pt \\
        --device cuda:0 --out ~/data/experiments/ace/energy_fs.pt

Model. Every feature B_f of fit.py's basis (smooth positions, D^l(x_j), pair harmonics) belongs to the joints whose
rotations it depends on most directly (`owners`): a pair of points to the joints that place the two points, a triple
to the joint that places its centre, one-body terms to their joint, a pair harmonic to its two joints. Every joint j
has K = 2 densities, linear in its features, and an energy through the embedding F:

    phi_j^(k)(x, t) = sum_{f in N(j)} c_jf^(k)(t) B_f(x),     E~_t(x) = sum_j [ phi_j^(1) + sqrt(1 + phi_j^(2)^2) - 1 ]

(a smooth Finnis-Sinclair-like embedding: quadratic near 0, |phi| far from it). E~_t = std(t) E_t, so the scaled score
of RSGM's loss is -grad E~_t, taken by autograd along the tangent directions x_j exp(eps e_a). The coefficients are
per bin of log s and interpolated linearly between bin centres, as in fit.py. Training: Adam on RSGM's loss (std
weighting, t uniform on [t_min, 1]), the noise drawn by the reference run's noising, as fit.py.

Init: phi^(1) from a linear fit of fit.py (--init, with its basis), each feature's coefficient on its first owner, and
phi^(2) small; then the energy starts as that linear fit (--check verifies it on the validation draws). A fit with a
joint density (fit.py --joint-density) keeps it as a fixed reference energy. "--embedding fs --k 1" has no embedding:
the linear energy, trained by Adam from the fit (the control of an embedding).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import (COMPARE, JOINT_PAIRS, OTHER_PAIRS, REFERENCE, ROTATIONS, SPLIT, SQUARE_DIM, TRIPLE_DIM,  # noqa: E402
                 Bins, WithHarmonics, draw, harmonic_basis)
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.ace_features import LEARNED  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.rsgm import exp_frame  # noqa: E402
from hp.models.so3_harmonics import PAIR_INDEX  # noqa: E402
from joint_density import JointDensity  # noqa: E402


def owners(basis: WithHarmonics) -> list:
    """For every feature of the basis, in its layout, the joints (indices into the 19 of x) it belongs to."""
    ace = basis.ace
    parents = ace.parents

    def owner(point):  # the joint whose rotation places the point; None for the root and the points fixed to it
        p = parents[point]
        return LEARNED.index(p) if p in LEARNED else None

    def both(*points):
        return sorted({o for o in (owner(p) for p in points) if o is not None})

    out = []
    pair_owners = [both(j, k) for j, k in ace.pairs]
    out += pair_owners * ace.radial  # T_1..T_radial, each a block over the pairs
    tri = [[owner(j)] if owner(j) is not None else both(k, q) for k, j, q in ace.triples]
    out += tri * (ace.angular * ace.triple_radial ** 2)
    if ace.twist_joints:
        out += [[LEARNED.index(j)] for j in ace.twist_joints] * (2 * ace.twist)
    for l in basis.wigner:  # D^l(x_j): [19, 2l+1, 2l+1]
        out += [[j] for j in range(19) for _ in range((2 * l + 1) ** 2)]
    for l in basis.pairs:  # D^l(G_j) of every joint
        out += [[j] for j in range(19) for _ in range((2 * l + 1) ** 2)]
    for l in basis.pairs:  # D^l(G_i^T G_k) of the pairs
        out += [[i, k] for i, k in PAIR_INDEX for _ in range((2 * l + 1) ** 2)]
    for l in basis.all_pairs:
        out += [[i, k] for i, k in OTHER_PAIRS for _ in range((2 * l + 1) ** 2)]
    if basis.products:
        out += [[i, k] for i, k in JOINT_PAIRS for _ in range(81)]
    out += [list(tr) for tr in basis.triple_joints for _ in range(TRIPLE_DIM)]
    out += [list(pair) for pair in basis.square_pairs for _ in range(SQUARE_DIM)]
    assert len(out) == basis.dim, (len(out), basis.dim)
    return out


class Embedded(torch.nn.Module):
    """E~_t(x) [B] (module doc): per-joint densities of the features and an embedding: "fs", the Finnis-Sinclair-like
    phi^(1) + sqrt(1 + phi^(2)^2) - 1 (+ the same for further densities), or "mlp", phi^(1) + MLP(phi^(2..K), u, e_j)
    with u = log s(t) / 4 and e_j a learned embedding of the joint, one MLP shared by the joints, its last layer zero
    at init (so the energy starts as phi^(1), the linear fit)."""

    def __init__(self, basis, bins: Bins, own: list, k: int = 2, embedding: str = "fs", hidden: int = 64,
                 joint_dim: int = 8):
        super().__init__()
        if embedding not in ("fs", "mlp"):
            raise ValueError(f"Unknown embedding '{embedding}'. Use 'fs' or 'mlp'.")
        self.basis, self.bins, self.embedding = basis, bins, embedding
        self.prior = None  # a reference energy (joint_density.JointDensity), added to the energy of the features
        if embedding == "mlp":
            self.joint_emb = torch.nn.Parameter(0.1 * torch.randn(19, joint_dim))
            self.mlp = torch.nn.Sequential(torch.nn.Linear(k - 1 + 1 + joint_dim, hidden), torch.nn.SiLU(),
                                           torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1))
            torch.nn.init.zeros_(self.mlp[-1].weight)
            torch.nn.init.zeros_(self.mlp[-1].bias)
        feat = [f for f, js in enumerate(own) for _ in js]
        joint = [j for js in own for j in js]
        self.register_buffer("feat", torch.tensor(feat))
        self.register_buffer("joint", torch.tensor(joint))
        self.register_buffer("first", torch.tensor([i == 0 for js in own for i in range(len(js))]))
        n = len(bins.centres)
        self.c = torch.nn.Parameter(torch.zeros(k, n, len(feat)))  # [K, bins, entries]
        self._dense = {}  # (bin, kind) -> (version of c, matrix): the coefficients of a bin, cached when c is fixed
        self.all_bins = None  # every bin in one product (None: on a GPU only; the CPU groups the poses by bin)
        # the products (x_i)_a (x_k)_b as a quadratic form: the entry of a product on owner j is a coefficient of
        # (x_j)_a' times (x_other)_b', A [K, 19 (j), 9 (a'), 171 (9 other + b')]; the other features as rows of a
        # matrix [K, P - 13851, 19]
        prod = basis.product_slice() if getattr(basis, "products", False) else slice(0, 0)
        f_all, j_all = torch.tensor(feat), torch.tensor(joint)
        is_prod = (f_all >= prod.start) & (f_all < prod.stop)
        self.quadratic = bool(is_prod.any())
        rows = torch.where(f_all >= prod.stop, f_all - (prod.stop - prod.start), f_all)
        self.register_buffer("lin_entry", torch.nonzero(~is_prod)[:, 0], persistent=False)
        self.register_buffer("lin_row", rows[~is_prod], persistent=False)
        self.register_buffer("lin_joint", j_all[~is_prod], persistent=False)
        q = f_all[is_prod] - prod.start
        pair, a, b = q // 81, (q % 81) // 9, q % 9
        pairs = torch.tensor(JOINT_PAIRS, dtype=torch.long)
        i, k, own = pairs[pair, 0], pairs[pair, 1], j_all[is_prod]
        first = own == i
        self.register_buffer("quad_entry", torch.nonzero(is_prod)[:, 0], persistent=False)
        self.register_buffer("quad_joint", own, persistent=False)
        self.register_buffer("quad_row", torch.where(first, a, b), persistent=False)
        self.register_buffer("quad_col", torch.where(first, 9 * k + b, 9 * i + a), persistent=False)
        self.n_linear = basis.dim - (prod.stop - prod.start)

    def init_linear(self, w: torch.Tensor, scale2: float = 1e-3) -> None:
        """phi^(1) = the linear fit w [bins, P] on the first owner of every feature; phi^(2) small."""
        with torch.no_grad():
            self.c[0] = w[:, self.feat.cpu()].to(self.c) * self.first.to(self.c)
            self.c[1:] = scale2 * torch.randn_like(self.c[1:]) * w[:, self.feat.cpu()].abs().mean().to(self.c)

    def coefficients(self, t: torch.Tensor) -> torch.Tensor:
        """c(t) [K, B, entries], linear in log s between bin centres."""
        b = self.bins
        u = b.ref.s(t.detach().cpu().double()).log()  # in float64 on the CPU (mps has no float64)
        pos = ((u - b.centres[0]) / (b.centres[1] - b.centres[0])).clamp(0, len(b.centres) - 1)
        lo = pos.floor().long().clamp(max=len(b.centres) - 2)
        a = (pos - lo).to(self.c)[None, :, None]
        lo = lo.to(self.c.device)
        return (1 - a) * self.c[:, lo] + a * self.c[:, lo + 1]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        e = self.feature_energy(x, t)
        if self.prior is not None:
            e = e + self.prior.energy(x, self.bins.ref.s(t.detach().cpu().double()).to(e))
        return e

    def feature_energy(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """E~_t [B] without the reference energy (the joint density)."""
        if self.quadratic:  # the products as a quadratic form of the entries of x, not as columns
            phi = self.densities_split(self.basis(x, products=False), x.flatten(1), t)
            e = self.embed(phi, t, x.shape[0]).sum(-1)
        else:
            e = self.energy_of_features(self.basis(x), t)
        return e

    def dense(self, b: int) -> torch.Tensor:
        """[K, P, 19]: the coefficients of bin b, feature by joint (zero where a feature does not belong to a joint)."""
        cached = self._dense.get(b)
        if cached is not None and cached[0] == self.c._version and not self.c.requires_grad:
            return cached[1]
        k, p = self.c.shape[0], self.basis.dim
        kk = torch.arange(k, device=self.c.device).repeat_interleave(len(self.feat))
        m = torch.zeros(k, p, 19, device=self.c.device, dtype=self.c.dtype).index_put(
            (kk, self.feat.repeat(k), self.joint.repeat(k)), self.c[:, b].reshape(-1))
        if not self.c.requires_grad:
            self._dense[b] = (self.c._version, m)
        return m

    def _matrices(self, b: int):
        """(L [K, P - 13851, 19], A [K, 19 * 9, 171]) of bin b: the linear and the quadratic part."""
        cached = self._dense.get((b, "split"))
        if cached is not None and cached[0] == self.c._version and not self.c.requires_grad:
            return cached[1]
        k, dev, dt = self.c.shape[0], self.c.device, self.c.dtype
        c = self.c[:, b]  # [K, entries]

        def ks(n):
            return torch.arange(k, device=dev).repeat_interleave(n)

        n_lin, n_quad = len(self.lin_entry), len(self.quad_entry)
        lin = torch.zeros(k, self.n_linear, 19, device=dev, dtype=dt).index_put(
            (ks(n_lin), self.lin_row.repeat(k), self.lin_joint.repeat(k)), c[:, self.lin_entry].reshape(-1))
        quad = torch.zeros(k, 19, 9, 171, device=dev, dtype=dt).index_put(
            (ks(n_quad), self.quad_joint.repeat(k), self.quad_row.repeat(k), self.quad_col.repeat(k)),
            c[:, self.quad_entry].reshape(-1)).reshape(k, 19 * 9, 171)
        if not self.c.requires_grad:
            self._dense[(b, "split")] = (self.c._version, (lin, quad))
        return lin, quad

    def _matrices_range(self, b0: int, b1: int):
        """(L [n, K, P - 13851, 19], A [n, K, 19 * 9, 171]) of the bins b0 ... b1 (n of them), built at once."""
        cached = self._dense.get((b0, b1, "range"))
        if cached is not None and cached[0] == self.c._version and not self.c.requires_grad:
            return cached[1]
        k, dev, dt = self.c.shape[0], self.c.device, self.c.dtype
        n = b1 - b0 + 1
        c = self.c[:, b0:b1 + 1].transpose(0, 1)  # [n, K, entries]

        def index(*cols):  # every bin and density with the entries' coordinates cols
            e = len(cols[0])
            nb = torch.arange(n, device=dev).repeat_interleave(k * e)
            kk = torch.arange(k, device=dev).repeat_interleave(e).repeat(n)
            return (nb, kk) + tuple(v.repeat(n * k) for v in cols)

        lin = torch.zeros(n, k, self.n_linear, 19, device=dev, dtype=dt).index_put(
            index(self.lin_row, self.lin_joint), c[:, :, self.lin_entry].reshape(-1))
        quad = torch.zeros(n, k, 19, 9, 171, device=dev, dtype=dt).index_put(
            index(self.quad_joint, self.quad_row, self.quad_col),
            c[:, :, self.quad_entry].reshape(-1)).reshape(n, k, 19 * 9, 171)
        if not self.c.requires_grad:
            self._dense[(b0, b1, "range")] = (self.c._version, (lin, quad))
        return lin, quad

    def densities_split(self, features: torch.Tensor, m: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """phi [K, B, 19] from the features without the products [B, P - 13851] and the entries m [B, 171] of x."""
        lo, a = self._bin_weights(t)
        mj = m.reshape(len(m), 19, 9)
        all_bins = self.all_bins if self.all_bins is not None else features.is_cuda
        b0, b1 = int(lo.min()), int(lo.max()) + 1
        if all_bins and b1 - b0 > 1:  # several noise levels on a GPU: every bin used in one product, no groups
            lin, quad = self._matrices_range(b0, b1)
            n, k, p = lin.shape[0], lin.shape[1], lin.shape[2]
            w = torch.zeros(len(lo), n, dtype=features.dtype)
            rows = torch.arange(len(lo))
            w[rows, lo - b0] = (1 - a).to(w)
            w[rows, lo + 1 - b0] = a.to(w)
            w = w.to(features.device)
            y = (features @ lin.permute(2, 0, 1, 3).reshape(p, n * k * 19)).reshape(-1, n, k, 19)
            h = (m @ quad.permute(3, 0, 1, 2).reshape(171, n * k * 171)).reshape(-1, n, k, 19, 9)
            y = y + (h * mj[:, None, None]).sum(-1)
            return torch.einsum("bnkj,bn->kbj", y, w)
        mats = {}  # every bin built once per call

        def part(f, mm, mmj, g):
            if g not in mats:
                mats[g] = self._matrices(g)
            lin, quad = mats[g]
            h = (mm @ quad.transpose(1, 2)).reshape(quad.shape[0], len(mm), 19, 9)  # [K, n, 19, 9]
            return f @ lin + (h * mmj[None]).sum(-1)

        return self._grouped(lo, a, features, lambda idx, g: part(features[idx], m[idx], mj[idx], g))

    def _bin_weights(self, t: torch.Tensor):
        b = self.bins
        u = b.ref.s(t.detach().cpu().double()).log()  # in float64 on the CPU (mps has no float64)
        pos = ((u - b.centres[0]) / (b.centres[1] - b.centres[0])).clamp(0, len(b.centres) - 1)
        lo = pos.floor().long().clamp(max=len(b.centres) - 2)
        return lo, pos - lo

    def _grouped(self, lo, a, features, part) -> torch.Tensor:
        """[K, B, 19]: (1 - a) part(bin lo) + a part(bin lo + 1), one call per group of poses with the same lo."""
        dev = features.device
        groups = lo.unique().tolist()
        if len(groups) == 1:
            g, idx = groups[0], torch.arange(len(lo), device=dev)
            w = a.to(features)[None, :, None]
            return (1 - w) * part(idx, g) + w * part(idx, g + 1)
        parts, order = [], []
        for g in groups:
            idx = torch.nonzero(lo == g)[:, 0]
            w = a[idx].to(features)[None, :, None]
            idx_d = idx.to(dev)
            parts.append((1 - w) * part(idx_d, g) + w * part(idx_d, g + 1))
            order.append(idx)
        inverse = torch.argsort(torch.cat(order)).to(dev)
        return torch.cat(parts, dim=1)[:, inverse]

    def densities(self, features: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """phi [K, B, 19] of all the features [B, P] at times t [B]: the coefficients linear in log s between the bin
        centres, as coefficients(), by one matrix product per pair of neighbouring bins used."""
        lo, a = self._bin_weights(t)
        mats = {}

        def part(idx, g):
            if g not in mats:
                mats[g] = self.dense(g)
            return features[idx] @ mats[g]

        return self._grouped(lo, a, features, part)

    def energy_of_features(self, features: torch.Tensor, t: torch.Tensor, per_joint: bool = False) -> torch.Tensor:
        """E~_t [B] (or [B, 19] per joint) from the features B(x) [B, P] of the basis."""
        phi = self.densities(features, t)  # [K, B, 19]
        energy = self.embed(phi, t, features.shape[0])
        return energy if per_joint else energy.sum(-1)

    def embed(self, phi: torch.Tensor, t: torch.Tensor, n: int) -> torch.Tensor:
        """The energy per joint [B, 19] of the densities phi [K, B, 19]."""
        if self.embedding == "fs":
            energy = phi[0] + (torch.sqrt(1 + phi[1:] ** 2) - 1).sum(0)
        else:
            u = (self.bins.ref.s(t.detach().cpu().double()).log() / 4).to(phi)  # the noise level, as RSGM's net_level
            inp = torch.cat([phi[1:].permute(1, 2, 0), u[:, None, None].expand(-1, 19, 1),
                             self.joint_emb[None].expand(n, -1, -1).to(phi)], dim=-1)  # [B, 19, K - 1 + 1 + E]
            energy = phi[0] + self.mlp(inp)[..., 0]
        return energy

    def compile(self) -> "Embedded":
        """torch.compile the energy of the features for the scores without a graph (sampling, the likelihood ODE);
        the scores that are differentiated again (DPS, training) stay eager: compile has no double backward."""
        self._compiled = torch.compile(self.feature_energy, dynamic=False)
        return self

    def scaled_score(self, x: torch.Tensor, t: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        """-grad E~_t [B, J, 3] along the tangent directions (the scaled score of RSGM's loss)."""
        compiled = getattr(self, "_compiled", None)
        energy = compiled if compiled is not None and not create_graph and not x.requires_grad else self.feature_energy
        with torch.enable_grad():
            delta = torch.zeros(x.shape[0], x.shape[1], 3, device=x.device, dtype=x.dtype, requires_grad=True)
            e = energy(exp_frame(x, delta), t).sum()
            (g,) = torch.autograd.grad(e, delta, create_graph=create_graph)
        if self.prior is None:
            return -g
        s = self.bins.ref.s(t.detach().cpu().double()).to(x.device)
        graph = x.requires_grad or create_graph  # DPS differentiates the score in x; the prior has no parameters
        return -g + self.prior.scaled_score(x, s, graph=graph and x.requires_grad)


def losses(model, xt, target, t, batch) -> torch.Tensor:
    out = []
    for i in range(0, len(xt), batch):
        sl = slice(i, i + batch)
        pred = model.scaled_score(xt[sl], t[sl]).detach().double().cpu()
        out.append(((pred - target[sl]) ** 2).sum((1, 2)))
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", default="~/data/experiments/ace/energy_b_smooth_w123_p123.pt",
                        help="a linear fit of fit.py; its basis (--wigner, --pairs) is taken from it")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--compare", nargs="*", default=COMPARE[:1])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding", choices=("fs", "mlp"), default="fs")
    parser.add_argument("--k", type=int, default=2, help="densities per joint")
    parser.add_argument("--hidden", type=int, default=64, help="width of the MLP embedding")
    parser.add_argument("--valid", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--check", action="store_true", help="only: the energy at init against the linear fit")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    saved = torch.load(Path(args.init).expanduser(), map_location="cpu", weights_only=True)
    s = saved["settings"]
    ref = load_model(Path(args.reference).expanduser(), device="cpu", weights="weights_ema").eval()
    basis = harmonic_basis(s, saved.get("triples"))
    bins = Bins(ref, ref.loss_t_min, s["bins"])
    assert torch.equal(saved["edges"], bins.edges)
    model = Embedded(basis, bins, owners(basis), k=args.k, embedding=args.embedding, hidden=args.hidden).to(device)
    model.init_linear(saved["w"])
    if "prior" in saved:  # the joint density of the fit, a fixed reference energy (not trained)
        model.prior = JointDensity(saved["prior"]["centres"], saved["prior"]["s0"]).to(device)
    print(f"{basis.dim} features, {model.c.numel():,} coefficients ({model.c.shape[2]} joint-feature entries x "
          f"{model.c.shape[1]} bins x K = {model.c.shape[0]})", flush=True)

    valid = load_reference("valid", ROTATIONS, SPLIT)
    valid = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(0))[:args.valid]]
    g = torch.Generator().manual_seed(0)  # the validation draws of fit.py
    tv = ref.loss_t_min + (1 - ref.loss_t_min) * torch.rand(len(valid), dtype=torch.float64, generator=g)
    xv, yv, stdv = draw(ref, valid, tv, g)
    xv_d, tv_d = xv.float().to(device), tv.float().to(device)

    def evaluate():
        loss = losses(model, xv_d, yv, tv_d, 500)
        k = bins.index(tv)
        return float(loss.mean()), [float(loss[k == i].mean()) for i in range(len(bins.centres))], float(loss.median())

    if args.check:
        mean, per, med = evaluate()
        print(f"init: valid loss {mean:.3f} (median {med:.3f}); the linear fit: {saved['table']['ACE energy']['mean']:.3f}")
        return

    compare = {}
    for r in args.compare:
        net = load_model(Path(r).expanduser(), device=args.device, weights="weights_ema").eval()
        with torch.no_grad():
            pred = torch.cat([net.score(xv_d[i:i + 500], tv_d[i:i + 500]).double().cpu() for i in range(0, len(xv), 500)])
        compare[Path(r).expanduser().parent.name] = float((((pred * stdv[..., None]) - yv) ** 2).sum((1, 2)).mean())
    print("networks on the same draws:", compare, flush=True)

    train = load_reference("train", ROTATIONS, SPLIT)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = torch.Generator().manual_seed(args.seed + 1)
    history = []
    mean, per, med = evaluate()
    print(f"step 0: valid loss {mean:.3f} (median {med:.3f})", flush=True)
    history.append(dict(step=0, valid=mean, median=med, per_bin=per))
    t0, running = time.perf_counter(), []
    for step in range(1, args.steps + 1):
        x0 = train[torch.randint(len(train), (args.batch,), generator=gen)]
        t = ref.loss_t_min + (1 - ref.loss_t_min) * torch.rand(args.batch, dtype=torch.float64, generator=gen)
        xt, target, _ = draw(ref, x0, t, gen)
        pred = model.scaled_score(xt.float().to(device), t.float().to(device), create_graph=True)
        loss = ((pred - target.float().to(device)) ** 2).sum((1, 2)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running.append(float(loss))
        if step % args.eval_every == 0 or step == args.steps:
            mean, per, med = evaluate()
            history.append(dict(step=step, valid=mean, median=med, per_bin=per, train=sum(running) / len(running)))
            print(f"step {step}: train {sum(running) / len(running):.3f}, valid loss {mean:.3f} (median {med:.3f}), "
                  f"{(time.perf_counter() - t0) / step:.2f} s/step", flush=True)
            running = []
    edges = [f"{bins.edges[i].exp():.0e}" for i in range(len(bins.centres))]
    print("| model | mean | " + " | ".join(f"s {e}" for e in edges) + " |")
    print(f"| {args.embedding.upper()} energy, K = {args.k} | {mean:.3f} | " + " | ".join(f"{v:.2f}" for v in per) + " |")
    lin = saved["table"]["ACE energy"]
    print(f"| linear energy (--init) | {lin['mean']:.3f} | " + " | ".join(f"{v:.2f}" for v in lin["per_bin"]) + " |")
    for name in saved["table"]:
        if name.startswith("rsgm"):
            r = saved["table"][name]
            print(f"| {name} | {r['mean']:.3f} | " + " | ".join(f"{v:.2f}" for v in r["per_bin"]) + " |")
    if args.out:
        out = Path(args.out).expanduser()
        torch.save(dict(c=model.c.detach().cpu(), state=model.state_dict(), edges=bins.edges, history=history,
                        settings=vars(args),
                        basis=dict(wigner=basis.wigner, pairs=basis.pairs)), out)
        out.with_suffix(".json").write_text(json.dumps(dict(history=history, settings=vars(args), compare=compare),
                                                       indent=1))
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
