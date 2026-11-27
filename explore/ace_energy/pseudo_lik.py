"""A pairwise matrix-Fisher model of the 19 local rotations, fitted by pseudo-likelihood on clean poses (no noise levels,
no score matching). The energy

    E(x) = - sum_j <A_j, x_j> - sum_{j<k} x_j . B_jk . x_k        (x_j: the 9 entries of a local rotation)

is the basis of fit.py --no-descriptors --wigner 1 --products (D^1 of every joint and the 81 products of every pair,
14,022 coefficients) without the joint density. With v the 171 entries of a pose and W the symmetric coupling of the
blocks B_jk (zero blocks on the diagonal), E = -<a, v> - v.W.v / 2, and the conditional of one joint given the others
is exactly a matrix Fisher distribution, p(x_j | x_-j) ~ exp <F_j, x_j> with F_j = A_j + sum_k B_jk x_k. Its
normalizer c(F) (Haar measure of mass 1) depends only on the proper singular values s1 >= s2 >= |s3| of F:

    c(F) = int_{-1}^{1} 1/2 I0((s1 - s2)(1 - u)/2) I0((s1 + s2)(1 + u)/2) exp(s3 u) du,

computed by quadrature in theta (u = cos theta, theta = pi v^2, Gauss-Legendre in v), with the exponentially scaled
Bessel functions; its gradient is U diag(d log c / ds) V^T. The pseudo-likelihood sum_j [log c(F_j) - <F_j, x_j>] is
convex in (A, B); it is minimized by L-BFGS on --n training poses with a ridge on B, one fit per --ridge, the ridge
chosen by the pseudo-likelihood of the validation poses (8,601).

Tests of the chosen fit (validation), on the 200 poses of complete.py and their saved completions (network, P+L, MLP;
K = 10 each), every occlusion:

- holes: the share of each row's hypotheses whose energy lies below the truth's, over all poses and the worst 20% (by
  the mean error of P+L's hypotheses), as gt_energy.py --shares; pose 189 with the legs hidden;
- ranking: the error of the hypothesis of lowest energy among the network's 10 (and among the pool of 30), next to the
  network's mean and the choice by P+L at s = 0.02 (the energies saved by complete.py --rank), with paired intervals.

On the CPU, from the root of the repository:

    python explore/ace_energy/pseudo_lik.py --check                      # c(F) against Monte Carlo
    python explore/ace_energy/pseudo_lik.py --ridges 1e-4 1e-3 1e-2      # fits, choice, tests
    python explore/ace_energy/pseudo_lik.py --device cuda:0 --no-tests   # the fits only, on a GPU
    python explore/ace_energy/pseudo_lik.py --load ~/data/experiments/ace/pl/energy_pl.pt   # the tests of a saved fit
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose import bootstrap  # noqa: E402
from fit import ROTATIONS, SPLIT  # noqa: E402
from gt_energy import ROWS  # noqa: E402
from hp.evaluation.completion import KEEP, valid_poses  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402

ACE = Path("~/data/experiments/ace").expanduser()
J, D = 19, 171
NODES = 64  # max error of log c 2e-6 against 128 nodes, s up to 1e4
_V, _WV = np.polynomial.legendre.leggauss(NODES)
V = torch.tensor((_V + 1) / 2, dtype=torch.float64)  # in [0, 1]
WV = torch.tensor(_WV / 2, dtype=torch.float64)
THETA = torch.pi * V ** 2
WEIGHT = WV * 2 * torch.pi * V * torch.sin(THETA)  # d theta = 2 pi v dv; du = sin theta d theta
COS = torch.cos(THETA)


def proper_svd(f: torch.Tensor):
    """U, s, V of f [..., 3, 3] with det U = det V = 1, s1 >= s2 >= |s3| (s3 may be negative)."""
    u, s, vh = torch.linalg.svd(f)
    d = torch.sign(torch.det(u) * torch.det(vh))
    s = torch.cat([s[..., :2], s[..., 2:] * d[..., None]], -1)
    u = torch.cat([u[..., :2], u[..., 2:] * d[..., None, None]], -1)
    return u, s, vh.transpose(-1, -2)


def i0e(x: torch.Tensor) -> torch.Tensor:
    """exp(-x) I0(x) for x >= 0 with the right derivative i1e - i0e also at 0 (torch's is 0 there)."""
    y = torch.special.i0e(x)
    return y.detach() + (x - x.detach()) * (torch.special.i1e(x) - y).detach()


def log_c_of_s(s: torch.Tensor) -> torch.Tensor:
    """log c of proper singular values s [..., 3] (float64)."""
    a = (s[..., 0] - s[..., 1])[..., None] / 2
    b = (s[..., 0] + s[..., 1])[..., None] / 2
    lam = (s[..., 1] + s[..., 2])[..., None]
    w = 1 - COS.to(s.device)  # 1 - u
    f = 0.5 * i0e(a * w) * i0e(b * (2 - w)) * torch.exp(-lam * w)
    return s.sum(-1) + torch.log((f * WEIGHT.to(s.device)).sum(-1))


class LogC(torch.autograd.Function):
    """log c(F) [...] of F [..., 3, 3]; the gradient is the mean of the distribution, U diag(d log c / ds) V^T."""

    @staticmethod
    def forward(ctx, f):
        u, s, v = proper_svd(f.double())
        with torch.enable_grad():
            s = s.detach().requires_grad_(True)
            out = log_c_of_s(s)
            ds, = torch.autograd.grad(out.sum(), s)
        ctx.save_for_backward((u * ds[..., None, :]) @ v.transpose(-1, -2))
        return out.detach().to(f.dtype)

    @staticmethod
    def backward(ctx, g):
        mean, = ctx.saved_tensors
        return (g[..., None, None] * mean).to(g.dtype)


def log_c(f: torch.Tensor, chunk: int = 1_000_000) -> torch.Tensor:
    flat = f.reshape(-1, 3, 3)
    return torch.cat([LogC.apply(flat[i:i + chunk]) for i in range(0, len(flat), chunk)]).reshape(f.shape[:-2])


class PairwiseFisher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(D, dtype=torch.float64))
        self.m = torch.nn.Parameter(torch.zeros(D, D, dtype=torch.float64))
        block = torch.arange(D) // 9
        self.register_buffer("upper", (block[:, None] < block[None, :]).double())  # the blocks j < k

    def coupling(self) -> torch.Tensor:
        w = self.m * self.upper
        return w + w.T

    def fields(self, v: torch.Tensor) -> torch.Tensor:
        """F [N, 19, 3, 3] of the conditionals, of poses v [N, 171]."""
        return (self.a + v @ self.coupling()).reshape(-1, J, 3, 3)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """E [...] of poses x [..., 19, 3, 3] (up to a constant)."""
        v = x.reshape(-1, D).to(self.a)
        return (-(v @ self.a) - 0.5 * ((v @ self.coupling()) * v).sum(-1)).reshape(x.shape[:-3])

    def conditional_nll(self, x: torch.Tensor) -> torch.Tensor:
        """-log p(x_j | x_-j) [N, 19] (nats, Haar measure of mass 1)."""
        v = x.reshape(-1, D).to(self.a)
        f = self.fields(v)
        return log_c(f) - (f * v.reshape(-1, J, 3, 3)).sum((-1, -2))

    def ridge(self) -> torch.Tensor:
        return ((self.m * self.upper) ** 2).sum()


def fit(train: torch.Tensor, ridge: float, iters: int, init=None) -> PairwiseFisher:
    model = PairwiseFisher().to(train.device)
    if init is not None:
        model.load_state_dict(init.state_dict())
    opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=iters, history_size=20, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    calls = [0]
    t0 = time.time()

    def closure():
        opt.zero_grad()
        loss = model.conditional_nll(train).sum(1).mean() + 0.5 * ridge * model.ridge()
        loss.backward()
        calls[0] += 1
        if calls[0] % 25 == 0:
            print(f"    evaluation {calls[0]}: {float(loss):.4f} ({time.time() - t0:.0f} s)", flush=True)
        return loss

    opt.step(closure)
    return model


def check() -> None:
    """log c against Monte Carlo over Haar rotations, the mean against the gradient, and the quadrature against 4x
    the nodes."""
    import roma

    g = torch.Generator().manual_seed(0)
    r = roma.random_rotmat(400_000, dtype=torch.float64)
    print("| s (proper) | log c quadrature | log c Monte Carlo | max |mean - MC mean| |")
    print("|---|---|---|---|")
    for s in ([0.0, 0.0, 0.0], [1.0, 0.5, 0.2], [3.0, 2.0, -1.0], [6.0, 6.0, 6.0], [8.0, 1.0, -0.5]):
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=g))
        q = q * torch.sign(torch.det(q))
        f = (q * torch.tensor(s, dtype=torch.float64)) @ q.T  # U = q, V = q
        f.requires_grad_(True)
        lc = log_c(f[None])[0]
        mean, = torch.autograd.grad(lc, f)
        e = (r * f.detach()).sum((-1, -2))
        mc = torch.logsumexp(e, 0) - np.log(len(r))
        mc_mean = (torch.softmax(e, 0)[:, None, None] * r).sum(0)
        lc = lc.detach()
        print(f"| {s} | {float(lc):.5f} | {float(mc):.5f} | {float((mean - mc_mean).abs().max()):.4f} |")
    # the quadrature at large concentrations against 4x the nodes (the Monte Carlo cannot reach them)
    global WEIGHT, COS
    big = torch.tensor([[50.0, 40.0, 30.0], [500.0, 400.0, -300.0], [5000.0, 4000.0, 3000.0], [2e4, 50.0, -10.0],
                        [1e5, 9e4, 8e4], [3.0, 0.1, 0.0]], dtype=torch.float64)
    base = log_c_of_s(big)
    keep = WEIGHT, COS
    v, wv = np.polynomial.legendre.leggauss(4 * NODES)
    v, wv = torch.tensor((v + 1) / 2), torch.tensor(wv / 2)
    WEIGHT, COS = wv * 2 * torch.pi * v * torch.sin(torch.pi * v ** 2), torch.cos(torch.pi * v ** 2)
    fine = log_c_of_s(big)
    WEIGHT, COS = keep
    for s, a, b in zip(big.tolist(), base.tolist(), fine.tolist()):
        print(f"s {s}: log c {a:.6f}, with {4 * NODES} nodes {b:.6f}")


def shares(model, gt, files, occ) -> tuple:
    e_gt = model.energy(gt)  # [200]
    hyp = {r: model.energy(files[r][occ]["poses"].float()) for r in files}  # [200, 10]
    return e_gt, hyp


def tests(model) -> None:
    gt = valid_poses(n=1000)[:200][:, KEEP].double()
    files = {r: torch.load(ACE / f, weights_only=True)["occlusions"] for r, f in ROWS.items()}
    print("\n== holes at s = 0: the share of each row's hypotheses below the truth (worst 20%: by P+L's mean error)")
    print("| occlusion | network | worst 20% | P+L | worst 20% | MLP | worst 20% |")
    print("|---|---|---|---|---|---|---|")
    for occ in files["network"]:
        e_gt, hyp = shares(model, gt, files, occ)
        err = files["P+L"][occ]["mpvpe"].double().mean(1)
        worst = err > err.quantile(0.8)
        cells = []
        for r in files:
            below = (hyp[r] < e_gt[:, None]).double().mean(1)
            cells += [f"{float(below.mean()):.2f}", f"{float(below[worst].mean()):.2f}"]
        print(f"| {occ} | " + " | ".join(cells) + " |")
        if occ == "legs":
            p = 189
            legs = f"  pose {p}, legs: truth {float(e_gt[p]):.1f};"
            for r in files:
                order = files[r][occ]["mpvpe"][p].argsort()
                legs += f"\n    {r}: " + "; ".join(f"{float(files[r][occ]['mpvpe'][p, i]):.0f} mm {float(hyp[r][p, i]):.1f}"
                                                for i in order)
            print(legs)
    print("\n== ranking: MPVPE (mm) of the hypothesis of lowest energy; 200 poses, K = 10")
    print("| occlusion | network mean | network, lowest P+L E at s 0.02 | network, lowest PL E | pool of 30, lowest PL E "
          "| PL - P+L choice |")
    print("|---|---|---|---|---|---|")
    rows = []
    for occ in files["network"]:
        net = files["network"][occ]
        err = net["mpvpe"].double()
        pick = lambda e, m: m.gather(1, e.argmin(1, keepdim=True))[:, 0]  # noqa: E731
        by_pl = pick(net["energies"][1].double(), err)
        e_net = model.energy(net["poses"].float())
        by_ours = pick(e_net, err)
        pool_e = torch.cat([model.energy(files[r][occ]["poses"].float()) for r in files], 1)
        pool_err = torch.cat([files[r][occ]["mpvpe"].double() for r in files], 1)
        by_pool = pick(pool_e, pool_err)
        rows.append((err.mean(1), by_pl, by_ours, by_pool))
        print(f"| {occ} | {float(err.mean()):.2f} | {float(by_pl.mean()):.2f} | {float(by_ours.mean()):.2f} | "
              f"{float(by_pool.mean()):.2f} | {bootstrap((by_ours - by_pl).numpy())} |")
    m = [torch.stack(c).mean(0) for c in zip(*rows)]
    print(f"| **mean** | {float(m[0].mean()):.2f} | {float(m[1].mean()):.2f} | {float(m[2].mean()):.2f} | "
          f"{float(m[3].mean()):.2f} | {bootstrap((m[2] - m[1]).numpy())} |")
    print(f"  PL choice - network mean: {bootstrap((m[2] - m[0]).numpy())}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--ridges", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--n", type=int, default=100_000, help="training poses")
    parser.add_argument("--iters", type=int, default=300, help="L-BFGS iterations per fit")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--save", default=str(ACE / "pl" / "energy_pl.pt"))
    parser.add_argument("--load", default=None, help="only the tests of a saved fit")
    parser.add_argument("--no-tests", action="store_true", help="only the fits (where the completions are missing)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.check:
        check()
        return
    if args.load:
        saved = torch.load(Path(args.load).expanduser(), weights_only=True)
        model = PairwiseFisher()
        model.load_state_dict(saved["state"])
        print(f"fit {args.load}: ridge {saved['ridge']}")
        tests(model)
        return
    g = torch.Generator().manual_seed(0)
    train = load_reference("train", ROTATIONS, SPLIT)
    train = train[torch.randperm(len(train), generator=g)[:args.n]].double().to(args.device)
    valid = load_reference("valid", ROTATIONS, SPLIT).double().to(args.device)
    print(f"{len(train):,} training poses, {len(valid):,} validation poses; per-joint conditional NLL, nats", flush=True)
    with torch.no_grad():
        zero = PairwiseFisher().to(args.device).conditional_nll(valid).sum(1).mean()
    print(f"uniform (F = 0): {float(zero):.3f}")
    fits, prev = {}, None
    for ridge in sorted(args.ridges, reverse=True):  # strong to weak, each from the last
        t0 = time.time()
        print(f"ridge {ridge:g}:", flush=True)
        model = fit(train, ridge, args.iters, prev)
        with torch.no_grad():
            tr = model.conditional_nll(train[:20_000]).sum(1).mean()
            va = model.conditional_nll(valid)
        fits[ridge] = (model, float(va.sum(1).mean()))
        print(f"  train {float(tr):.3f}, valid {fits[ridge][1]:.3f}, |B| {float(model.ridge()) ** 0.5:.1f}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        print("  valid per joint: " + " ".join(f"{float(x):.2f}" for x in va.mean(0)), flush=True)
        prev = model
    best = min(fits, key=lambda r: fits[r][1])
    print(f"chosen by validation: ridge {best:g} (valid {fits[best][1]:.3f})")
    out = Path(args.save).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state={k: v.cpu() for k, v in fits[best][0].state_dict().items()}, ridge=best, valid={r: v for r, (_, v) in fits.items()},
                    n=args.n), out)
    print(f"saved {out}")
    if not args.no_tests:
        tests(fits[best][0].cpu())


if __name__ == "__main__":
    main()
