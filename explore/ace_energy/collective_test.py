"""Would many-body terms over all joints help a linear fit? A test without a new fit, as pair_test.py: the residual of
the fit (target minus its scaled score) is regressed on the gradients of candidate features, on training poses, and the
drop of the loss is measured on validation poses. On the CPU, from the root of the repository:

    python explore/ace_energy/collective_test.py --fit ~/data/experiments/ace/energy_b_rich_products_jd_r0.001.pt

Candidates:

- collective channels A_r = sum_j sum_a W_r,ja u_ja (u the three scaled principal local coordinates of a joint,
  fit.py triple_coordinates; W the K main components of the 57 numbers over training poses, every channel scaled to
  unit variance) and their products: A_r A_s A_t, r <= s <= t ("cubic K"), or all four ("quartic K"). "random": K random
  orthogonal directions instead of the main components, the control.
- the pair mixtures of every parent and child, log p(x_i, x_k) - log p(x_i) - log p(x_k) with p a mixture over 5000
  training poses (joint_density.py, the bandwidths of the fit's joint density times --scale), one coefficient per pair.

A drop is the mean over the bins below s = 0.04 (the loss gap of fits is the same mean). The far third: the validation
poses whose nearest training pose (all joints) is farthest.
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import HAT, REFERENCE, ROTATIONS, SPLIT, triple_coordinates  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.rsgm import PARENTS19  # noqa: E402
from inspect_energy import energy_predictions, load_energy, network_predictions, stratified_draws  # noqa: E402


class Channels:
    """K channels of the poses and their gradients, whitened on training poses."""

    def __init__(self, train: torch.Tensor, k: int, kind: str):
        self.t = triple_coordinates(train)
        z = self.local(train[::20].float())[0].flatten(1).double()
        mean = z.mean(0)
        cov = (z - mean).T @ (z - mean) / len(z)
        if kind == "pca":
            vals, vecs = torch.linalg.eigh(cov)
            w = vecs.flip(-1)[:, :k].T
        else:
            w = torch.linalg.qr(torch.randn(57, k, dtype=torch.float64,
                                            generator=torch.Generator().manual_seed(0)))[0].T
        self.w = w / (w @ cov @ w.T).diagonal().sqrt()[:, None]  # [K, 57], unit variance
        self.mean = mean

    def local(self, x: torch.Tensor):
        m = x.double().flatten(-2)
        u = torch.einsum("bjp,jpa->bja", m - self.t["centre"].double(), self.t["axes"].double())
        dm = torch.einsum("bjpq,cqr->bjcpr", x.double(), HAT.double()).flatten(-2)
        return u, torch.einsum("bjcp,jpa->bjca", dm, self.t["axes"].double())

    def __call__(self, x: torch.Tensor):
        """(A [B, K], dA [B, 57, K])."""
        u, du = self.local(x)
        a = (u.flatten(1) - self.mean) @ self.w.T
        w = self.w.reshape(len(self.w), 19, 3)
        return a, torch.einsum("bjca,kja->bjck", du, w).flatten(1, 2)


def monomials(a: torch.Tensor, da: torch.Tensor, order: int) -> torch.Tensor:
    """The gradients [57 B, P] of the products of `order` channels, r <= s <= ..."""
    idx = torch.tensor(list(itertools.combinations_with_replacement(range(a.shape[1]), order)))
    out = 0.0
    for pos in range(order):
        rest = torch.ones(a.shape[0], len(idx), dtype=a.dtype)
        for q in range(order):
            if q != pos:
                rest = rest * a[:, idx[:, q]]
        out = out + da[:, :, idx[:, pos]] * rest[:, None]
    return out.reshape(-1, len(idx))


class PairMixtures:
    """grad of log p(x_i, x_k) - log p(x_i) - log p(x_k), times std, for every parent and child, [57 B, pairs]."""

    def __init__(self, prior, scale: float):
        self.prior, self.scale = prior, scale
        self.pairs = [(PARENTS19[k], k) for k in range(19) if PARENTS19[k] >= 0]

    def __call__(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        p = self.prior
        tr = torch.bmm(x.float().flatten(-2).transpose(0, 1), p.centres)  # [J, B, n]
        keep = p.s0
        p.s0 = keep * self.scale
        k = p.kappa(s.float()).T  # [J, B]
        p.s0 = keep
        logit = 0.5 * k[..., None] * (tr - 3)
        std = torch.sqrt(-torch.expm1(-s)).float()
        out = torch.zeros(len(x), 57, len(self.pairs))

        def score(pi, j):  # d/d eps of the log of the mixture with responsibilities pi [B, n] along joint j
            mean = torch.einsum("bn,pn->bp", pi, p.centres[j]).reshape(-1, 3, 3)
            r = mean.transpose(-1, -2) @ x[:, j].float()
            v = torch.stack([r[..., 1, 2] - r[..., 2, 1], r[..., 2, 0] - r[..., 0, 2], r[..., 0, 1] - r[..., 1, 0]], -1)
            return std[:, None] * 0.5 * k[j][:, None] * v

        for e, (i, c) in enumerate(self.pairs):
            both = torch.softmax(logit[i] + logit[c], -1)
            for j in (i, c):
                alone = torch.softmax(logit[j], -1)
                # minus: the features are energies, and the score is minus their gradient; sign absorbed by the weights
                out[:, 3 * j:3 * j + 3, e] = score(both, j) - score(alone, j)
        return out.reshape(-1, len(self.pairs)).double()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", default="~/data/experiments/ace/energy_b_rich_products_jd_r0.001.pt")
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--valid", type=int, default=4000)
    parser.add_argument("--ridge", type=float, nargs="+", default=[1e-3, 1e-2])
    parser.add_argument("--s-max", type=float, default=0.04)
    parser.add_argument("--candidates", nargs="+",
                        default=["cubic:20:pca", "cubic:20:random", "cubic:30:pca", "quartic:12:pca", "pairs:1",
                                 "pairs:3", "pairs:10"])
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    energy, ref, bins, info = load_energy(args.fit)
    net = load_model(Path(REFERENCE).expanduser(), device="cpu", weights="weights_ema").eval()
    pools = {s: load_reference(s, ROTATIONS, SPLIT) for s in ("train", "valid")}
    ks = [k for k in range(len(bins.centres)) if float(bins.edges[k + 1].exp()) <= args.s_max]
    # the distance of every validation pose to training poses (chordal nearest, then the mean angle)
    v, t = pools["valid"], pools["train"]
    best, near = torch.full((len(v),), 9e9), torch.zeros(len(v), dtype=torch.long)
    for i in range(0, len(t), 200_000):
        d, j = torch.cdist(v.flatten(1), t[i:i + 200_000].flatten(1)).min(1)
        near = torch.where(d < best, j + i, near)
        best = torch.minimum(best, d)
    rel = (v.transpose(-1, -2) @ t[near]).double()
    dist = torch.rad2deg(torch.arccos(((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))).mean(1)
    limit = torch.quantile(dist, 2 / 3)

    feats = {}
    for name in args.candidates:
        kind, *rest = name.split(":")
        if kind == "pairs":
            feats[name] = PairMixtures(energy.prior, float(rest[0]))
        else:
            ch = Channels(t, int(rest[0]), rest[1])
            order = {"quadratic": 2, "cubic": 3, "quartic": 4}[kind]
            feats[name] = lambda x, s, ch=ch, order=order: monomials(*ch(x), order)
    print(f"{info}; {args.train} training and {args.valid} validation poses per bin, {len(ks)} bins", flush=True)
    drop = {n: {r: torch.zeros(2, dtype=torch.float64) for r in args.ridge} for n in feats}  # all, far third
    tdrop = {n: {r: 0.0 for r in args.ridge} for n in feats}
    gaps = torch.zeros(2, dtype=torch.float64)
    for k in ks:
        t0 = time.perf_counter()
        data = {}
        for seed, (split, n) in enumerate((("train", args.train), ("valid", args.valid))):
            d = stratified_draws(ref, bins, k, pools[split], n, 30 + seed)
            r = (d["target"] - energy_predictions(energy, d, 250)).flatten(1)
            data[split] = (d, r)
        g = torch.Generator().manual_seed(1000 * 31 + k)
        pick = torch.randperm(len(v), generator=g)[:args.valid]
        far = dist[pick] > limit
        assert torch.equal(v[pick], data["valid"][0]["x0"])
        d, r = data["valid"]
        base = (r ** 2).sum(1)
        ln = ((network_predictions(net, d) - d["target"]) ** 2).sum((1, 2))
        gaps += torch.stack([(base - ln).mean(), (base - ln)[far].mean()]) / len(ks)
        for name, f in feats.items():
            gram = rhs = 0.0
            x, rt, s = data["train"][0]["xt"], data["train"][1], data["train"][0]["s"]
            for i in range(0, len(x), 250):
                phi = f(x[i:i + 250], s[i:i + 250])
                gram = gram + phi.T @ phi
                rhs = rhs + phi.T @ rt[i:i + 250].reshape(-1)
            for rd in args.ridge:
                m = gram.clone()
                m.diagonal().add_(rd * float(m.diagonal().mean()))
                w = torch.linalg.solve(m, rhs)
                tdrop[name][rd] += float(2 * w @ rhs - w @ gram @ w) / len(x) / len(ks)
                pred = torch.cat([(f(d["xt"][i:i + 250], d["s"][i:i + 250]) @ w).reshape(-1, 57)
                                  for i in range(0, len(r), 250)])
                new = ((r - pred) ** 2).sum(1)
                drop[name][rd] += torch.stack([(base - new).mean(), (base - new)[far].mean()]) / len(ks)
        print(f"bin {k}: s {float(bins.centres[k].exp()):.1e}, {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"\ngap to the network {gaps[0]:.3f}, far third {gaps[1]:.3f} (mean over the bins)")
    print("| candidate | features | ridge | drop on train | drop on valid | drop on the far third |")
    print("|---|---|---|---|---|---|")
    for name, f in feats.items():
        size = f(pools["valid"][:2], torch.full((2,), 0.01, dtype=torch.float64)).shape[1]
        for rd in args.ridge:
            a, b = drop[name][rd]
            print(f"| {name} | {size} | {rd:g} | {tdrop[name][rd]:.3f} | {a:.3f} | {b:.3f} |")
    if args.save:
        torch.save(dict(gaps=gaps, drop={n: {str(r): x for r, x in v.items()} for n, v in drop.items()}, tdrop={
            n: {str(r): x for r, x in v.items()} for n, v in tdrop.items()}), args.save)


if __name__ == "__main__":
    main()
