"""Would D^2(x_i) (x) D^1(x_k) terms help a linear fit? A test without a new fit: for every pair of joints (i, k) the
residual of the fit (target minus its scaled score) is regressed on the gradients of the pair's features, on training
poses, and the drop of the loss is measured on validation poses. On the CPU, from the root of the repository:

    python explore/ace_energy/pair_test.py --fit ~/data/experiments/ace/energy_b_rich_products_jd_r0.001.pt

Features of a pair, uncontracted: D^2(x_i)_ab D^1(x_k)_cd and D^1(x_i)_ab D^2(x_k)_cd, 450 numbers ("full"), or the
quadratic forms of the three principal local coordinates of one joint times the coordinates of the other, 36 numbers
("compressed", fit.py --squares). The features depend on x_i and x_k only, so only the 6 rows (three tangent directions
of the two joints) of a pose are not zero. Ridge: relative to the mean diagonal. The drop per pair is in units of the
loss (the sum of squares over the 57 directions) at one noise level. Pairs overlap, so drops are not additive: the sum
is a rough upper bound of what the pairs give together. The same test of the first-order terms (D^1 (x) D^1), which a
fit with --products already has, shows what "nothing left" looks like; on a fit without products it is a control.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.func import jacfwd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import HAT, JOINT_PAIRS, ROTATIONS, SPLIT, triple_coordinates  # noqa: E402
from hp.data.amass_cfg import AmassJoints4Learning  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.so3_harmonics import wigner_d  # noqa: E402
from inspect_energy import energy_predictions, load_energy, stratified_draws  # noqa: E402

NAMES = [j.name for j in AmassJoints4Learning]


def generators(l: int) -> torch.Tensor:
    """[3, 2l+1, 2l+1]: the derivatives of D^l(exp(eps e_a)) at eps = 0."""
    from hp.models.rsgm import exp_frame
    eye = torch.eye(3, dtype=torch.float64)
    return jacfwd(lambda d: wigner_d(exp_frame(eye[None], d[None])[0], l))(torch.zeros(3, dtype=torch.float64)
                                                                           ).permute(2, 0, 1)


class Terms:
    """The values and derivatives of the blocks of one joint: {"d1": ([N, 9], [N, 3, 9]), "d2": ([N, 25], [N, 3, 25]),
    "u": ([N, 3], ...), "uu": ([N, 6], ...)}."""

    def __init__(self, x: torch.Tensor, triples: dict):
        x = x.double()
        self.blocks = {}
        for name, l in (("d1", 1), ("d2", 2)):
            d, g = wigner_d(x, l), generators(l)
            self.blocks[name] = (d.flatten(-2), torch.einsum("npq,aqr->napr", d, g).flatten(-2))
        self.x, self.triples = x, triples

    def local(self, j: int):
        u = torch.einsum("np,pa->na", self.x.flatten(-2) - self.triples["centre"][j].double(),
                         self.triples["axes"][j].double())
        dm = torch.einsum("npq,cqr->ncpr", self.x, HAT.double()).flatten(-2)
        du = torch.einsum("ncp,pa->nca", dm, self.triples["axes"][j].double())
        q = [(a, b) for a in range(3) for b in range(a, 3)]
        uu = torch.stack([u[:, a] * u[:, b] for a, b in q], -1)
        duu = torch.stack([du[..., a] * u[:, None, b] + u[:, None, a] * du[..., b] for a, b in q], -1)
        return (u, du), (uu, duu)


def rows(first, second):
    """The feature gradients of a product of two blocks: for the two joints [2 * 3 N rows ..., F]."""
    (f, df), (s, ds) = first, second
    outer = torch.einsum("nap,nq->napq", df, s).flatten(-2)  # joint i: [N, 3, F]
    inner = torch.einsum("np,naq->napq", f, ds).flatten(-2)  # joint k
    return torch.cat([outer.reshape(-1, outer.shape[-1]), inner.reshape(-1, inner.shape[-1])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", default="~/data/experiments/ace/energy_b_rich_products_jd_r0.001.pt")
    parser.add_argument("--s", type=float, nargs="+", default=[1.2e-3, 5.4e-3, 2.5e-2],
                        help="noise levels (bin centres)")
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--valid", type=int, default=5000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    energy, ref, bins, info = load_energy(args.fit)
    pools = {s: load_reference(s, ROTATIONS, SPLIT) for s in ("train", "valid")}
    triples = triple_coordinates(pools["train"])
    print(f"{info}; {args.train} training and {args.valid} validation poses per level, ridge {args.ridge:g}")
    for s in args.s:
        k = int((torch.as_tensor(s).log().double() - bins.centres).abs().argmin())
        data = {}
        for seed, (split, n) in enumerate((("train", args.train), ("valid", args.valid))):
            d = stratified_draws(ref, bins, k, pools[split], n, 10 + seed)
            data[split] = (d["xt"], d["target"] - energy_predictions(energy, d, 250))
        joint = {split: [Terms(x[:, j], triples) for j in range(19)] for split, (x, _) in data.items()}
        base = float((data["valid"][1] ** 2).sum((1, 2)).mean())
        zero = {kind: {} for kind in ("full", "compressed", "first order")}
        for i, m in JOINT_PAIRS:
            for kind in zero:
                gains = {}
                store = {}
                for split, (x, r) in data.items():
                    a, b = joint[split][i], joint[split][m]
                    if kind == "compressed":
                        (ua, ub2), (va, vb2) = a.local(i), b.local(m)
                        phi = torch.cat([rows(ub2, va), rows(ua, vb2)], -1)
                    elif kind == "full":
                        phi = torch.cat([rows(a.blocks["d2"], b.blocks["d1"]),
                                         rows(a.blocks["d1"], b.blocks["d2"])], -1)
                    else:
                        phi = rows(a.blocks["d1"], b.blocks["d1"])
                    store[split] = (phi, torch.cat([r[:, i].reshape(-1), r[:, m].reshape(-1)]))
                phi, y = store["train"]
                gram = phi.T @ phi
                gram.diagonal().add_(args.ridge * float(gram.diagonal().mean()))
                w = torch.linalg.solve(gram, phi.T @ y)
                for split in ("train", "valid"):
                    p, t = store[split]
                    n = len(t) / 6
                    gains[split] = float(((t ** 2).sum() - ((t - p @ w) ** 2).sum()) / n)
                zero[kind][(i, m)] = gains
        print(f"\ns = {float(bins.centres[k].exp()):.1e}: validation residual loss of the fit {base:.2f}")
        print("| features | sum of the drops over 171 pairs, train / valid | pairs with a drop on valid "
              "| best pairs (valid) |\n|---|---|---|---|")
        for kind, res in zero.items():
            v = {p: g["valid"] for p, g in res.items()}
            best = sorted(v.items(), key=lambda t: -t[1])[:args.top]
            pos = sum(x > 0 for x in v.values())
            print(f"| {kind} | {sum(g['train'] for g in res.values()):.2f} / {sum(max(x, 0) for x in v.values()):.2f} "
                  f"| {pos} | " + ", ".join(f"{NAMES[a]}–{NAMES[b]} {x:.2f}" for (a, b), x in best) + " |",
                  flush=True)


if __name__ == "__main__":
    main()
