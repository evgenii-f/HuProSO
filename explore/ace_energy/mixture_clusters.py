"""Step 0 of a mixture of linear energies: would clusters of poses give a mixture a handle on the wrong legs of
completion? No energy is fitted. k-means (K clusters, k-means++ start) on the 171 entries of the 19 local rotations of
training poses; then, on the first 200 validation poses with the legs hidden (the completions of complete.py:
network, P+L, MLP; K = 10 each):

- where the truth and the hypotheses fall: the share of every row's hypotheses in the cluster of the truth, over all
  poses and the worst 20% (by the MLP's mean error);
- how much the clusters tell about the legs: the share of the variance of the leg entries explained by the cluster
  (R^2), with the cluster taken from the whole pose and from the upper body alone (the nearest centroid on the 13
  upper joints): the second is what a mixture can use when the legs are hidden;
- the clusters: size and the mean angle of a few joints (degrees from the rest pose).

On the CPU, from the root of the repository:

    python explore/ace_energy/mixture_clusters.py --k 4 8 --pose 189
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import ROTATIONS, SPLIT  # noqa: E402
from hp.evaluation.completion import KEEP, valid_poses  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402

ACE = Path("~/data/experiments/ace").expanduser()
ROWS = {"network": "diag2/inpaint_network_rank.pt", "P+L": "diag2/inpaint_products.pt",
        "MLP": "diag5/inpaint_mlp_pl_k8.pt"}
LEGS = [0, 1, 3, 4, 6, 7]  # LeftUpLeg, RightUpLeg, LeftLeg, RightLeg, LeftFoot, RightFoot (of the 19)
UPPER = [j for j in range(19) if j not in LEGS]
SHOW = {"hip L": 0, "hip R": 1, "knee L": 3, "knee R": 4, "spine0": 2, "arm L": 13, "arm R": 14, "elbow L": 15,
        "elbow R": 16}


def kmeans(z: torch.Tensor, k: int, iters: int, gen: torch.Generator) -> torch.Tensor:
    """Centroids [k, D] of the rows of z by Lloyd's iterations from a k-means++ start."""
    c = z[torch.randint(len(z), (1,), generator=gen)]
    for _ in range(k - 1):
        d = torch.cdist(z, c).min(1).values ** 2
        c = torch.cat([c, z[torch.multinomial(d / d.sum(), 1, generator=gen)]])
    for _ in range(iters):
        a = torch.cdist(z, c).argmin(1)
        c = torch.stack([z[a == i].mean(0) if (a == i).any() else c[i] for i in range(k)])
    return c


def entries(x: torch.Tensor, joints) -> torch.Tensor:
    return x[..., joints, :, :].flatten(-3)


def angle(x: torch.Tensor) -> torch.Tensor:
    tr = x.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def r2(z: torch.Tensor, a: torch.Tensor, k: int) -> float:
    """The share of the variance of z [N, D] explained by the labels a [N]."""
    total = ((z - z.mean(0)) ** 2).sum()
    within = sum(((z[a == i] - z[a == i].mean(0)) ** 2).sum() for i in range(k) if (a == i).sum() > 1)
    return float(1 - within / total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--n", type=int, default=200_000, help="training poses for k-means")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--pose", type=int, default=189)
    parser.add_argument("--save", default=str(ACE / "mixture_clusters.pt"))
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(0)
    train = load_reference("train", ROTATIONS, SPLIT)
    x = train[torch.randperm(len(train), generator=gen)[:args.n]].float()
    z, zl, zu = entries(x, range(19)), entries(x, LEGS), entries(x, UPPER)
    gt = valid_poses(n=1000)[:200][:, KEEP].float()
    hyp = {r: torch.load(ACE / f, weights_only=True)["occlusions"]["legs"] for r, f in ROWS.items()}
    worst = hyp["MLP"]["mpvpe"].double().mean(1) > hyp["MLP"]["mpvpe"].double().mean(1).quantile(0.8)
    saved = {}
    for k in args.k:
        c = kmeans(z, k, args.iters, gen)
        cu = c.reshape(k, 19, 3, 3)[:, UPPER].flatten(1)
        a = torch.cdist(z, c).argmin(1)
        au = torch.cdist(zu, cu).argmin(1)
        print(f"\n== K = {k}: legs R^2 by the cluster of the whole pose {r2(zl, a, k):.3f}, of the upper body alone "
              f"{r2(zl, au, k):.3f}; upper body R^2 {r2(zu, a, k):.3f}; the two labels agree on "
              f"{float((a == au).double().mean()):.2f} of the poses", flush=True)
        print("| cluster | share | " + " | ".join(SHOW) + " |")
        print("|---|---|" + "---|" * len(SHOW))
        for i in range(k):
            m = angle(x[a == i]).mean(0)
            print(f"| {i} | {float((a == i).double().mean()):.2f} | " + " | ".join(f"{float(m[j]):.0f}"
                                                                                for j in SHOW.values()) + " |")
        ag = torch.cdist(entries(gt, range(19)), c).argmin(1)  # [200], the cluster of the truth
        print("| row | hypotheses in the truth's cluster | worst 20% | in its cluster by the upper body alone |")
        print("|---|---|---|---|")
        out = {}
        for r, o in hyp.items():
            ah = torch.cdist(entries(o["poses"].float(), range(19)), c).argmin(-1)  # [200, 10]
            same = (ah == ag[:, None]).double().mean(1)
            # the hypotheses keep the truth's upper body; is the cluster of the upper body the hypothesis' cluster?
            ahu = torch.cdist(entries(o["poses"].float(), UPPER), cu).argmin(-1)
            follow = (ah == ahu).double().mean(1)
            out[r] = dict(clusters=ah, same=same)
            print(f"| {r} | {float(same.mean()):.2f} | {float(same[worst].mean()):.2f} | {float(follow.mean()):.2f} |")
        p = args.pose
        print(f"pose {p}: the truth in cluster {int(ag[p])}")
        for r, o in hyp.items():
            order = o["mpvpe"][p].argsort()
            print(f"  {r}: " + ", ".join(f"{float(o['mpvpe'][p, i]):.0f} mm -> {int(out[r]['clusters'][p, i])}"
                                         for i in order))
        saved[k] = dict(centroids=c, truth=ag, rows=out)
    torch.save(saved, args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
