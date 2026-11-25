"""How far generated poses go beyond the joint ranges of the training poses, for samples saved by generate.py --out. On
the CPU, from the root of the repository:

    python explore/ace_energy/joint_ranges.py ~/data/experiments/ace/generate_1000/energy_*.pt
    python explore/ace_energy/joint_ranges.py --samples 1000 --filter ~/data/experiments/ace/generate/*.pt

Every joint's rotation is split into the twist about the bone to its child and the swing, whose two coordinates are the
principal axes of the swing in the training poses. The range of a coordinate is its 0.05%..99.95% quantiles over 400,000
training poses. Printed for every file and, as a yardstick, the validation poses: the share of poses with at least one
joint outside its range, the mean excess beyond the limit of the values that are outside (deg), and the width ratio to
the training poses of the middle 68% and 99.5% of the coordinates (mean over joints and coordinates). With --filter also
the harness (Wasserstein, precision) of the poses inside the ranges, next to the same number of all poses: how much of a
model's gap comes from the poses outside.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import roma
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import ROTATIONS, SPLIT  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.evaluation.evaluate import EvalConfig, _coverage, _wasserstein  # noqa: E402
from hp.models.ace_features import LEARNED, AceFeatures  # noqa: E402

QUANTILES = (0.0005, 0.9995)
N_TRAIN = 400_000


def bone_axes() -> torch.Tensor:
    """[19, 3] the unit vector from every joint to its child (of several children the neck, else the first)."""
    ace = AceFeatures()
    parents, offsets = ace.parents, ace.offsets.double()
    axes = []
    for j in LEARNED:
        children = [c for c in range(len(parents)) if parents[c] == j]
        child = children[0] if len(children) == 1 else next((c for c in children if c == 12), children[0])
        axes.append(offsets[child] / offsets[child].norm())
    return torch.stack(axes)


def twist_and_swing(x: torch.Tensor, axes: torch.Tensor):
    """x_j = swing * R_axis(twist): the twist [N, 19] and the rotation vectors of the swings [N, 19, 3], in degrees."""
    q = roma.rotmat_to_unitquat(x.double())
    twist = 2 * torch.atan2((q[..., :3] * axes).sum(-1), q[..., 3])
    twist = torch.remainder(twist + np.pi, 2 * np.pi) - np.pi
    t = torch.cat([axes.expand(*q.shape[:-1], 3) * torch.sin(twist / 2)[..., None],
                   torch.cos(twist / 2)[..., None]], -1)
    s = roma.quat_product(q, roma.quat_conjugation(t))
    return torch.rad2deg(twist), torch.rad2deg(roma.unitquat_to_rotvec(s * torch.sign(s[..., 3:] + 1e-300)))


class Ranges:
    """The ranges of the training poses in the coordinates (twist, swing 1, swing 2) of every joint."""

    def __init__(self, train: torch.Tensor):
        self.axes = bone_axes()
        pick = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:N_TRAIN]
        poses = train[pick]
        _, swing = twist_and_swing(poses, self.axes)
        axes = []
        for j in range(19):
            c = swing[:50_000, j]
            axes.append(torch.linalg.svd(c - c.mean(0), full_matrices=False)[2][:2])
        self.swing_axes = torch.stack(axes)
        self.train = self.coordinates(poses)
        self.low, self.high = (torch.tensor(np.quantile(self.train.numpy(), q, axis=0)) for q in QUANTILES)

    def coordinates(self, x: torch.Tensor) -> torch.Tensor:
        """[N, 19, 3]"""
        twist, swing = twist_and_swing(x, self.axes)
        return torch.stack([twist, *(torch.einsum("njk,jk->nj", swing, self.swing_axes[:, i]) for i in range(2))], -1)

    def outside(self, c: torch.Tensor) -> torch.Tensor:
        """[N] whether the pose has a coordinate outside its range."""
        return ((c < self.low) | (c > self.high)).flatten(1).any(1)

    def row(self, name: str, x: torch.Tensor) -> None:
        c = self.coordinates(x)
        excess = torch.cat([(c - self.high)[c > self.high], (self.low - c)[c < self.low]])

        def width(tail):
            def span(v):
                v = v.numpy()
                return torch.tensor(np.quantile(v, 1 - tail, axis=0) - np.quantile(v, tail, axis=0))
            return float((span(c) / span(self.train)).mean())

        share = 100 * float(self.outside(c).double().mean())
        print(f"| {name} | {len(x)} | {share:.1f}% | {float(excess.mean()):.1f} | {width(0.16):.2f} | "
              f"{width(0.0025):.2f} |", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples_files", nargs="+", help=".pt files of generate.py --out")
    parser.add_argument("--samples", type=int, default=None, help="the first N of every file (default: all)")
    parser.add_argument("--filter", action="store_true", help="also the harness of the poses inside the ranges")
    args = parser.parse_args()
    train, valid = load_reference("train", ROTATIONS, SPLIT), load_reference("valid", ROTATIONS, SPLIT)
    ranges = Ranges(train)
    files = {Path(f).stem: torch.load(Path(f).expanduser(), weights_only=True) for f in args.samples_files}
    print("| model | poses | with a joint outside | mean excess (deg) | width ratio, 68% | 99.5% |\n"
          "|---|---|---|---|---|---|")
    ranges.row("validation poses", valid)
    for name, x in files.items():
        ranges.row(name, x[:args.samples])
    if args.filter:
        n = args.samples or 1000
        cfg = EvalConfig(n_generated=n, n_wasserstein=n, n_queries=n, repeats=1, n_likelihood=0)
        print("\n| model | poses | Wasserstein | precision |\n|---|---|---|---|")
        for name, x in files.items():
            bad = ranges.outside(ranges.coordinates(x))
            for what, y in (("all", x[:n]), ("inside the ranges", x[~bad][:n])):
                w = _wasserstein(valid.float(), y, cfg, np.random.default_rng(0), 19)
                c = _coverage(valid.float(), y, cfg, np.random.default_rng(0))
                print(f"| {name}, {what} | {len(y)} | {w['generated']['joint_mean_deg']['mean']:.2f} | "
                      f"{c['precision']['mean_deg']['mean']:.2f} |", flush=True)


if __name__ == "__main__":
    main()
