"""Two test rows of the completion cache compared pose by pose, by difficulty: per occlusion the poses split into five
equal groups by the mean of both rows' MPVPE mean on the pose (a difficulty that favours neither row, so no
regression to the mean), and in each group the mean difference (b - a) of MPVPE mean (with a 95% interval), of best
of K and of APD. Run from the root (the metrics of both rows must be in the cache, e.g. after pose_completion.ipynb):

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/test_rows_by_difficulty.py \
        "PoseD-Flow: their checkpoint" "PoseD-Flow: their checkpoint, D-Flow t17"

MPVPE in mm, APD in cm.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from hp.evaluation.completion import OCCLUSIONS, cache_path

CACHE = "~/data/experiments/completion/cache"


def plain(cache: Path, label: str, occlusion: str) -> dict:
    path = cache_path(cache, label, occlusion).with_name(f"{occlusion}_metrics.pt")
    return {key: value.double().numpy() for key, value in torch.load(path, weights_only=True)["plain"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", help="label of the reference row")
    parser.add_argument("b", help="label of the compared row")
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    cache = Path(args.cache).expanduser()
    print(f"b - a, b = {args.b}, a = {args.a}")
    print("| occlusion | difficulty (mm) | poses | Δ mean (mm) | Δ best of K (mm) | Δ APD (cm) |")
    print("|---|---|---|---|---|---|")
    for occlusion in OCCLUSIONS:
        a, b = plain(cache, args.a, occlusion), plain(cache, args.b, occlusion)
        mean_a, mean_b = a["mpvpe"].mean(1), b["mpvpe"].mean(1)
        difficulty, d = (mean_a + mean_b) / 2, mean_b - mean_a
        best, apd = b["mpvpe"].min(1) - a["mpvpe"].min(1), b["apd"] - a["apd"]
        edges = np.quantile(difficulty, np.linspace(0, 1, args.groups + 1))
        for low, high in zip(edges[:-1], edges[1:]):
            s = (difficulty >= low) & (difficulty <= high)
            print(f"| {occlusion} | {low:.0f}-{high:.0f} | {s.sum()} | {d[s].mean():+.2f} ± "
                  f"{1.96 * d[s].std() / np.sqrt(s.sum()):.2f} | {best[s].mean():+.2f} | {apd[s].mean():+.2f} |")
        print(f"| {occlusion} | all | {len(d)} | {d.mean():+.2f} ± {1.96 * d.std() / np.sqrt(len(d)):.2f} | "
              f"{best.mean():+.2f} | {apd.mean():+.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
