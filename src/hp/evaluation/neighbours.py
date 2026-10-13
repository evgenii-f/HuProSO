"""Nearest neighbours between sets of poses: the basis of recall and precision.

Recall: how close is a real pose to the nearest generated pose (are all real poses covered)?
Precision: how close is a generated pose to the nearest real pose (are the generated poses realistic)?
Both are distances to the nearest neighbour, in a streaming search that never holds more than
queries x chunk distances, so 100 000 candidates are no problem.

Distances are computed in float32 by default. Near zero the angle is only resolved down to about 0.03 degrees
(float32 rotation matrices are orthogonal to about 1e-7, and arccos is steep at 1); pass dtype=torch.float64
with float64 poses to resolve smaller distances.
"""
from typing import Dict

import numpy as np
import torch

from hp.evaluation.geodesic import flatten_joints, pose_distances


def nearest_distances(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    reduction: str = "mean",
    chunk: int = 20_000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Distance [Nq] of every query pose [Nq, J, 3, 3] to its nearest pose among `candidates` [Nc, J, 3, 3].

    The search is exact. `reduction` is how the joints are combined: "mean" (radians per joint) or "sum".
    """
    if queries.shape[1:] != candidates.shape[1:]:
        raise ValueError(f"Poses of different shapes: {tuple(queries.shape[1:])} and {tuple(candidates.shape[1:])}.")
    q, c = flatten_joints(queries.to(dtype)), flatten_joints(candidates.to(dtype))
    best = torch.full((len(q),), float("inf"), dtype=dtype)
    for start in range(0, len(c), chunk):
        best = torch.minimum(best, pose_distances(q, c[start : start + chunk], reduction).min(dim=1).values)
    return best


def summarize(distances_rad) -> Dict[str, float]:
    """Mean, median, 95th percentile (degrees) and the share of distances under 5 degrees."""
    deg = np.rad2deg(np.asarray(distances_rad, dtype=np.float64))
    return {
        "mean_deg": float(deg.mean()),
        "median_deg": float(np.median(deg)),
        "p95_deg": float(np.percentile(deg, 95)),
        "within_5deg": float((deg <= 5).mean()),
    }
