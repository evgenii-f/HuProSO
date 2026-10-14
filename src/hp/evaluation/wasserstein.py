"""Wasserstein distance between two sets of poses, and the noise floor that makes it readable.

The distance is optimal transport between the two sets (uniform weights) with the geodesic angle as the cost
(POT's exact solver, as in the HuProSO code). It is computed for every joint alone (rotations on SO(3)) and for
the whole pose, where the cost of a pair of poses is the sum of the angles of its joints. Radians.

Two finite samples of the *same* distribution already have a distance above zero, and the more joints the
larger it is. The noise floor is the distance of two disjoint samples of the real poses, which is the smallest
distance a finite sample can have. It is not a goal for a model: poses of one recording are strongly
correlated (neighbouring frames are nearly identical), so a model cannot be as close to a test set as that.
A contrast set of real poses from *other* subjects shows what "as good as real unseen data" means.
"""
from typing import Callable, Dict, Optional

import numpy as np
import ot
import torch

from hp.evaluation.geodesic import flatten_joints, joint_angles


def wasserstein_distances(a: torch.Tensor, b: torch.Tensor, max_iter: int = 10_000_000,
                          step: Optional[Callable[[], object]] = None) -> Dict:
    """a [n, J, 3, 3], b [m, J, 3, 3] -> {"joint_sum_rad": float, "per_joint_rad": array [J]}.
    `step` is called after each of the J + 1 transport problems (e.g. a progress bar's update)."""
    step = step or (lambda: None)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"Poses of different shapes: {tuple(a.shape[1:])} and {tuple(b.shape[1:])}.")
    fa, fb = flatten_joints(a.double()), flatten_joints(b.double())
    p, q = np.full(len(fa), 1 / len(fa)), np.full(len(fb), 1 / len(fb))
    total = np.zeros((len(fa), len(fb)))
    per_joint = []
    for joint in range(fa.shape[1]):
        cost = joint_angles(fa, fb, joint).numpy()
        per_joint.append(ot.emd2(p, q, cost, numItermax=max_iter))
        step()
        total += cost
    joint_sum = float(ot.emd2(p, q, total, numItermax=max_iter))
    step()
    return {"joint_sum_rad": joint_sum, "per_joint_rad": np.array(per_joint)}


def wasserstein_with_floor(reference: torch.Tensor, generated: torch.Tensor, n: int, rng: np.random.Generator,
                           contrast: Optional[torch.Tensor] = None, step: Optional[Callable[[], object]] = None) -> Dict:
    """Distance of `n` generated poses to `n` real ones, and the noise floor: the distance of two disjoint
    samples of `n` real poses. With `contrast` (real poses of other subjects) its distance to the real poses
    is given too. The sample size is reduced if there are too few poses; it is in the result.
    `step` is called after every transport problem: (J + 1) per pair of samples, 2 or 3 pairs."""
    n = min(n, len(reference) // 2, len(generated), *([] if contrast is None else [len(contrast)]))
    order = rng.permutation(len(reference))
    a, b = reference[order[:n]], reference[order[n : 2 * n]]
    sample = generated[rng.choice(len(generated), n, replace=False)]
    out = {"n": n, "generated": wasserstein_distances(sample, a, step=step), "floor": wasserstein_distances(b, a, step=step)}
    if contrast is not None:
        out["contrast"] = wasserstein_distances(contrast[rng.choice(len(contrast), n, replace=False)], a, step=step)
    return out
