"""Metrics of the PoseD-Flow paper (Nadar, Foti, Birdal, CVPR 2026; pose_generation.py and experiment/utils.py there).

    dNN   the distance of a generated pose to its nearest training pose: per joint the angle acos|<q, q'>| of the
          unit quaternions (half the rotation angle, in radians), averaged over the joints, minimum over the
          training poses. Their numbers are on 21 joints against their whole training set.
    sets  every metric is computed on 20 sets of 500 samples and reported as mean +- the half-width of the 95%
          confidence interval of the mean (Student t).

    FID   the Frechet distance between Gaussians fitted to the 22 SMPL joint positions [66] of the samples and of
          the data (their precomputed examples/gt_mean_var_v2.npz: muj, covj). The joints come from SMPL neutral with
          betas = 0, no global rotation and no translation (smplx joints[:, :22]). In square meters.
    APD   the mean over all pairs of samples of the mean over the 22 joints of the Euclidean distance, in cm.

The joint positions need only the rest skeleton of SMPL neutral (scripts/preprocess/convert_smpl_skeleton.py writes
it from the official .pkl): with betas = 0 smplx moves the rest joints rigidly along the kinematic tree.
"""
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import roma
import torch
from scipy import linalg, stats

from hp.data.amass_cfg import AmassJoints4Learning

SMPL_JOINTS = 22  # pelvis + the 21 body joints (the hands are left at rest and are not used)


@torch.no_grad()
def nearest_neighbour_distance(poses: torch.Tensor, train: torch.Tensor, query_chunk: int = 64,
                               train_chunk: int = 500_000) -> torch.Tensor:
    """dNN [Q] of poses [Q, J, 3, 3] against training poses [N, J, 3, 3] or unit quaternions [N, J, 4].
    The computation runs on the device of `train`."""
    device = train.device
    t = roma.rotmat_to_unitquat(train) if train.shape[-2:] == (3, 3) else train
    q = roma.rotmat_to_unitquat(poses.to(device, t.dtype))
    best = []
    for qc in q.split(query_chunk):
        m = torch.full((len(qc),), float("inf"), device=device, dtype=t.dtype)
        for tc in t.split(train_chunk):
            # sum over the joints of acos|<q, q'>|: einsum of the dot products, joint by joint
            dots = torch.einsum("qjc,njc->qnj", qc, tc).abs().clamp(max=1.0)
            m = torch.minimum(m, torch.acos(dots).mean(-1).min(1).values)
        best.append(m)
    return torch.cat(best).cpu()


def confidence(values) -> Dict[str, float]:
    """Mean and the half-width of the 95% confidence interval of the mean, as their tables."""
    v = np.asarray(values, dtype=np.float64)
    half = stats.sem(v) * stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else float("nan")
    return {"mean": float(v.mean()), "ci95": float(half)}


def repeated_sets(sample: Callable[[int, int], torch.Tensor], metric: Callable[[torch.Tensor], float],
                  sets: int = 20, size: int = 500) -> Dict[str, float]:
    """metric over `sets` sets of `size` poses; sample(size, k) gives the k-th set."""
    return confidence([metric(sample(size, k)) for k in range(sets)])


class SmplSkeleton:
    """Forward kinematics of the first 22 joints of SMPL with betas = 0 (the .npz of convert_smpl_skeleton.py)."""

    def __init__(self, path="~/data/body_models/smpl/smpl_neutral_skeleton.npz"):
        data = np.load(Path(path).expanduser(), allow_pickle=False)
        self.rest = torch.as_tensor(data["joints"][:SMPL_JOINTS], dtype=torch.float64)
        self.parents = [int(p) for p in data["parents"][:SMPL_JOINTS]]

    def body_rotations(self, poses: torch.Tensor) -> torch.Tensor:
        """[B, 19 or 21, 3, 3] -> the rotations of SMPL joints 1 ... 21 [B, 21, 3, 3]; for our 19 joints the toes
        (SMPL 10, 11) are the identity. The toes are leaves of the 22 joints: they move no joint position."""
        if poses.shape[1] == 21:
            return poses
        if poses.shape[1] != len(AmassJoints4Learning):
            raise ValueError(f"Expected 19 or 21 joints, got {poses.shape[1]}.")
        full = torch.eye(3, dtype=poses.dtype, device=poses.device).repeat(len(poses), 21, 1, 1)
        full[:, [j.value - 1 for j in AmassJoints4Learning]] = poses
        return full

    def joints(self, poses: torch.Tensor) -> torch.Tensor:
        """Joint positions [B, 22, 3] (meters) of poses [B, 19 or 21, 3, 3]."""
        rot = self.body_rotations(poses.double().cpu())
        b = len(rot)
        rest = self.rest
        global_rot = [torch.eye(3, dtype=torch.float64).expand(b, 3, 3)]  # no global orientation
        positions = [rest[0].expand(b, 3)]
        for j in range(1, SMPL_JOINTS):
            p = self.parents[j]
            positions.append(positions[p] + global_rot[p] @ (rest[j] - rest[p]))
            global_rot.append(global_rot[p] @ rot[:, j - 1])
        return torch.stack(positions, dim=1)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """As their calculate_frechet_distance: ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 sqrt(S1 S2))."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid(joints: torch.Tensor, reference_mean: np.ndarray, reference_cov: np.ndarray) -> float:
    """FID of joint positions [B, 22, 3] against the Gaussian of the data (muj [66], covj [66, 66])."""
    x = joints.reshape(len(joints), -1).double().numpy()
    return frechet_distance(reference_mean, reference_cov, x.mean(0), np.cov(x, rowvar=False))


def apd(joints: torch.Tensor) -> float:
    """Average pairwise distance in cm: the mean over pairs of the mean over joints of |x_i - x_j|."""
    j = joints.double().transpose(0, 1)  # [22, B, 3]
    d = torch.cdist(j, j).mean(0)  # [B, B]
    i, k = torch.triu_indices(len(d), len(d), offset=1)
    return float(d[i, k].mean()) * 100
