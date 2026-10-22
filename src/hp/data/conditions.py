"""Conditions of a pose: the positions of its joints (keypoints), as the HuProSO3 paper conditions on them.

    c = keypoints(poses)                  # [N, 22, 3], meters
    c = keypoints(poses, modality="2D")   # [N, 22, 2], the x and y of the 3D positions (as the paper's code)

Forward kinematics of the 22 body joints of SMPL with the fixed skeleton of the paper's code
(body_models/smpl_skeleton.npz) and no global rotation: the root is at the origin, the root rotation and the toe
rotations (joints 0, 10, 11, not in AmassJoints4Learning) are the identity. It gives the same positions as
hp.utils.skeleton_forward_kinematics.perform_forward_kinematics, but works on rotation matrices directly (no
round trip through axis-angle), in the dtype and on the device of the poses, and is differentiable.

Joints that the keypoints cannot see: the wrists (20, 21) and the head (15) have no child among the 22 joints, so
their rotations do not move any keypoint; the twist of the feet about the foot-toe axis neither.
"""
import os
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch

from hp.data.amass_cfg import AmassJoints4Learning
from hp.utils.paths import get_path

NUM_KEYPOINTS = 22
MODALITIES = {"3D": 3, "2D": 2}


@lru_cache(maxsize=None)
def _skeleton() -> Tuple[Tuple[int, ...], np.ndarray]:
    """(parents of the 22 joints, their bone offsets [22, 3] from the parent in the rest pose)."""
    path = os.path.join(get_path("body_models"), "smpl_skeleton.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. Take body_models/smpl_skeleton.npz from "
                                "https://github.com/wei-mao-2019/HisRepItself and put it into the 'body_models' folder.")
    skeleton = np.load(path)
    parents = tuple(int(p) for p in skeleton["parents"][:NUM_KEYPOINTS])
    rest = skeleton["p3d0"][0, :NUM_KEYPOINTS].astype(np.float64)
    offsets = rest - rest[[max(p, 0) for p in parents]]
    offsets[0] = 0  # the root is at the origin
    return parents, offsets


def condition_dim(modality: str = "3D") -> int:
    """The length of a flattened condition: 22 keypoints x 3 (or x 2)."""
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality '{modality}'. Use one of {sorted(MODALITIES)}.")
    return NUM_KEYPOINTS * MODALITIES[modality]


def keypoints(poses: torch.Tensor, modality: str = "3D") -> torch.Tensor:
    """Positions [N, 22, 3] (or [N, 22, 2]) in meters of poses [N, 19, 3, 3] (the joints of AmassJoints4Learning)."""
    condition_dim(modality)
    if poses.shape[1:] != (len(AmassJoints4Learning), 3, 3):
        raise ValueError(f"Keypoints need poses [N, {len(AmassJoints4Learning)}, 3, 3], got {tuple(poses.shape)}.")
    parents, offsets = _skeleton()
    offsets = torch.as_tensor(offsets, dtype=poses.dtype, device=poses.device)
    n = poses.shape[0]
    local = [None] * NUM_KEYPOINTS  # rotation of each joint relative to its parent; None: the identity
    for k, joint in enumerate(AmassJoints4Learning):
        local[joint.value] = poses[:, k]
    rotations, positions = [None] * NUM_KEYPOINTS, [poses.new_zeros(n, 3)] * NUM_KEYPOINTS
    for i in range(1, NUM_KEYPOINTS):
        p = parents[i]
        parent_rotation = rotations[p]  # None: the identity
        bone = offsets[i].expand(n, 3) if parent_rotation is None else parent_rotation @ offsets[i]
        positions[i] = positions[p] + bone
        rotations[i] = local[i] if parent_rotation is None else (parent_rotation if local[i] is None else parent_rotation @ local[i])
    out = torch.stack(positions, dim=1)
    return out if modality == "3D" else out[..., :2]
