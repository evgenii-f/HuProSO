"""Geodesic distances between poses, the building block of the evaluation.

A pose is a tensor [J, 3, 3] of joint rotations; sets of poses are [N, J, 3, 3]. The distance of two poses is
the geodesic angle (radians) between the rotations of a joint, averaged ("mean") or summed ("sum", the
definition of the HuProSO paper) over the joints. The angle comes from the trace: cos(angle) = (tr(A^T B) - 1) / 2,
and tr(A^T B) is the dot product of the flattened matrices, so a whole matrix of distances is a few matmuls.
"""
import torch


def flatten_joints(poses: torch.Tensor) -> torch.Tensor:
    """[N, J, 3, 3] -> [N, J, 9]"""
    return poses.reshape(poses.shape[0], poses.shape[1], 9)


def joint_angles(a: torch.Tensor, b: torch.Tensor, joint: int) -> torch.Tensor:
    """Angle [n, m] between every rotation of `joint` in a [n, J, 9] and in b [m, J, 9], from flattened poses."""
    trace = a[:, joint] @ b[:, joint].T
    return torch.arccos(((trace - 1) / 2).clamp(-1, 1))


def pose_distances(a: torch.Tensor, b: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Distance [n, m] between every pose of a [n, J, 9] and every pose of b [m, J, 9] (flattened poses)."""
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', not '{reduction}'.")
    total = sum(joint_angles(a, b, j) for j in range(a.shape[1]))
    return total / a.shape[1] if reduction == "mean" else total
