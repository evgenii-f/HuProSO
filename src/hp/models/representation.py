"""Rotation representations a model can work in.

The dataset and the evaluation always use rotation matrices [B, J, 3, 3]. A model that works in another
space (6D, axis-angle, quaternion) converts with a Representation. Whatever a model outputs, `from_vector`
returns proper rotations (it projects onto SO(3)), so the evaluation never sees an invalid sample.
"""
from abc import ABC, abstractmethod
from typing import Dict

import roma
import torch
import torch.nn.functional as F


class Representation(ABC):
    name: str
    dim: int  # numbers per joint

    @abstractmethod
    def to_vector(self, rotmat: torch.Tensor) -> torch.Tensor:
        """[..., 3, 3] rotation matrices -> [..., dim]."""

    @abstractmethod
    def from_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """[..., dim] -> [..., 3, 3] proper rotations, for any input values."""

    def pose_to_vector(self, pose: torch.Tensor) -> torch.Tensor:
        """[B, J, 3, 3] -> [B, J * dim], the joints one after the other."""
        return self.to_vector(pose).flatten(start_dim=1)

    def pose_from_vector(self, vector: torch.Tensor, num_joints: int) -> torch.Tensor:
        """[B, J * dim] -> [B, J, 3, 3]."""
        return self.from_vector(vector.reshape(vector.shape[0], num_joints, self.dim))


class Matrix(Representation):
    """The 9 matrix entries. The nearest rotation (SVD) is taken when converting back."""

    name, dim = "matrix", 9

    def to_vector(self, rotmat):
        return rotmat.reshape(*rotmat.shape[:-2], 9)

    def from_vector(self, vector):
        return roma.special_procrustes(vector.reshape(*vector.shape[:-1], 3, 3))


class SixD(Representation):
    """The first two columns of the matrix (Zhou et al., 2019); Gram-Schmidt when converting back."""

    name, dim = "6d", 6

    def to_vector(self, rotmat):
        return rotmat[..., :, :2].reshape(*rotmat.shape[:-2], 6)

    def from_vector(self, vector):
        return roma.special_gramschmidt(vector.reshape(*vector.shape[:-1], 3, 2))


class AxisAngle(Representation):
    """The rotation vector (axis times angle); every vector is a valid rotation."""

    name, dim = "axis_angle", 3

    def to_vector(self, rotmat):
        return roma.rotmat_to_rotvec(rotmat)

    def from_vector(self, vector):
        return roma.rotvec_to_rotmat(vector)


class Quaternion(Representation):
    """The unit quaternion (x, y, z, w); it is normalised when converting back."""

    name, dim = "quaternion", 4

    def to_vector(self, rotmat):
        return roma.rotmat_to_unitquat(rotmat)

    def from_vector(self, vector):
        return roma.unitquat_to_rotmat(F.normalize(vector, dim=-1))


REPRESENTATIONS: Dict[str, Representation] = {r.name: r for r in (Matrix(), SixD(), AxisAngle(), Quaternion())}


def get_representation(name: str) -> Representation:
    try:
        return REPRESENTATIONS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown rotation representation '{name}'. Available: {sorted(REPRESENTATIONS)}.") from None
