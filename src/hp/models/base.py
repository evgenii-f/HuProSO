"""The interface of the model zoo.

Every pose model (normalizing flow, diffusion, flow matching, autoregressive flow, ...) is a PoseModel. At its
boundary a model always speaks rotation matrices [batch, joints, 3, 3], the format of the dataset and of the
evaluation. A model that works in another space (6D, axis-angle, ...) converts internally with `encode` and
`decode`, and `decode` always returns valid rotations.

A model implements:
    loss(pose, cond)      the training objective: a likelihood, a denoising error, a velocity regression, ...
    sample(n, cond)       n poses
and optionally
    log_prob(pose, cond)  the exact log-likelihood, for models that have one (has_log_prob = True)

`cond` is an optional conditioning vector [batch, cond_dim] (e.g. features of 3D or 2D keypoints for inverse
kinematics and uplifting). A model that cannot be conditioned keeps supports_cond = False.
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import torch
from torch import nn

from hp.models.representation import get_representation


class PoseModel(nn.Module, ABC):
    name: ClassVar[str] = ""  # set by @register_model
    has_log_prob: ClassVar[bool] = False  # True if log_prob is implemented
    supports_cond: ClassVar[bool] = False  # True if the model can be conditioned
    # The space the density of log_prob lives on, e.g. "SO(3)^J". Log-likelihoods are comparable only within
    # one space: a density on R^(6J) of a 6D model is a different quantity.
    density_space: ClassVar[Optional[str]] = None

    def __init__(self, num_joints: int, representation: str = "matrix", cond_dim: int = 0):
        super().__init__()
        if cond_dim and not self.supports_cond:
            raise ValueError(f"Model '{self.name or type(self).__name__}' cannot be conditioned, but cond_dim={cond_dim}.")
        self.num_joints = num_joints
        self.representation = get_representation(representation)
        self.cond_dim = cond_dim
        self.build_args: dict = {}  # how to rebuild the model; filled in by build_model

    @abstractmethod
    def loss(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scalar training loss of a batch of poses [B, J, 3, 3] (mean over the batch)."""

    @abstractmethod
    def sample(self, n: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """n poses [n, J, 3, 3], valid rotations. `cond` has one row per sample: [n, cond_dim]."""

    def log_prob(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Exact log-likelihood [B] of the poses. Only models with has_log_prob implement it."""
        raise NotImplementedError(f"Model '{self.name or type(self).__name__}' has no exact log-likelihood.")

    # ---- helpers for the models ----

    def encode(self, pose: torch.Tensor) -> torch.Tensor:
        """[B, J, 3, 3] -> [B, J * dim] in the representation of the model."""
        return self.representation.pose_to_vector(pose)

    def decode(self, vector: torch.Tensor) -> torch.Tensor:
        """[B, J * dim] -> [B, J, 3, 3], projected onto valid rotations."""
        return self.representation.pose_from_vector(vector, self.num_joints)

    def check_cond(self, cond: Optional[torch.Tensor], batch: int) -> None:
        """Raise if `cond` does not fit the model: required when cond_dim > 0, forbidden otherwise."""
        if self.cond_dim == 0:
            if cond is not None:
                raise ValueError(f"Model '{self.name}' is not conditional, but a condition was given.")
        elif cond is None:
            raise ValueError(f"Model '{self.name}' is conditional (cond_dim={self.cond_dim}) and needs a condition.")
        elif tuple(cond.shape) != (batch, self.cond_dim):
            raise ValueError(f"The condition must have shape {(batch, self.cond_dim)}, got {tuple(cond.shape)}.")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
