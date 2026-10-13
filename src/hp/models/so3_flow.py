"""The normalizing flow on the product space of SO(3) manifolds (HuProSO3), as a PoseModel.

It wraps the existing estimator of hp.estimators.rotnorm unchanged, so the numbers are the same as with the
SO3PriorTrainer (tests/test_so3_flow.py checks this against the pretrained prior id_0900).
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from hp.estimators.rotnorm.flow.flow import get_flow
from hp.estimators.rotnorm.flow.flow_nd import get_nd_flow
from hp.estimators.rotnorm.utils import sd
from hp.models.base import PoseModel
from hp.models.registry import register_model

# The `so3` section of the config of the paper's unconditional prior (experiments/pretrained_models/id_0900).
PAPER_FLOW_CONFIG = dict(
    dist="mobiusflow", data_dir="data", condition=False, n_dim=19, dim_cond_feature=66, feature_dim=64,
    feature_channels=64, embedding=0, layers=12, segments=32, rot="16Trans", lu=0, num_workers=4,
    frequent_permute=0, last_affine=0, first_affine=0, independent=False, nd_conditioning=False,
)


@register_model("so3_flow")
class SO3Flow(PoseModel):
    """Exact log-likelihood on SO(3)^J. `flow` overrides entries of PAPER_FLOW_CONFIG (e.g. {"layers": 2})."""

    has_log_prob = True
    density_space = "SO(3)^J"

    def __init__(self, num_joints: int, flow: Optional[dict] = None, representation: str = "matrix", cond_dim: int = 0):
        if representation != "matrix":
            raise ValueError("The flow works on rotation matrices directly; use representation='matrix'.")
        super().__init__(num_joints, "matrix", cond_dim)
        flow = OmegaConf.to_container(flow) if isinstance(flow, DictConfig) else (flow or {})
        config = OmegaConf.create({**PAPER_FLOW_CONFIG, **flow})
        config.n_dim = num_joints
        self.estimator = get_flow(config) if num_joints == 1 else get_nd_flow(config)

    def _shaped(self, pose: torch.Tensor) -> torch.Tensor:
        return pose.reshape(-1, 3, 3) if self.num_joints == 1 else pose.reshape(-1, self.num_joints, 3, 3)

    def log_prob(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_cond(cond, len(pose))
        _, log_prob = self.estimator(self._shaped(pose))
        return log_prob

    def loss(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The negative mean log-likelihood (NaNs ignored), as in PriorTrainer.forward_kld."""
        return -torch.nanmean(self.log_prob(pose, cond))

    def sample_with_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """n poses and their log-likelihood: uniform rotations pushed through the inverse flow."""
        if self.num_joints == 1:
            queries = sd.generate_queries(n, mode="random")
        else:
            queries = torch.stack([sd.generate_queries(n, mode="random") for _ in range(self.num_joints)]).permute(1, 0, 2, 3)
        with torch.no_grad():
            samples, log_prob = self.estimator.inverse(queries.to(self.device))
        return samples.reshape(n, self.num_joints, 3, 3), log_prob

    def sample(self, n: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_cond(cond, n)
        return self.sample_with_log_prob(n)[0]

    # The flow keeps the order of its dimensions in a numpy array outside the weights. It is part of the
    # state, so that state_dict() and load_state_dict() carry it.
    def get_extra_state(self):
        permute = getattr(self.estimator, "permute_dims", None)
        return {} if permute is None else {"permute_dims": torch.as_tensor(np.asarray(permute))}

    def set_extra_state(self, state) -> None:
        if "permute_dims" in state:
            self.estimator.permute_dims = state["permute_dims"].cpu().numpy()

    def load_legacy_checkpoint(self, directory, end_str: str = "") -> None:
        """Load the checkpoint format of the SO3PriorTrainer (model_ckpt_pdf_estimator*.pt and permute_dims.pt)."""
        directory = Path(directory)
        weights = torch.load(directory / f"model_ckpt_pdf_estimator{end_str}.pt", map_location=self.device)
        self.estimator.load_state_dict(weights)
        self.estimator.permute_dims = np.loadtxt(directory / "permute_dims.pt")
