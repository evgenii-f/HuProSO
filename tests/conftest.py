import importlib
import os
from pathlib import Path
from unittest import mock

import pytest
import roma
import torch
from omegaconf import OmegaConf

from hp.utils.paths import REPO_ROOT

PRETRAINED = REPO_ROOT / "experiments" / "pretrained_models" / "id_0900"


def fixed_rotations(batch: int = 8, joints: int = 19, seed: int = 0) -> torch.Tensor:
    """The same random rotation matrices [batch, joints, 3, 3] on every call."""
    torch.manual_seed(seed)
    return roma.random_rotmat(batch * joints).reshape(batch, joints, 3, 3).float()


@pytest.fixture(scope="session")
def legacy_flow():
    """The existing SO3PriorTrainer with the shipped pretrained prior id_0900 (CPU, no dataset needed)."""
    cfg = OmegaConf.load(PRETRAINED / "config.yaml")
    cfg.only_eval, cfg.evaluate, cfg.standard_mode, cfg.condition = True, False, True, False
    cfg.device = "cpu"
    cfg.save.path_dir = str(PRETRAINED)
    with mock.patch.dict(os.environ, {"HUPROSO_DEVICE": "cpu"}):
        trainer = getattr(importlib.import_module("hp.core.trainer"), cfg.trainer)(cfg)
    trainer.load_model(save_path=str(PRETRAINED))
    trainer.pdf_estimator.eval()
    return trainer
