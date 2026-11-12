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


def tiny_body_model(path, joints: int = 6, vertices: int = 20, seed: int = 0):
    """A made-up body model with the arrays of an SMPL-X .npz that hp.models.vposer.BodyMesh reads: a chain of `joints`
    joints (joint i a child of joint (i - 1) // 2), random vertices, skinning weights and pose blend shapes."""
    import numpy as np

    rng = np.random.default_rng(seed)
    weights = rng.random((vertices, joints)).astype(np.float32)
    regressor = rng.random((joints, vertices)).astype(np.float32)
    np.savez(path, v_template=rng.normal(size=(vertices, 3)).astype(np.float32),
             J_regressor=regressor / regressor.sum(1, keepdims=True),
             kintree_table=np.stack([[2 ** 32 - 1] + [(i - 1) // 2 for i in range(1, joints)], list(range(joints))]),
             posedirs=(0.01 * rng.normal(size=(vertices, 3, 9 * (joints - 1)))).astype(np.float32),
             weights=weights / weights.sum(1, keepdims=True))
    return str(path)


TINY_BODY = tiny_body_model(Path(__import__("tempfile").mkdtemp()) / "tiny_body.npz")  # for the model contract
