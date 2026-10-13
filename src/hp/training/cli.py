"""Train a model of the zoo from the command line (scripts/train_model.py). Settings: config/train.yaml."""
import logging
import os
import sys
from contextlib import contextmanager
from typing import Optional, Sequence
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from hp.data.amass import AMASS
from hp.data.amass_cfg import AmassJoints4Learning
from hp.models import build_model
from hp.training.config import load_config, make_run_dir
from hp.training.trainer import TrainConfig, Trainer
from hp.utils.device import resolve_device


@contextmanager
def _env(**values):
    """Set environment variables for a block only; None entries are left alone."""
    with mock.patch.dict(os.environ, {k: str(v) for k, v in values.items() if v is not None}):
        yield


def main(argv: Optional[Sequence[str]] = None):
    """Train as configured. Returns the run folder."""
    logging.getLogger().setLevel(logging.INFO)
    cfg = load_config(sys.argv[1:] if argv is None else argv)
    np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)

    with _env(HUPROSO_AMASS_SPLIT=cfg.data.split):
        data = {mode: AMASS(mode=mode, path_to_dataset=cfg.data.rotations, rotation_representation="matrix",
                            selected_joints=AmassJoints4Learning) for mode in ("train", "valid")}
    model = build_model(cfg.model.name, num_joints=len(AmassJoints4Learning), **OmegaConf.to_container(cfg.model.args))

    run_dir = make_run_dir(cfg.run_root, cfg.run_name or cfg.model.name)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    train_cfg = TrainConfig(**{**OmegaConf.to_container(cfg.train), "device": resolve_device(None if cfg.train.device == "auto" else cfg.train.device)})
    writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    print(f"Training {cfg.model.name} ({model.num_parameters():,} parameters) on {train_cfg.device}; run folder {run_dir}")
    Trainer(model, data["train"], data["valid"], train_cfg, run_dir=run_dir, writer=writer).fit()
    print(f"Saved to {run_dir}")
    return run_dir
