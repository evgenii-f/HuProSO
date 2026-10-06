"""Helpers to train the unconditional pose prior with scripts/train_prior.py (used by the notebook and the tests)."""
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from hp.utils.paths import REPO_ROOT


def load_train_script():
    """The module scripts/train_prior.py; its main(config) does the training and saves the run."""
    spec = importlib.util.spec_from_file_location("train_prior", REPO_ROOT / "scripts" / "train_prior.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_config(
    base_config,
    exp_id: int,
    save_root: str,
    max_iterations: int,
    valid_interval: int = 500,
    valid_wasserstein: bool = False,
    device: str = "cpu",
    seed: int = 10,
    clip_grad_norm: float = 1.0,
    save_ckpt_every: int = 1000,
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """Training config from the config of a pretrained prior (e.g. experiments/pretrained_models/id_0900).

    The old config lacks clip_grad_norm and save_ckpt_every, which the current trainer reads, so these two
    training-only keys are added. The model keys that the old config also lacks (clamp_value, only_mobius, ...)
    stay absent on purpose: they are read with a fallback, and adding them would change the model.
    `overrides` sets further keys by their dotted name, e.g. {"batch_size": 64, "model.num_flow_layers": 2}.
    """
    cfg = OmegaConf.load(Path(base_config))
    cfg.exp_id = exp_id
    cfg.seed = seed
    cfg.device = device
    cfg.comment = "prior trained on the compressed data"
    cfg.max_iterations = max_iterations
    cfg.valid.interval = valid_interval
    cfg.valid.compute_wasser = valid_wasserstein
    cfg.clip_grad_norm = clip_grad_norm
    cfg.save_ckpt_every = save_ckpt_every
    cfg.save.path_dir = str(save_root)
    for key, value in (overrides or {}).items():
        OmegaConf.update(cfg, key, value)
    return cfg
