"""Train a model of the zoo from the command line (scripts/train_model.py). Settings: config/train.yaml."""
import json
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from hp.data.amass import AMASS
from hp.data.amass_cfg import AmassJoints4Learning
from hp.models import build_model
from hp.training.config import load_config, load_resume_config, make_run_dir
from hp.training.trainer import TrainConfig, Trainer, find_checkpoint, read_state
from hp.utils.device import resolve_device


EXIT_STOPPED = 75  # exit status of a run that stopped before its last step (EX_TEMPFAIL): resubmit it with resume=


@contextmanager
def _stop_on_signals(trainer):
    """SIGTERM and SIGUSR1 (what a job scheduler sends before it kills a job) make the trainer stop after the current
    step with a checkpoint, instead of dying in the middle of a step."""
    names = [n for n in ("SIGTERM", "SIGUSR1") if hasattr(signal, n)]
    previous = {}
    try:
        for name in names:
            previous[name] = signal.signal(getattr(signal, name), lambda num, frame: trainer.request_stop(f"signal {signal.Signals(num).name}"))
    except ValueError:  # not the main thread: signals cannot be caught here
        pass
    try:
        yield
    finally:
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)


def exit_code(run_dir) -> int:
    """0 if the run is finished, EXIT_STOPPED if it stopped early and can be resumed."""
    status = json.loads((Path(run_dir) / "status.json").read_text())
    return 0 if status["finished"] else EXIT_STOPPED


@contextmanager
def _env(**values):
    """Set environment variables for a block only; None entries are left alone."""
    with mock.patch.dict(os.environ, {k: str(v) for k, v in values.items() if v is not None}):
        yield


def main(argv: Optional[Sequence[str]] = None):
    """Train as configured. Returns the run folder.

    `resume=<run folder or checkpoint>` goes on with a run that was stopped (or killed): in the same run folder, with
    the settings the run had; only RESUMABLE settings may be given besides."""
    started = time.monotonic()  # the time limit counts from here: loading the data is part of the job
    logging.getLogger().setLevel(logging.INFO)
    argv = list(sys.argv[1:] if argv is None else argv)
    resume = next((a.split("=", 1)[1] for a in reversed(argv) if a.startswith("resume=")), None)
    argv = [a for a in argv if not a.startswith("resume=")]
    if resume:
        run_dir, checkpoint = find_checkpoint(resume)
        cfg = load_resume_config(run_dir, argv)
    else:
        cfg = load_config(argv)
    np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)

    with _env(HUPROSO_AMASS_SPLIT=cfg.data.split):
        data = {mode: AMASS(mode=mode, path_to_dataset=cfg.data.rotations, rotation_representation="matrix",
                            selected_joints=AmassJoints4Learning, load_dtype=cfg.data.get("load_dtype"))
                for mode in ("train", "valid")}
    model = build_model(cfg.model.name, num_joints=len(AmassJoints4Learning), **OmegaConf.to_container(cfg.model.args))

    train_cfg = TrainConfig(**{**OmegaConf.to_container(cfg.train), "device": resolve_device(None if cfg.train.device == "auto" else cfg.train.device)})
    # the saved config says what the run used, not "the default": e.g. weight_decay null becomes the optimizer's default
    cfg.train.optimizer, cfg.train.weight_decay = train_cfg.optimizer, train_cfg.weight_decay
    step = 0
    if resume:
        step = read_state(checkpoint)["step"]
        if step < cfg.train.max_iterations:  # a finished run gets only its final files checked, no new session
            OmegaConf.save(cfg, run_dir / f"config_resume_{step:06d}.yaml")  # config.yaml stays as the run began
    else:
        run_dir = make_run_dir(cfg.run_root, cfg.run_name or cfg.model.name)
        OmegaConf.save(cfg, run_dir / "config.yaml")
    writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        # on a resume, the events after the checkpoint (of the stopped run) are replaced by the new ones
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), purge_step=step + 1 if resume else None)
    if resume and step >= train_cfg.max_iterations:
        verb = f"The run is finished ({step} of {train_cfg.max_iterations} steps), checking its final files:"
    else:
        verb = f"Resuming from step {step} of {train_cfg.max_iterations}:" if resume else "Training"
    print(f"{verb} {cfg.model.name} ({model.num_parameters():,} parameters) on {train_cfg.device}; run folder {run_dir}")
    trainer = Trainer(model, data["train"], data["valid"], train_cfg, run_dir=run_dir, writer=writer)
    with _stop_on_signals(trainer):
        trainer.fit(resume=checkpoint if resume else None, started_at=started)
    if trainer.step < train_cfg.max_iterations:
        print(f"Stopped at step {trainer.step} of {train_cfg.max_iterations} ({trainer.stop_reason}). "
              f"Go on with: python scripts/train_model.py resume={run_dir}")
    else:
        print(f"Saved to {run_dir}")
    return run_dir
