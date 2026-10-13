"""Train any PoseModel through its loss().

It follows the loop of the old PriorTrainer (Adam, StepLR, gradient clipping, no update on a non-finite loss), so
that the SO(3) flow trains the same way (tests/test_trainer.py compares the losses of both), and adds what other
model families need: an exponential moving average (EMA) of the weights, and conditioned batches.

A dataset item is a pose, flattened [J * 9] or [J, 3, 3], or a dict {"theta": pose, "condition": features}
as the AMASS dataset gives for conditioning. The model gets poses [B, J, 3, 3].
"""
import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hp.models.base import PoseModel
from hp.models.io import save_model


@dataclass
class TrainConfig:
    max_iterations: int = 20_000
    batch_size: int = 2000
    lr: float = 5e-3
    scheduler: str = "step"  # "step" as in the paper, or "none"
    lr_step_fraction: float = 0.7  # StepLR: the learning rate decays at multiples of this share of max_iterations
    lr_gamma: float = 0.5
    clip_grad_norm: Optional[float] = 1.0  # None: no clipping
    ema_decay: Optional[float] = None  # e.g. 0.999; None: no EMA
    valid_interval: int = 500
    checkpoint_every: int = 1000  # 0: only the final checkpoint
    device: str = "cpu"
    num_workers: int = 0
    progress: bool = True


@dataclass
class History:
    loss: List[float] = field(default_factory=list)  # the training loss of every step
    valid_iteration: List[int] = field(default_factory=list)
    valid_loss: List[float] = field(default_factory=list)


class EMA:
    """Exponential moving average of the weights, with the usual warm-up of the decay."""

    def __init__(self, model: PoseModel, decay: float):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: PoseModel) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for averaged, p in zip(self.model.parameters(), model.parameters()):
            averaged.lerp_(p, 1 - decay)  # decay * averaged + (1 - decay) * p
        for averaged, b in zip(self.model.buffers(), model.buffers()):
            averaged.copy_(b)


class Trainer:
    def __init__(
        self,
        model: PoseModel,
        train_set: Dataset,
        valid_set: Optional[Dataset] = None,
        config: Optional[TrainConfig] = None,
        run_dir=None,
        writer=None,
    ):
        """`run_dir`: where checkpoints and history.json go (None: nothing is written).
        `writer`: anything with add_scalar(tag, value, step), e.g. a TensorBoard SummaryWriter."""
        self.config = config or TrainConfig()
        c = self.config
        if len(train_set) == 0:
            raise ValueError("The training set is empty.")
        self.model = model.to(c.device)
        self.ema = EMA(self.model, c.ema_decay) if c.ema_decay else None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=c.lr)
        self.scheduler = None
        if c.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=c.lr_step_fraction * c.max_iterations, gamma=c.lr_gamma
            )
        elif c.scheduler != "none":
            raise ValueError(f"Unknown scheduler '{c.scheduler}'. Use 'step' or 'none'.")
        self.train_loader = DataLoader(train_set, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
        self.valid_loader = None
        if valid_set is not None:
            self.valid_loader = DataLoader(valid_set, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.writer = writer
        self.history = History()

    def _unpack(self, batch):
        """(poses [B, J, 3, 3], condition or None) on the device."""
        cond = None
        if isinstance(batch, dict):
            batch, cond = batch["theta"], batch.get("condition")
        pose = batch.reshape(-1, self.model.num_joints, 3, 3).to(self.config.device)
        return pose, None if cond is None else cond.to(self.config.device)

    def training_step(self, batch) -> torch.Tensor:
        pose, cond = self._unpack(batch)
        self.optimizer.zero_grad()
        loss = self.model.loss(pose, cond)
        if torch.isfinite(loss):  # a non-finite loss gives no update, as in the old trainer
            loss.backward()
            if self.config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            if self.ema:
                self.ema.update(self.model)
        return loss.detach()

    @torch.no_grad()
    def validate(self) -> float:
        """Mean loss over the validation set (of the EMA weights if there are any)."""
        model = self.ema.model if self.ema else self.model
        model.eval()
        losses = [model.loss(*self._unpack(batch)).item() for batch in self.valid_loader]
        self.model.train()
        return sum(losses) / len(losses)

    def save(self, directory=None) -> None:
        directory = self.run_dir if directory is None else Path(directory)
        save_model(self.model, directory)
        if self.ema:
            save_model(self.ema.model, directory, weights="weights_ema")

    def fit(self) -> History:
        c = self.config
        self.model.train()
        step = 0
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        with tqdm(total=c.max_iterations, disable=not c.progress) as progress:
            while step < c.max_iterations:
                for batch in self.train_loader:
                    loss = self.training_step(batch).item()
                    step += 1
                    progress.update(1)
                    self.history.loss.append(loss)
                    if self.writer:
                        self.writer.add_scalar("Loss/train", loss, step)
                    if self.valid_loader and step % c.valid_interval == 0:
                        valid_loss = self.validate()
                        self.history.valid_iteration.append(step)
                        self.history.valid_loss.append(valid_loss)
                        if self.writer:
                            self.writer.add_scalar("Loss/valid", valid_loss, step)
                        logging.info(f"Step {step}: valid loss {valid_loss:.2f} | train loss {loss:.2f}")
                    if self.run_dir and c.checkpoint_every and step % c.checkpoint_every == 0:
                        self.save(self.run_dir / f"checkpoint_{step:06d}")
                    if step >= c.max_iterations:
                        break
        if self.run_dir:
            self.save()
            (self.run_dir / "history.json").write_text(json.dumps(asdict(self.history)))
        return self.history
