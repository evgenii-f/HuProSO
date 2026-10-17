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
import math
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hp.models.base import PoseModel
from hp.models.io import plain_build_args, save_model

STATE_FILE = "train_state.pt"  # next to model.json and the weights: what is needed to go on training
STATE_FORMAT = 1
METRIC_FILE = "metric.json"  # in a checkpoint: what it is ranked by, see TrainConfig.keep_best
CHECKPOINT = re.compile(r"checkpoint_(\d{6})")
# settings of the optimization: they must not change when a run is resumed (the learning rate schedule depends on them)
LOCKED = ("max_iterations", "batch_size", "lr", "optimizer", "weight_decay", "scheduler", "lr_step_fraction", "lr_gamma",
          "warmup_steps", "clip_grad_norm", "ema_decay")
OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


@dataclass
class TrainConfig:
    max_iterations: int = 20_000
    batch_size: int = 2000
    lr: float = 5e-3
    optimizer: str = "adam"  # "adam" as in the paper, or "adamw" (Loshchilov & Hutter: the weight decay is decoupled)
    weight_decay: float = 0.0  # adam: an L2 penalty added to the gradient; adamw: the weights shrink by lr * weight_decay
    scheduler: str = "step"  # "step" as in the paper, "cosine" (decay to 0 after a linear warm-up), or "none"
    lr_step_fraction: float = 0.7  # StepLR: the learning rate decays at multiples of this share of max_iterations
    lr_gamma: float = 0.5
    warmup_steps: int = 0  # cosine: steps of the linear warm-up
    clip_grad_norm: Optional[float] = 1.0  # None: no clipping
    ema_decay: Optional[float] = None  # e.g. 0.999; None: no EMA
    valid_interval: int = 500
    checkpoint_every: int = 1000  # 0: only the final checkpoint
    keep_last: int = 3  # the newest checkpoints kept, the others are deleted unless they are among the best
    keep_best: int = 3  # also kept: the checkpoints with the lowest validation loss (training loss without a validation set)
    # keep_last and keep_best may overlap (fewer are kept then); the newest checkpoint is always kept, to resume from;
    # keep_last = keep_best = 0: all checkpoints are kept
    time_limit_minutes: Optional[float] = None  # stop cleanly (with a checkpoint) after this long; None: no limit
    device: str = "cpu"
    num_workers: int = 0
    progress: bool = True


def make_optimizer(parameters, config: "TrainConfig") -> torch.optim.Optimizer:
    """The optimizer of `config` (optimizer, lr, weight_decay). With weight_decay 0, adam and adamw are the same."""
    if config.optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{config.optimizer}'. Use {' or '.join(repr(k) for k in OPTIMIZERS)}.")
    if not config.weight_decay >= 0:
        raise ValueError(f"weight_decay must be >= 0, got {config.weight_decay}.")
    return OPTIMIZERS[config.optimizer](parameters, lr=config.lr, weight_decay=config.weight_decay)


@dataclass
class History:
    loss: List[float] = field(default_factory=list)  # the training loss of every step
    valid_iteration: List[int] = field(default_factory=list)
    valid_loss: List[float] = field(default_factory=list)


def _rng_state(device: str) -> dict:
    """The random generators as tensors, numbers and lists only, so that torch.load(weights_only=True) reads them."""
    version, internal, gauss = random.getstate()
    _, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "torch": torch.get_rng_state(),
        "python": {"version": version, "internal": list(internal), "gauss": gauss},
        "numpy": {"keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos), "has_gauss": int(has_gauss), "cached": float(cached)},
    }
    kind = torch.device(device).type
    if kind == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif kind == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state: dict, device: str) -> None:
    torch.set_rng_state(state["torch"])
    python = state["python"]
    random.setstate((python["version"], tuple(python["internal"]), python["gauss"]))
    n = state["numpy"]
    np.random.set_state(("MT19937", n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"]))
    kind = torch.device(device).type
    if kind == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    elif kind == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def read_state(directory) -> dict:
    """train_state.pt of a checkpoint folder (see Trainer.save_state)."""
    path = Path(directory) / STATE_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: {directory} is no checkpoint of a run that can be resumed (it was written before "
            "training could be resumed, or it holds only weights)."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != STATE_FORMAT:
        raise ValueError(f"{path} has format {state.get('format')}, this code reads format {STATE_FORMAT}.")
    return state


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
        self.optimizer = make_optimizer(self.model.parameters(), c)
        self.scheduler = None
        if c.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=c.lr_step_fraction * c.max_iterations, gamma=c.lr_gamma
            )
        elif c.scheduler == "cosine":
            def factor(step):  # of the learning rate, after `step` optimizer steps
                if step < c.warmup_steps:
                    return (step + 1) / c.warmup_steps
                progress = (step - c.warmup_steps) / max(1, c.max_iterations - c.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        elif c.scheduler != "none":
            raise ValueError(f"Unknown scheduler '{c.scheduler}'. Use 'step', 'cosine' or 'none'.")
        self.train_loader = DataLoader(train_set, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers)
        self.valid_loader = None
        if valid_set is not None:
            self.valid_loader = DataLoader(valid_set, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.writer = writer
        self.history = History()
        self.step = 0  # optimizer steps made so far
        self.stop_reason: Optional[str] = None  # set by request_stop() or the time limit: fit() stops after the step
        self.last_save_seconds = 0.0  # how long the last checkpoint took: reserved before the time limit
        self._valid = None  # (step, loss) of the last validation

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

    def request_stop(self, reason: str = "stop requested") -> None:
        """Ask fit() to stop after the current step with a checkpoint (only sets a flag, so a signal handler may call it)."""
        if self.stop_reason is None:
            self.stop_reason = reason

    def save_state(self, directory) -> None:
        """The weights (and the EMA weights) as save() writes them, plus train_state.pt: the step, the state of the
        optimizer and the scheduler, the EMA counter, the history and the random generators. Training can go on
        from such a folder (see fit(resume=...))."""
        directory = Path(directory)
        self.save(directory)
        state = {
            "format": STATE_FORMAT,
            "step": self.step,
            "config": asdict(self.config),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "ema_updates": self.ema.updates if self.ema else None,
            "history": asdict(self.history),
            "rng": _rng_state(self.config.device),
            "data_sizes": self._data_sizes(),
        }
        torch.save(state, directory / STATE_FILE)

    def load_state(self, directory) -> None:
        """Go on from a folder written by save_state(): weights, optimizer, scheduler, EMA, history, step and the random
        generators are restored, and the next step is the one after it. The model and the optimization settings must
        be those of the run that wrote it (LOCKED)."""
        directory = Path(directory)
        state = read_state(directory)
        c = self.config
        # a setting that the run's state does not have was added later: the run had its default
        defaults = {f.name: f.default for f in fields(TrainConfig)}
        saved_config = {k: state["config"].get(k, defaults[k]) for k in LOCKED}
        changed = {k: (saved_config[k], getattr(c, k)) for k in LOCKED if saved_config[k] != getattr(c, k)}
        if changed:
            raise ValueError("Cannot resume with other optimization settings than the run had (saved, now): "
                             + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in changed.items()) + ".")
        if "data_sizes" in state and state["data_sizes"] != self._data_sizes():
            raise ValueError(f"Cannot resume on other data: the run had {state['data_sizes']} poses, now there are "
                             f"{self._data_sizes()}. Only the place of the same data may change.")
        saved = json.loads((directory / "model.json").read_text())["build_args"]
        if saved != json.loads(json.dumps(plain_build_args(self.model))):
            raise ValueError(f"Cannot resume: the model is not the one of the checkpoint ({saved}).")
        self.model.load_state_dict(torch.load(directory / "weights.pt", map_location=c.device, weights_only=True))
        if self.ema:
            if state["ema_updates"] is None or not (directory / "weights_ema.pt").is_file():
                raise ValueError(f"Cannot resume: {directory} holds no EMA weights, but ema_decay is set.")
            self.ema.model.load_state_dict(torch.load(directory / "weights_ema.pt", map_location=c.device, weights_only=True))
            self.ema.updates = state["ema_updates"]
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler:
            self.scheduler.load_state_dict(state["scheduler"])
        self.history = History(**state["history"])
        self.step = state["step"]
        _set_rng_state(state["rng"], c.device)

    def _data_sizes(self) -> dict:
        return {"train": len(self.train_loader.dataset), "valid": len(self.valid_loader.dataset) if self.valid_loader else None}

    def _set_aside_later_states(self) -> None:
        """After resuming from step s: the checkpoints after s and the final files of the run folder belong to the
        trajectory that is abandoned. They are moved to <run>/abandoned_after_<s>[_n] (not deleted), so that neither
        save_checkpoint() keeps them as if they were of this trajectory nor find_checkpoint() resumes from them."""
        if not self.run_dir.is_dir():
            return  # a new run folder: nothing in it yet
        later = [p for p in list_checkpoints(self.run_dir) if int(CHECKPOINT.fullmatch(p.name).group(1)) > self.step]
        final = [self.run_dir / n for n in (STATE_FILE, "weights.pt", "weights_ema.pt", "history.json")
                 if (self.run_dir / n).is_file()]
        if (self.run_dir / STATE_FILE).is_file() and read_state(self.run_dir)["step"] <= self.step:
            final = []  # the final state is the one resumed from (or older): nothing of it is abandoned
        if not later and not final:
            return
        target = self.run_dir / f"abandoned_after_{self.step:06d}"
        n = 1
        while target.exists():
            n += 1
            target = self.run_dir / f"abandoned_after_{self.step:06d}_{n}"
        target.mkdir()
        for path in later + final:
            shutil.move(str(path), str(target / path.name))
        logging.info(f"Resumed from step {self.step}: the later states of the run were moved to {target}.")
        self._write_status(finished=False)  # the run is unfinished again: it must not say so if this session dies

    def _checkpoint_metric(self, validate: bool) -> Optional[dict]:
        """What a checkpoint of this step is ranked by (keep_best): the validation loss, which is computed now if the
        step has none yet and `validate` is set; without a validation set the mean training loss of the last
        valid_interval steps. None if keep_best is off, or there is no validation loss of this step."""
        c = self.config
        if c.keep_best <= 0:
            return None
        if not self.valid_loader:
            window = self.history.loss[-max(1, c.valid_interval):]
            return {"kind": "train", "value": sum(window) / len(window)}
        if self._valid is None or self._valid[0] != self.step:
            if not validate:
                return None
            loss = self.validate()
            self._valid = (self.step, loss)
            self.history.valid_iteration.append(self.step)
            self.history.valid_loss.append(loss)
            if self.writer:
                self.writer.add_scalar("Loss/valid", loss, self.step)
        return {"kind": "valid", "value": self._valid[1]}

    def save_checkpoint(self, metric: Optional[dict] = None) -> Path:
        """<run_dir>/checkpoint_<step>, written complete or not at all: first into a folder that ends in .tmp, which is
        renamed when it is done, so that a run killed in the middle of saving leaves no broken checkpoint. `metric` (from
        _checkpoint_metric) is written into it for keep_best. The checkpoints that keep_last and keep_best do not keep
        are deleted."""
        final = self.run_dir / f"checkpoint_{self.step:06d}"
        if not final.is_dir():  # a checkpoint of this step is the one the run was resumed from: keep it
            start = time.monotonic()
            tmp = final.with_name(final.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            self.save_state(tmp)
            if metric is not None:
                (tmp / METRIC_FILE).write_text(json.dumps(metric))
            tmp.rename(final)
            self.last_save_seconds = time.monotonic() - start
        self._prune_checkpoints()
        return final

    def _prune_checkpoints(self) -> None:
        for leftover in self.run_dir.glob("checkpoint_*.tmp"):  # of a run that was killed while saving
            shutil.rmtree(leftover, ignore_errors=True)
        c = self.config
        checkpoints = list_checkpoints(self.run_dir)
        if (c.keep_last <= 0 and c.keep_best <= 0) or not checkpoints:
            return
        keep = {checkpoints[-1]}  # the newest is the one to resume from
        if c.keep_last > 0:
            keep.update(checkpoints[-c.keep_last:])
        if c.keep_best > 0:
            kind = "valid" if self.valid_loader else "train"
            ranked = []
            for i, p in enumerate(checkpoints):
                value = _score(p, kind)
                if value is not None:
                    ranked.append((value, -i, p))  # ties: the newer
            keep.update(p for _, _, p in sorted(ranked, key=lambda r: r[:2])[: c.keep_best])
        for old in checkpoints:
            if old not in keep:
                shutil.rmtree(old)

    def fit(self, callback: Optional[Callable[[int, Optional[float]], None]] = None, *, resume=None,
            started_at: Optional[float] = None) -> History:
        """Train for max_iterations steps. `callback(step, valid_loss)` is called every valid_interval steps (after
        the validation; valid_loss is None without a validation set). It may raise to stop the training, e.g.
        optuna.TrialPruned.

        `resume`: a folder written by save_state() (a checkpoint, or the run folder of a finished run). The training
        goes on from its step to max_iterations. A new pass over the data starts there, so a run stopped inside a
        pass differs from an uninterrupted one in the order of the poses (not otherwise); at the end of a pass it is
        the same run, bit for bit. Checkpoints after its step and the final files of the run folder (of the trajectory
        that is abandoned) are moved aside, see _set_aside_later_states().

        Without `resume` the step count starts at 0, also when fit() is called again on the same Trainer.

        `started_at`: the time.monotonic() from which `time_limit_minutes` counts (default: now); a script passes the
        start of the process, so that loading the data counts too."""
        c = self.config
        if resume is None:
            self.step, self.stop_reason = 0, None
        else:
            self.load_state(resume)
            if self.run_dir:
                self._set_aside_later_states()
            if self.step >= c.max_iterations:  # the run is finished: (re)write its final files, they may be missing
                logging.info(f"The run is finished: {self.step} of {c.max_iterations} steps are made.")
                if self.run_dir:
                    self._finish()
                return self.history
        self.model.train()
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic() if started_at is None else started_at
        with tqdm(total=c.max_iterations, initial=self.step, disable=not c.progress) as progress:
            while self.step < c.max_iterations and self.stop_reason is None:
                for batch in self.train_loader:
                    loss = self.training_step(batch).item()
                    self.step += 1
                    step = self.step
                    progress.update(1)
                    self.history.loss.append(loss)
                    if self.writer:
                        self.writer.add_scalar("Loss/train", loss, step)
                    if step % c.valid_interval == 0:
                        valid_loss = None
                        if self.valid_loader:
                            valid_loss = self.validate()
                            self._valid = (step, valid_loss)
                            self.history.valid_iteration.append(step)
                            self.history.valid_loss.append(valid_loss)
                            if self.writer:
                                self.writer.add_scalar("Loss/valid", valid_loss, step)
                            logging.info(f"Step {step}: valid loss {valid_loss:.2f} | train loss {loss:.2f}")
                        if callback:
                            callback(step, valid_loss)
                    if self.run_dir and c.checkpoint_every and step % c.checkpoint_every == 0:
                        self.save_checkpoint(self._checkpoint_metric(validate=True))
                    # the stop must leave time for its checkpoint: as long as the last one took
                    if c.time_limit_minutes and (time.monotonic() - started + self.last_save_seconds) / 60 >= c.time_limit_minutes:
                        self.request_stop(f"time limit of {c.time_limit_minutes} minutes")
                    if step >= c.max_iterations or self.stop_reason:
                        break
        if self.step < c.max_iterations:  # stopped: a checkpoint to resume from, and no final files
            logging.info(f"Stopped at step {self.step} of {c.max_iterations}: {self.stop_reason}.")
            if self.run_dir:
                self.save_checkpoint(self._checkpoint_metric(validate=False))  # no time for a validation now
                self._write_status(finished=False)
        elif self.run_dir:
            self._finish()
        return self.history

    def _finish(self) -> None:
        """The final weights and state in the run folder, the history and the status."""
        self.save_state(self.run_dir)
        (self.run_dir / "history.json").write_text(json.dumps(asdict(self.history)))
        self._write_status(finished=True)

    def _write_status(self, finished: bool) -> None:
        """status.json: whether the run is done (what a job script asks to decide whether to resubmit)."""
        status = {"finished": finished, "step": self.step, "max_iterations": self.config.max_iterations,
                  "stopped_by": None if finished else self.stop_reason}
        (self.run_dir / "status.json").write_text(json.dumps(status, indent=1))


def list_checkpoints(run_dir) -> List[Path]:
    """The finished checkpoints of a run folder, the oldest first (a folder that is still being written ends in .tmp)."""
    found = []
    for path in Path(run_dir).iterdir():
        match = CHECKPOINT.fullmatch(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return [path for _, path in sorted(found)]


def find_checkpoint(path):
    """(run folder, folder to resume from) for what the user names: a checkpoint, a run folder of a finished run (it
    holds the final state), or the run folder of a run that stopped (its newest checkpoint with a state)."""
    path = Path(path).expanduser()
    if (path / STATE_FILE).is_file():
        return (path.parent if CHECKPOINT.fullmatch(path.name) else path), path
    if path.is_dir():
        for checkpoint in reversed(list_checkpoints(path)):
            if (checkpoint / STATE_FILE).is_file():
                return path, checkpoint
    raise FileNotFoundError(f"Nothing to resume in {path}: no {STATE_FILE} in it or in a checkpoint folder of it.")


def _score(checkpoint, kind: str) -> Optional[float]:
    """The loss a checkpoint is ranked by (METRIC_FILE) if it is of this kind and finite, else None."""
    path = Path(checkpoint) / METRIC_FILE
    if not path.is_file():
        return None
    metric = json.loads(path.read_text())
    value = metric.get("value")
    return float(value) if metric.get("kind") == kind and value is not None and math.isfinite(value) else None


def best_checkpoint(run_dir) -> Optional[Path]:
    """The kept checkpoint with the lowest validation loss (the training loss for a run without a validation set), or
    None if no checkpoint has a metric (keep_best was 0, or the checkpoints are of an older kind)."""
    checkpoints = list_checkpoints(run_dir)
    for kind in ("valid", "train"):
        ranked = [(_score(p, kind), -i, p) for i, p in enumerate(checkpoints) if _score(p, kind) is not None]
        if ranked:
            return min(ranked, key=lambda r: r[:2])[2]
    return None
