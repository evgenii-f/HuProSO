import importlib
import json

import numpy as np
import pytest
import roma
import torch
from conftest import fixed_rotations
from test_models_interface import Gaussian
from test_training_smoke import BASE_CONFIG, SMALL, rotations  # noqa: F401  (rotations is a fixture)
from torch.utils.data import Dataset

from hp.core.prior_training import make_config
from hp.data.amass import AMASS
from hp.data.amass_cfg import AmassJoints4Learning
from hp.models import build_model, register_model, registry
from hp.models.io import load_model
from hp.training import EMA, TrainConfig, Trainer


class Poses(Dataset):
    """Random flattened poses [J * 9], or dicts with a condition."""

    def __init__(self, n=64, joints=4, cond_dim=0):
        torch.manual_seed(1)
        self.poses = roma.random_rotmat(n * joints).reshape(n, joints * 9).float()
        self.cond = torch.randn(n, cond_dim) if cond_dim else None

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, i):
        return self.poses[i] if self.cond is None else {"theta": self.poses[i], "condition": self.cond[i]}


def config(**kwargs):
    return TrainConfig(**{**dict(batch_size=16, lr=0.05, progress=False, valid_interval=2, checkpoint_every=0), **kwargs})


@pytest.fixture
def make_gaussian(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    register_model("gaussian")(Gaussian)
    return lambda **kw: build_model("gaussian", num_joints=4, representation="6d", **kw)


def test_exactly_max_iterations_steps_are_made(make_gaussian):
    for steps in (6, 7):  # 64 poses / 16 = 4 batches per epoch: 6 and 7 steps end inside the second epoch
        history = Trainer(make_gaussian(), Poses(), None, config(max_iterations=steps)).fit()
        assert len(history.loss) == steps


def test_the_loss_goes_down(make_gaussian):
    history = Trainer(make_gaussian(), Poses(n=256), None, config(max_iterations=60)).fit()
    assert np.mean(history.loss[-10:]) < np.mean(history.loss[:10])


def test_validation_runs_at_the_interval(make_gaussian):
    history = Trainer(make_gaussian(), Poses(), Poses(n=32), config(max_iterations=5)).fit()
    assert history.valid_iteration == [2, 4] and np.isfinite(history.valid_loss).all()


def test_the_learning_rate_decays_at_the_configured_share(make_gaussian):
    trainer = Trainer(make_gaussian(), Poses(), None, config(max_iterations=10, lr_step_fraction=0.5, lr_gamma=0.5))
    trainer.fit()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.05 * 0.5**2)  # decayed after 5 and 10 steps


def test_ema_follows_the_weights_with_a_warm_up(make_gaussian):
    model = make_gaussian()
    ema = EMA(model, decay=0.9)
    before = ema.model.mean.detach().clone()
    with torch.no_grad():
        model.mean.add_(1.0)
    ema.update(model)
    decay = 2 / 11  # min(0.9, (1 + 1) / (10 + 1)): the first updates follow the weights quickly
    assert torch.allclose(ema.model.mean, decay * before + (1 - decay) * model.mean)
    assert not any(p.requires_grad for p in ema.model.parameters())


def test_checkpoints_history_and_ema_are_written(make_gaussian, tmp_path):
    trainer = Trainer(make_gaussian(), Poses(), Poses(n=32), config(max_iterations=5, checkpoint_every=2, ema_decay=0.9), run_dir=tmp_path / "run")
    trainer.fit()
    run = tmp_path / "run"
    assert sorted(p.name for p in run.iterdir()) == [
        "checkpoint_000002", "checkpoint_000004", "history.json", "model.json", "weights.pt", "weights_ema.pt"]
    assert len(json.loads((run / "history.json").read_text())["loss"]) == 5
    pose = fixed_rotations(joints=4)
    assert torch.equal(load_model(run).loss(pose), trainer.model.eval().loss(pose))
    assert torch.equal(load_model(run, weights="weights_ema").loss(pose), trainer.ema.model.loss(pose))
    assert not torch.equal(trainer.ema.model.mean, trainer.model.mean)  # the average is not the raw weights
    assert (run / "checkpoint_000002" / "model.json").is_file()


def test_a_non_finite_loss_gives_no_update():
    class Poisoned(Gaussian):
        calls = 0

        def loss(self, pose, cond=None):
            Poisoned.calls += 1
            return super().loss(pose, cond) * (float("nan") if Poisoned.calls == 1 else 1.0)

    model = Poisoned(num_joints=4)
    trainer = Trainer(model, Poses(), None, config(max_iterations=2))
    batch = next(iter(trainer.train_loader))
    before = model.mean.detach().clone()
    assert torch.isnan(trainer.training_step(batch))
    assert torch.equal(model.mean, before) and trainer.scheduler.last_epoch == 0  # nothing moved
    trainer.training_step(batch)
    assert not torch.equal(model.mean, before) and trainer.scheduler.last_epoch == 1


def test_conditioned_batches_reach_the_model(make_gaussian):
    model = make_gaussian(cond_dim=3)  # the model raises if the condition is missing or has the wrong shape
    history = Trainer(model, Poses(cond_dim=3), Poses(n=32, cond_dim=3), config(max_iterations=2)).fit()
    assert len(history.loss) == 2


def test_bad_settings_are_refused(make_gaussian):
    class Empty(Dataset):
        def __len__(self):
            return 0

    with pytest.raises(ValueError, match="empty"):
        Trainer(make_gaussian(), Empty(), None, config())
    with pytest.raises(ValueError, match="scheduler"):
        Trainer(make_gaussian(), Poses(), None, config(scheduler="cosine"))


def test_the_flow_trains_like_the_old_trainer(rotations, tmp_path):  # noqa: F811
    """Same seed, same data, same settings: the loss of every step and the weights afterwards are the same."""
    steps = 4
    old_cfg = make_config(BASE_CONFIG, exp_id=1, save_root=str(tmp_path / "old"), max_iterations=steps,
                          valid_interval=10_000, save_ckpt_every=10_000, device="cpu", overrides={**SMALL, "so3.layers": 2})
    np.random.seed(0), torch.manual_seed(0)
    old = getattr(importlib.import_module("hp.core.trainer"), old_cfg.trainer)(old_cfg)
    old_loss, _ = old.train()

    np.random.seed(0), torch.manual_seed(0)
    model = build_model("so3_flow", num_joints=19, flow={"layers": 2})
    train_set = AMASS(mode="train", rotation_representation="matrix", selected_joints=AmassJoints4Learning)
    new = Trainer(model, train_set, None, TrainConfig(max_iterations=steps, batch_size=64, progress=False))
    new_loss = new.fit().loss

    np.testing.assert_allclose(new_loss, old_loss, rtol=1e-5)
    new_weights, old_weights = model.estimator.state_dict(), old.pdf_estimator.state_dict()
    assert new_weights.keys() == old_weights.keys()
    for key in new_weights:
        torch.testing.assert_close(new_weights[key], old_weights[key], rtol=1e-5, atol=1e-6)
