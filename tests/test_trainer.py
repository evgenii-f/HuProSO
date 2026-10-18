import importlib
import inspect
import json
import time

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
from hp.training import EMA, TrainConfig, Trainer, best_checkpoint, list_checkpoints


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
        "checkpoint_000002", "checkpoint_000004", "history.json", "model.json", "status.json", "train_state.pt", "weights.pt", "weights_ema.pt"]
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


def test_the_cosine_schedule_warms_up_and_decays_to_zero(make_gaussian):
    trainer = Trainer(make_gaussian(), Poses(), None, config(max_iterations=20, scheduler="cosine", warmup_steps=4, valid_interval=1))
    lrs = []
    trainer.fit(callback=lambda step, _: lrs.append(trainer.optimizer.param_groups[0]["lr"]))  # after every step
    assert lrs[:4] == pytest.approx([0.05 * k / 4 for k in (2, 3, 4)] + [0.05])  # linear warm-up, then the full rate
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:])) and lrs[-1] == pytest.approx(0.0, abs=1e-12)


def _steps(model, steps=1, **kwargs):
    """The weights before and after `steps` training steps on the first batch. The weights start at fixed non-zero
    values (the model's zeros would hide any weight decay)."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p))
    trainer = Trainer(model, Poses(), None, config(max_iterations=steps, scheduler="none", **kwargs))
    before = _params(trainer)
    batch = next(iter(trainer.train_loader))
    for _ in range(steps):
        trainer.training_step(batch)
    return before, _params(trainer), trainer


def test_adamw_shrinks_the_weights_apart_from_the_gradient_step(make_gaussian):
    before, adam, _ = _steps(make_gaussian())
    _, adamw, trainer = _steps(make_gaussian(), optimizer="adamw", weight_decay=0.1)
    assert isinstance(trainer.optimizer, torch.optim.AdamW) and trainer.optimizer.param_groups[0]["weight_decay"] == 0.1
    for p0, a, w in zip(before, adam, adamw):  # decoupled: the same Adam step, and the weights shrink by lr * weight_decay
        assert p0.abs().min() > 0
        assert torch.allclose(w - a, -0.05 * 0.1 * p0, atol=1e-6)


def test_weight_decay_defaults_to_the_pytorch_default_of_the_optimizer(make_gaussian):
    for name, optimizer in (("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)):
        pytorch_default = inspect.signature(optimizer).parameters["weight_decay"].default
        cfg = TrainConfig(optimizer=name)
        assert cfg.weight_decay == pytorch_default  # adam 0, adamw 0.01
        assert Trainer(make_gaussian(), Poses(), None, cfg).optimizer.param_groups[0]["weight_decay"] == pytorch_default
    assert TrainConfig(optimizer="adamw", weight_decay=0.0).weight_decay == 0.0  # an explicit value is kept
    assert TrainConfig(optimizer="adam", weight_decay=0.1).weight_decay == 0.1


def test_adamw_by_default_is_not_adam(make_gaussian):
    _, adam, _ = _steps(make_gaussian(), steps=3)
    _, adamw, _ = _steps(make_gaussian(), steps=3, optimizer="adamw")  # weight_decay 0.01: the weights shrink
    assert all(not torch.equal(a, w) for a, w in zip(adam, adamw))


def test_without_weight_decay_adam_and_adamw_make_the_same_steps(make_gaussian):
    _, adam, trainer = _steps(make_gaussian(), steps=3)
    _, adamw, _ = _steps(make_gaussian(), steps=3, optimizer="adamw", weight_decay=0.0)
    assert type(trainer.optimizer) is torch.optim.Adam and trainer.optimizer.param_groups[0]["weight_decay"] == 0.0
    for a, w in zip(adam, adamw):
        assert torch.equal(a, w)


def test_weight_decay_in_adam_is_an_l2_penalty_in_the_gradient(make_gaussian):
    class Penalized(Gaussian):
        def loss(self, pose, cond=None):  # weight_decay / 2 * |p|^2, with weight_decay 0.1
            return super().loss(pose, cond) + 0.05 * sum((p**2).sum() for p in self.parameters())

    # without clipping: the optimizer adds weight_decay * p after the gradient is clipped, a penalty in the loss is clipped too
    _, decayed, _ = _steps(make_gaussian(), steps=3, weight_decay=0.1, clip_grad_norm=None)
    _, penalized, _ = _steps(Penalized(num_joints=4), steps=3, clip_grad_norm=None)
    _, adam, _ = _steps(make_gaussian(), steps=3, clip_grad_norm=None)
    for d, p, a in zip(decayed, penalized, adam):
        assert torch.allclose(d, p, atol=1e-6) and not torch.allclose(d, a, atol=1e-4)


def test_the_callback_comes_at_the_interval_and_can_stop_the_training(make_gaussian):
    calls = []
    Trainer(make_gaussian(), Poses(), Poses(n=32), config(max_iterations=6)).fit(callback=lambda s, v: calls.append((s, v)))
    assert [s for s, _ in calls] == [2, 4, 6] and all(np.isfinite(v) for _, v in calls)
    calls = []
    Trainer(make_gaussian(), Poses(), None, config(max_iterations=4)).fit(callback=lambda s, v: calls.append((s, v)))
    assert calls == [(2, None), (4, None)]  # no validation set: no validation loss

    class Stop(Exception):
        pass

    def stop(step, _):
        raise Stop

    with pytest.raises(Stop):
        Trainer(make_gaussian(), Poses(), None, config(max_iterations=10)).fit(callback=stop)


def test_bad_settings_are_refused(make_gaussian):
    class Empty(Dataset):
        def __len__(self):
            return 0

    with pytest.raises(ValueError, match="empty"):
        Trainer(make_gaussian(), Empty(), None, config())
    with pytest.raises(ValueError, match="scheduler"):
        Trainer(make_gaussian(), Poses(), None, config(scheduler="linear"))
    with pytest.raises(ValueError, match="optimizer 'sgd'"):
        Trainer(make_gaussian(), Poses(), None, config(optimizer="sgd"))
    with pytest.raises(ValueError, match="weight_decay"):
        Trainer(make_gaussian(), Poses(), None, config(weight_decay=-0.1))


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


def test_a_checkpoint_holds_everything_to_go_on(make_gaussian, tmp_path):
    trainer = Trainer(make_gaussian(), Poses(), Poses(n=32), config(max_iterations=5, checkpoint_every=4, ema_decay=0.9), run_dir=tmp_path / "run")
    trainer.fit()
    folder = tmp_path / "run" / "checkpoint_000004"
    assert sorted(p.name for p in folder.iterdir()) == ["metric.json", "model.json", "train_state.pt", "weights.pt", "weights_ema.pt"]
    state = torch.load(folder / "train_state.pt", weights_only=True)  # plain tensors, numbers and lists only
    assert state["step"] == 4 and state["ema_updates"] == 4 and len(state["history"]["loss"]) == 4
    assert state["config"]["lr"] == 0.05 and "state" in state["optimizer"] and state["scheduler"]["last_epoch"] == 4
    assert {"torch", "python", "numpy"} <= state["rng"].keys()


def test_only_the_last_checkpoints_are_kept(make_gaussian, tmp_path):
    run = tmp_path / "run"
    Trainer(make_gaussian(), Poses(), None, config(max_iterations=10, checkpoint_every=2, keep_last=2, keep_best=0), run_dir=run).fit()
    assert [p.name for p in list_checkpoints(run)] == ["checkpoint_000008", "checkpoint_000010"]
    Trainer(make_gaussian(), Poses(), None, config(max_iterations=6, checkpoint_every=2, keep_last=0, keep_best=0), run_dir=tmp_path / "all").fit()
    assert len(list_checkpoints(tmp_path / "all")) == 3


def test_a_checkpoint_that_was_killed_while_saving_is_not_one(make_gaussian, tmp_path, monkeypatch):
    run = tmp_path / "run"
    trainer = Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, checkpoint_every=2), run_dir=run)
    with monkeypatch.context() as patch:  # the process dies while it writes train_state.pt
        patch.setattr(torch, "save", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            trainer.fit()
    assert list_checkpoints(run) == [] and [p.name for p in run.glob("checkpoint_*")] == ["checkpoint_000002.tmp"]
    Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, checkpoint_every=2), run_dir=run).fit()
    assert sorted(p.name for p in run.glob("checkpoint_*")) == ["checkpoint_000002", "checkpoint_000004"]  # the leftover is cleaned


def _params(trainer):
    return [p.detach().clone() for p in trainer.model.parameters()]


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
@pytest.mark.parametrize("scheduler", ["step", "cosine", "none"])
def test_a_resumed_run_equals_an_uninterrupted_one_at_the_end_of_a_pass(make_gaussian, tmp_path, scheduler, optimizer):
    # 64 poses / batch 16 = 4 steps per pass: the checkpoint at step 4 is at the end of a pass
    cfg = config(max_iterations=8, checkpoint_every=4, ema_decay=0.9, scheduler=scheduler, warmup_steps=1, lr_step_fraction=0.75,  # step: the rate halves at step 6
                 optimizer=optimizer, weight_decay=0.01)
    whole = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=tmp_path / "whole")
    whole.fit()
    torch.manual_seed(123)  # whatever the state of the generators is: it comes from the checkpoint
    resumed = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=tmp_path / "resumed")
    resumed.fit(resume=tmp_path / "whole" / "checkpoint_000004")
    assert resumed.step == 8 and resumed.history == whole.history
    for a, b in zip(_params(whole), _params(resumed)):
        assert torch.equal(a, b)
    for a, b in zip(whole.ema.model.parameters(), resumed.ema.model.parameters()):
        assert torch.equal(a, b)
    assert whole.ema.updates == resumed.ema.updates == 8
    assert whole.optimizer.param_groups[0]["lr"] == resumed.optimizer.param_groups[0]["lr"]
    assert torch.equal(whole.optimizer.state_dict()["state"][0]["exp_avg"], resumed.optimizer.state_dict()["state"][0]["exp_avg"])


def test_a_run_resumed_inside_a_pass_goes_on_from_its_step(make_gaussian, tmp_path):
    cfg = config(max_iterations=8, checkpoint_every=3, ema_decay=0.9, keep_best=0)
    whole = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=tmp_path / "whole")
    whole.fit()
    resumed = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=tmp_path / "resumed")
    resumed.fit(resume=tmp_path / "whole" / "checkpoint_000003")
    assert resumed.step == 8 and len(resumed.history.loss) == 8
    assert resumed.history.loss[:3] == whole.history.loss[:3] and np.isfinite(resumed.history.loss).all()
    assert resumed.history.valid_iteration == [2, 4, 6, 8]


def test_a_finished_run_is_not_trained_again(make_gaussian, tmp_path):
    cfg = config(max_iterations=4)
    first = Trainer(make_gaussian(), Poses(), None, cfg, run_dir=tmp_path / "run")
    first.fit()
    again = Trainer(make_gaussian(), Poses(), None, cfg, run_dir=tmp_path / "run")
    history = again.fit(resume=tmp_path / "run")  # the run folder itself holds the final state
    assert again.step == 4 and len(history.loss) == 4
    for a, b in zip(_params(first), _params(again)):
        assert torch.equal(a, b)


def test_resuming_with_other_optimization_settings_is_refused(make_gaussian, tmp_path):
    Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, checkpoint_every=2), run_dir=tmp_path / "run").fit()
    checkpoint = tmp_path / "run" / "checkpoint_000002"
    with pytest.raises(ValueError, match="lr 0.05 -> 0.1"):
        Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, lr=0.1)).fit(resume=checkpoint)
    with pytest.raises(ValueError, match="max_iterations 4 -> 8"):
        Trainer(make_gaussian(), Poses(), None, config(max_iterations=8)).fit(resume=checkpoint)
    with pytest.raises(ValueError, match="ema_decay"):
        Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, ema_decay=0.9)).fit(resume=checkpoint)
    with pytest.raises(ValueError, match="optimizer adam -> adamw"):
        Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, optimizer="adamw")).fit(resume=checkpoint)
    with pytest.raises(ValueError, match="weight_decay 0.0 -> 0.01"):
        Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, weight_decay=0.01)).fit(resume=checkpoint)
    Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, valid_interval=1)).fit(resume=checkpoint)  # others may change


def test_a_checkpoint_from_before_the_optimizer_setting_resumes_with_the_defaults(make_gaussian, tmp_path):
    cfg = config(max_iterations=4, checkpoint_every=2)
    Trainer(make_gaussian(), Poses(), None, cfg, run_dir=tmp_path / "run").fit()
    path = tmp_path / "run" / "checkpoint_000002" / "train_state.pt"
    state = torch.load(path, weights_only=True)
    del state["config"]["optimizer"], state["config"]["weight_decay"]  # as written before these settings existed
    torch.save(state, path)
    Trainer(make_gaussian(), Poses(), None, cfg).fit(resume=path.parent)  # it was Adam without weight decay
    with pytest.raises(ValueError, match="optimizer adam -> adamw"):
        Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, optimizer="adamw")).fit(resume=path.parent)


def test_a_run_of_adamw_without_weight_decay_resumes_only_without_it(make_gaussian, tmp_path):
    """Its state says weight_decay 0.0: a config that leaves weight_decay to the default (0.01 for adamw) is another run."""
    Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, checkpoint_every=2, optimizer="adamw", weight_decay=0.0),
            run_dir=tmp_path / "run").fit()
    checkpoint = tmp_path / "run" / "checkpoint_000002"
    Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, optimizer="adamw", weight_decay=0.0)).fit(resume=checkpoint)
    with pytest.raises(ValueError, match="weight_decay 0.0 -> 0.01"):
        Trainer(make_gaussian(), Poses(), None, config(max_iterations=4, optimizer="adamw")).fit(resume=checkpoint)


def test_resuming_another_model_or_a_checkpoint_without_state_is_refused(make_gaussian, tmp_path):
    cfg = config(max_iterations=4, checkpoint_every=2)
    Trainer(make_gaussian(), Poses(), None, cfg, run_dir=tmp_path / "run").fit()
    with pytest.raises(ValueError, match="not the one of the checkpoint"):
        Trainer(build_model("gaussian", num_joints=4, representation="matrix"), Poses(), None, cfg).fit(resume=tmp_path / "run" / "checkpoint_000002")
    (tmp_path / "run" / "checkpoint_000002" / "train_state.pt").unlink()  # a checkpoint of the old kind: weights only
    with pytest.raises(FileNotFoundError, match="can be resumed"):
        Trainer(make_gaussian(), Poses(), None, cfg).fit(resume=tmp_path / "run" / "checkpoint_000002")


def test_a_run_killed_after_its_last_checkpoint_gets_its_final_files(make_gaussian, tmp_path):
    run, cfg = tmp_path / "run", config(max_iterations=4, checkpoint_every=4)
    Trainer(make_gaussian(), Poses(), None, cfg, run_dir=run).fit()
    for name in ("weights.pt", "history.json", "train_state.pt"):  # the process died before it wrote them
        (run / name).unlink()
    Trainer(make_gaussian(), Poses(), None, cfg, run_dir=run).fit(resume=run / "checkpoint_000004")
    assert (run / "weights.pt").is_file() and (run / "train_state.pt").is_file()
    assert len(json.loads((run / "history.json").read_text())["loss"]) == 4


def test_a_stop_request_ends_the_run_with_a_checkpoint_to_resume_from(make_gaussian, tmp_path):
    run, cfg = tmp_path / "run", config(max_iterations=8, ema_decay=0.9)
    trainer = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=run)
    trainer.fit(callback=lambda step, loss: trainer.request_stop("test") if step == 2 else None)
    assert trainer.step == 2 and trainer.stop_reason == "test" and len(trainer.history.loss) == 2
    assert [p.name for p in list_checkpoints(run)] == ["checkpoint_000002"]
    assert not (run / "weights.pt").exists() and not (run / "history.json").exists()  # the run is not finished
    assert json.loads((run / "status.json").read_text()) == {"finished": False, "step": 2, "max_iterations": 8, "stopped_by": "test"}
    resumed = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=run)
    resumed.fit(resume=run / "checkpoint_000002")
    assert resumed.step == 8 and (run / "weights.pt").is_file()
    assert json.loads((run / "status.json").read_text())["finished"] is True


def test_the_time_limit_stops_the_run_after_a_step(make_gaussian, tmp_path):
    run = tmp_path / "run"
    trainer = Trainer(make_gaussian(), Poses(), None, config(max_iterations=50, time_limit_minutes=1e-9), run_dir=run)
    trainer.fit()
    assert trainer.step == 1 and "time limit" in trainer.stop_reason
    assert [p.name for p in list_checkpoints(run)] == ["checkpoint_000001"]


def test_a_stop_request_in_the_last_step_does_not_leave_the_run_unfinished(make_gaussian, tmp_path):
    run = tmp_path / "run"
    trainer = Trainer(make_gaussian(), Poses(), Poses(n=32), config(max_iterations=4), run_dir=run)
    trainer.fit(callback=lambda step, loss: trainer.request_stop() if step == 4 else None)
    assert trainer.step == 4 and json.loads((run / "status.json").read_text())["finished"] is True


def test_the_first_stop_reason_stays(make_gaussian):
    trainer = Trainer(make_gaussian(), Poses(), None, config())
    trainer.request_stop("first"), trainer.request_stop("second")
    assert trainer.stop_reason == "first"


def test_resuming_from_an_older_checkpoint_sets_the_later_states_aside(make_gaussian, tmp_path):
    run, cfg = tmp_path / "run", config(max_iterations=6, checkpoint_every=2)
    Trainer(make_gaussian(), Poses(), None, cfg, run_dir=run).fit()  # checkpoints 2, 4, 6 and the final state
    resumed = Trainer(make_gaussian(), Poses(), None, cfg, run_dir=run)
    resumed.fit(resume=run / "checkpoint_000002")
    aside = run / "abandoned_after_000002"
    assert sorted(p.name for p in aside.iterdir()) == ["checkpoint_000004", "checkpoint_000006", "history.json",
                                                       "train_state.pt", "weights.pt"]
    # the checkpoints and the final state of the run folder are all of the new trajectory now
    assert [p.name for p in list_checkpoints(run)] == ["checkpoint_000002", "checkpoint_000004", "checkpoint_000006"]
    final = torch.load(run / "weights.pt", weights_only=True)
    last = torch.load(run / "checkpoint_000006" / "weights.pt", weights_only=True)
    assert all(torch.equal(final[k], last[k]) for k in final)
    for a, b in zip(_params(resumed), [final[k] for k in final]):
        assert torch.equal(a, b)


def test_resuming_a_run_killed_while_finishing_marks_it_finished(make_gaussian, tmp_path):
    run, cfg = tmp_path / "run", config(max_iterations=4, ema_decay=0.9)
    trainer = Trainer(make_gaussian(), Poses(), None, cfg, run_dir=run)
    trainer.fit(callback=lambda step, loss: trainer.request_stop("job ends") if step == 2 else None)
    assert json.loads((run / "status.json").read_text())["finished"] is False
    finishing = Trainer(make_gaussian(), Poses(), None, cfg, run_dir=run)
    finishing._write_status = lambda finished: (_ for _ in ()).throw(KeyboardInterrupt())  # killed before the status
    with pytest.raises(KeyboardInterrupt):
        finishing.fit(resume=run / "checkpoint_000002")
    assert (run / "train_state.pt").is_file() and json.loads((run / "status.json").read_text())["finished"] is False
    Trainer(make_gaussian(), Poses(), None, cfg, run_dir=run).fit(resume=run)
    assert json.loads((run / "status.json").read_text())["finished"] is True  # else a job script would resubmit forever


def test_the_last_three_checkpoints_are_kept_by_default(make_gaussian, tmp_path):
    assert TrainConfig().keep_last == 3
    Trainer(make_gaussian(), Poses(), None, TrainConfig(max_iterations=10, batch_size=16, checkpoint_every=2, keep_best=0, progress=False), run_dir=tmp_path / "run").fit()
    assert [p.name for p in list_checkpoints(tmp_path / "run")] == ["checkpoint_000006", "checkpoint_000008", "checkpoint_000010"]


def test_resuming_on_other_data_is_refused(make_gaussian, tmp_path):
    cfg = config(max_iterations=4, checkpoint_every=2)
    Trainer(make_gaussian(), Poses(), None, cfg, run_dir=tmp_path / "run").fit()
    with pytest.raises(ValueError, match="other data"):
        Trainer(make_gaussian(), Poses(n=48), None, cfg).fit(resume=tmp_path / "run" / "checkpoint_000002")


def test_the_time_limit_counts_from_the_given_start_and_leaves_time_to_save(make_gaussian, tmp_path):
    cfg = config(max_iterations=50, time_limit_minutes=1.0)
    started_long_ago = Trainer(make_gaussian(), Poses(), None, cfg, run_dir=tmp_path / "a")
    started_long_ago.fit(started_at=time.monotonic() - 61)  # the data took a minute to load: the limit is reached
    assert started_long_ago.step == 1 and "time limit" in started_long_ago.stop_reason
    slow_saver = Trainer(make_gaussian(), Poses(), None, cfg, run_dir=tmp_path / "b")
    slow_saver.last_save_seconds = 60.0  # a checkpoint takes as long as the whole limit: stop at once
    slow_saver.fit()
    assert slow_saver.step == 1


def test_fit_called_again_trains_again(make_gaussian):
    trainer = Trainer(make_gaussian(), Poses(), None, config(max_iterations=3))
    trainer.fit()
    before = _params(trainer)
    trainer.fit()
    assert trainer.step == 3 and len(trainer.history.loss) == 6
    assert any(not torch.equal(a, b) for a, b in zip(before, _params(trainer)))


def test_a_resumed_run_that_dies_is_not_marked_finished(make_gaussian, tmp_path):
    run, cfg = tmp_path / "run", config(max_iterations=6, checkpoint_every=2)
    Trainer(make_gaussian(), Poses(), None, cfg, run_dir=run).fit()
    assert json.loads((run / "status.json").read_text())["finished"] is True

    def die(step, loss):
        raise KeyboardInterrupt  # killed in the resumed session, before it writes anything

    with pytest.raises(KeyboardInterrupt):
        Trainer(make_gaussian(), Poses(), None, config(max_iterations=6, checkpoint_every=2), run_dir=run).fit(die, resume=run / "checkpoint_000002")
    assert not (run / "weights.pt").exists()
    assert json.loads((run / "status.json").read_text())["finished"] is False  # a job script resubmits it


def test_the_best_checkpoints_by_validation_loss_are_kept_next_to_the_last_ones(make_gaussian, tmp_path):
    run = tmp_path / "run"
    cfg = config(max_iterations=16, checkpoint_every=2, keep_last=2, keep_best=2)
    trainer = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=run)
    trainer.fit()
    valid = dict(zip(trainer.history.valid_iteration, trainer.history.valid_loss))
    steps = [s for s in range(2, 17, 2)]
    best = sorted(steps, key=lambda s: (valid[s], -s))[:2]
    expected = sorted(set(best) | {14, 16})  # the best two and the last two (they may overlap)
    assert [int(p.name[-6:]) for p in list_checkpoints(run)] == expected
    assert best_checkpoint(run) == run / f"checkpoint_{best[0]:06d}"
    metric = json.loads((run / f"checkpoint_{best[0]:06d}" / "metric.json").read_text())
    assert metric == {"kind": "valid", "value": valid[best[0]]}


def test_without_a_validation_set_the_training_loss_ranks_the_checkpoints(make_gaussian, tmp_path):
    run = tmp_path / "run"
    cfg = config(max_iterations=12, checkpoint_every=3, keep_last=1, keep_best=2, valid_interval=2)
    trainer = Trainer(make_gaussian(), Poses(), None, cfg, run_dir=run)
    trainer.fit()
    window = {s: float(np.mean(trainer.history.loss[s - 2: s])) for s in (3, 6, 9, 12)}  # the last valid_interval steps
    best = sorted(window, key=lambda s: (window[s], -s))[:2]
    assert [int(p.name[-6:]) for p in list_checkpoints(run)] == sorted(set(best) | {12})
    assert json.loads((run / "checkpoint_000012" / "metric.json").read_text())["kind"] == "train"


def test_a_checkpoint_between_two_validations_is_validated_for_its_rank(make_gaussian, tmp_path):
    run = tmp_path / "run"
    trainer = Trainer(make_gaussian(), Poses(), Poses(n=32), config(max_iterations=6, checkpoint_every=3, keep_best=1), run_dir=run)
    trainer.fit()
    assert trainer.history.valid_iteration == [2, 3, 4, 6]  # step 3 is no multiple of valid_interval: validated for the checkpoint
    assert (run / "checkpoint_000003" / "metric.json").is_file()


def test_keep_best_alone_keeps_the_best_and_the_newest_and_zero_zero_keeps_all(make_gaussian, tmp_path):
    cfg = dict(max_iterations=12, checkpoint_every=2)
    only_best = Trainer(make_gaussian(), Poses(), Poses(n=32), config(keep_last=0, keep_best=2, **cfg), run_dir=tmp_path / "a")
    only_best.fit()
    valid = dict(zip(only_best.history.valid_iteration, only_best.history.valid_loss))
    best = sorted(range(2, 13, 2), key=lambda s: (valid[s], -s))[:2]
    assert [int(p.name[-6:]) for p in list_checkpoints(tmp_path / "a")] == sorted(set(best) | {12})  # the newest is always kept
    Trainer(make_gaussian(), Poses(), Poses(n=32), config(keep_last=0, keep_best=0, **cfg), run_dir=tmp_path / "b").fit()
    assert len(list_checkpoints(tmp_path / "b")) == 6


def test_a_stop_checkpoint_without_a_validation_is_kept_as_the_newest_but_never_as_a_best(make_gaussian, tmp_path):
    run = tmp_path / "run"
    cfg = config(max_iterations=8, checkpoint_every=2, keep_last=1, keep_best=1, valid_interval=4)
    trainer = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=run)
    step = trainer.training_step
    trainer.training_step = lambda batch: (trainer.request_stop("job ends") if trainer.step == 4 else None, step(batch))[1]
    trainer.fit()  # stops after step 5: the checkpoint of step 5 has no validation, there is no time for one
    assert trainer.step == 5 and not (run / "checkpoint_000005" / "metric.json").exists()
    assert (run / "checkpoint_000005").is_dir()  # the newest: kept, to resume from
    assert best_checkpoint(run) in (run / "checkpoint_000002", run / "checkpoint_000004")


def test_the_best_checkpoints_survive_a_resume_and_nan_is_never_best(make_gaussian, tmp_path):
    run = tmp_path / "run"
    cfg = config(max_iterations=12, checkpoint_every=2, keep_last=1, keep_best=2)
    first = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=run)
    first.fit(callback=lambda step, loss: first.request_stop("job ends") if step == 6 else None)
    resumed = Trainer(make_gaussian(), Poses(), Poses(n=32), cfg, run_dir=run)
    resumed.fit(resume=run / "checkpoint_000006")
    valid = dict(zip(resumed.history.valid_iteration, resumed.history.valid_loss))
    best = sorted(range(2, 13, 2), key=lambda s: (valid[s], -s))[:2]
    assert [int(p.name[-6:]) for p in list_checkpoints(run)] == sorted(set(best) | {12})
    (run / "checkpoint_000012" / "metric.json").write_text('{"kind": "valid", "value": NaN}')
    assert best_checkpoint(run) != run / "checkpoint_000012"
    assert best_checkpoint(tmp_path) is None  # no checkpoints, no best
