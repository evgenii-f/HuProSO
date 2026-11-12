"""Pose completion (hp/evaluation/completion.py) and D-Flow (hp/models/dflow.py)."""
import roma
import torch

from hp.evaluation.completion import KEEP, OCCLUSIONS, at_rest, run, summary
from hp.models.dflow import complete, geodesic_loss, prior_of
from hp.models.rsgm import RSGM, exp_frame


def small_rsgm():
    torch.manual_seed(0)
    return RSGM(19, hidden=(32, 32)).eval()


def test_the_geodesic_loss_is_the_mean_angle():
    r = roma.random_rotmat(10)
    angle = torch.full((10,), 0.3)
    axis = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    other = r @ roma.rotvec_to_rotmat(angle[:, None] * axis)
    assert torch.isclose(geodesic_loss(r, other), torch.tensor(0.3), atol=1e-3)


def test_completion_keeps_the_visible_joints_and_gives_rotations():
    model = small_rsgm()
    observed = roma.random_rotmat(3 * 19).view(3, 19, 3, 3)
    visible = torch.ones(19, dtype=torch.bool)
    visible[[0, 3, 6]] = False
    out = complete(prior_of(model, ode_steps=4), observed, visible, steps=2)
    assert torch.allclose(out[:, visible], observed[:, visible])
    assert (out.transpose(-1, -2) @ out - torch.eye(3)).abs().max() < 1e-4


def test_d_flow_lowers_the_loss():
    model = small_rsgm()
    prior = prior_of(model, ode_steps=4)
    target = roma.random_rotmat(4 * 19).view(4, 19, 3, 3)
    torch.manual_seed(1)
    start = prior.noise(4)
    with torch.no_grad():
        before = geodesic_loss(prior.generate(start)[0], target)
    from hp.models.dflow import dflow
    after = geodesic_loss(dflow(prior, lambda x: geodesic_loss(x, target), start, steps=30, running_weight=0.0), target)
    assert after < before


def test_the_default_score_prior_is_the_euler_ode_of_before():
    model = small_rsgm()
    torch.manual_seed(2)
    x0 = roma.random_rotmat(3 * 19).view(3, 19, 3, 3)
    x1, _ = prior_of(model, ode_steps=5).generate(x0)
    x, times = x0, torch.linspace(1.0, model.eps, 6)
    with torch.no_grad():
        for t0, t1 in zip(times[:-1], times[1:]):
            x = exp_frame(x, (t1 - t0) * model.velocity(x, t0.expand(3)))
    assert torch.allclose(x1, x, atol=1e-6)


def test_heun_edm_ode_is_close_to_a_fine_euler_ode_and_has_gradients():
    model = small_rsgm()
    torch.manual_seed(3)
    x0 = roma.random_rotmat(2 * 19).view(2, 19, 3, 3)
    with torch.no_grad():
        fine, _ = prior_of(model, ode_steps=2000).generate(x0)
        coarse, _ = prior_of(model, ode_steps=10).generate(x0)
        heun, _ = prior_of(model, ode_steps=10, sampler="heun", schedule="edm").generate(x0)
    assert torch.allclose(heun, prior_of(model, ode_steps=10, sampler="heun", schedule="edm").generate(x0)[0])
    distance = lambda a: roma.rotmat_geodesic_distance(a, fine).max()
    assert distance(heun) < distance(coarse)
    delta = torch.zeros(2, 19, 3, requires_grad=True)
    x1, cost = prior_of(model, ode_steps=3, sampler="heun", schedule="edm").generate(exp_frame(x0, delta))
    (geodesic_loss(x1, x0) + cost).backward()
    assert delta.grad.abs().sum() > 0


def test_d_flow_refuses_a_stochastic_sampler():
    import pytest

    with pytest.raises(ValueError):
        prior_of(small_rsgm(), sampler="euler_maruyama")


def test_the_prior_options_of_a_spec_reach_the_prior(tmp_path, monkeypatch):
    import hp.evaluation.completion as completion

    file = tmp_path / "weights_ema.pt"
    file.write_bytes(b"x")
    seen = []
    monkeypatch.setattr(completion, "build", lambda spec, device: small_rsgm())
    monkeypatch.setattr("hp.models.dflow.complete", lambda prior, observed, *a: seen.append(prior) or observed)
    spec = {"run": str(tmp_path), "prior": {"sampler": "heun", "schedule": "edm", "ode_steps": 7}}
    completion.cached_run(tmp_path, "m", spec, roma.random_rotmat(21).view(1, 21, 3, 3), "trunk", "cpu", k=1)
    assert (seen[0].sampler, seen[0].schedule, seen[0].ode_steps) == ("heun", "edm", 7)


def test_a_19_joint_model_fills_its_joints_and_rests_the_hidden_toes():
    gt = roma.random_rotmat(2 * 21).view(2, 21, 3, 3)
    pred = run(small_rsgm(), gt, "left_leg", k=2, steps=1, ode_steps=2, progress=False)
    assert pred.shape == (2, 2, 21, 3, 3)
    assert torch.allclose(pred[:, :, 9], torch.eye(3).expand(2, 2, 3, 3))  # the hidden left toe
    assert torch.allclose(pred[:, :, 10], gt[:, None, 10].expand(2, 2, 3, 3))  # the visible right toe
    visible = [j for j in range(21) if j not in OCCLUSIONS["left_leg"]]
    assert torch.allclose(pred[:, :, visible], gt[:, None, visible].expand(2, 2, len(visible), 3, 3))


def test_hypotheses_together_keep_the_visible_joints_and_differ():
    gt = roma.random_rotmat(3 * 21).view(3, 21, 3, 3)
    pred = run(small_rsgm(), gt, "trunk", k=4, steps=1, ode_steps=2, batch=2, together=True, progress=False)
    visible = [j for j in range(21) if j not in OCCLUSIONS["trunk"]]
    assert pred.shape == (3, 4, 21, 3, 3)
    assert torch.allclose(pred[:, :, visible], gt[:, None, visible].expand(3, 4, len(visible), 3, 3))
    assert not torch.allclose(pred[:, 0, 2], pred[:, 1, 2])  # the hypotheses of a pose are not copies


def test_the_cache_ignores_the_speed_settings(tmp_path, monkeypatch):
    import hp.evaluation.completion as completion

    file = tmp_path / "weights_ema.pt"
    file.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(completion, "build", lambda spec, device: None)
    monkeypatch.setattr(completion, "run", lambda *a, **kw: calls.append(kw) or torch.zeros(1, 1, 21, 3, 3))
    spec, gt = {"run": str(tmp_path)}, torch.zeros(1, 21, 3, 3)
    completion.cached_run(tmp_path, "m", spec, gt, "trunk", "cpu")
    assert completion.cached_run(tmp_path, "m", spec, gt, "trunk", None, batch=250, together=True) is not None
    completion.cached_run(tmp_path, "m", spec, gt, "trunk", "cpu", batch=250, together=True)
    assert len(calls) == 1


def test_the_metrics_are_computed_once_and_again_for_new_completions(tmp_path, monkeypatch):
    import os

    import hp.evaluation.completion as completion

    (tmp_path / "weights_ema.pt").write_bytes(b"x")
    monkeypatch.setattr(completion, "build", lambda spec, device: None)
    monkeypatch.setattr(completion, "run", lambda *a, **kw: torch.zeros(2, 3, 21, 3, 3))
    calls = []
    fake = lambda pred, gt, occlusion, body: calls.append(1) or {"mpvpe": pred[:, :, 0, 0, 0].double().numpy() + len(calls)}
    monkeypatch.setattr(completion, "metrics", fake)
    body = type("FakeBody", (), {"identity": {"model": 1}})()
    spec, gt = {"run": str(tmp_path)}, torch.zeros(2, 21, 3, 3)

    pred = completion.cached_run(tmp_path, "m", spec, gt, "trunk", "cpu")
    plain, rest = completion.cached_metrics(tmp_path, "m", "trunk", pred, gt, body)
    again, again_rest = completion.cached_metrics(tmp_path, "m", "trunk", pred, gt, body)
    assert len(calls) == 2  # the completions and the toes at rest, once
    assert (again["mpvpe"] == plain["mpvpe"]).all() and (again_rest["mpvpe"] == rest["mpvpe"]).all()

    os.utime(completion.cache_path(tmp_path, "m", "trunk"), (0, 0))  # other completions: computed again
    completion.cached_metrics(tmp_path, "m", "trunk", pred, gt, body)
    body.identity = {"model": 2}  # another body model: computed again
    completion.cached_metrics(tmp_path, "m", "trunk", pred, gt, body)
    assert len(calls) == 6


def test_the_summary_takes_mean_std_and_min_over_the_hypotheses():
    m = {"mpvpe": torch.tensor([[1.0, 3.0], [5.0, 5.0]]).numpy(), "mpjpe": torch.zeros(2, 2).numpy(),
         "mgeo": torch.zeros(2, 2).numpy(), "apd": torch.tensor([1.0, 3.0]).numpy()}
    s = summary(m)
    assert (s["mpvpe_mean"], s["mpvpe_std"], s["mpvpe_min"], s["apd"]) == (3.5, 0.5, 3.0, 2.0)


def test_the_occlusions_and_toes_fit_the_joints():
    assert all(max(j) < 21 for j in OCCLUSIONS.values()) and len(KEEP) == 19
    assert torch.equal(at_rest(torch.zeros(1, 21, 3, 3), (9, 10))[0, 9], torch.eye(3))
