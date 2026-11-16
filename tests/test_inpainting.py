"""Inpainting with the reverse process of RSGM (hp/models/inpainting.py) and run(method="inpaint")."""
import pytest
import roma
import torch

from hp.evaluation.completion import OCCLUSIONS, run
from hp.models.dposer import DPoser
from hp.models.inpainting import (_dposer_guidance, _dposer_renoise, _guidance, dposer_inpaint, dposer_times,
                                  rsgm_inpaint)
from hp.models.rsgm import RSGM, exp_frame


def small_rsgm():
    torch.manual_seed(0)
    return RSGM(19, hidden=(32, 32)).eval()


def angle(a, b):
    return roma.rotmat_geodesic_distance(a, b)


def test_inpainting_keeps_the_visible_joints_and_gives_rotations():
    gt = roma.random_rotmat(2 * 21).view(2, 21, 3, 3)
    pred = run(small_rsgm(), gt, "arms", k=2, method="inpaint", progress=False,
               prior=dict(steps=5, resample=2, guidance=0.1))
    visible = [j for j in range(21) if j not in OCCLUSIONS["arms"]]
    assert torch.allclose(pred[:, :, visible], gt[:, None, visible].expand(2, 2, len(visible), 3, 3), atol=1e-5)
    assert (pred.transpose(-1, -2) @ pred - torch.eye(3)).abs().max() < 1e-4
    hidden = list(OCCLUSIONS["arms"])
    assert angle(pred[:, 0, hidden], pred[:, 1, hidden]).mean() > 1e-3  # the hypotheses differ


def test_with_the_score_of_a_point_mass_the_hidden_joints_reach_it(monkeypatch):
    """A prior concentrated at the pose x*: its score at small s is -log(x*^T x) / s in the frame of x. The reverse
    process must bring the hidden joints to x*, whatever the visible ones are. On the EDM grid (dense at small t);
    the uniform grid of 200 steps leaves about 0.03 rad of discretization error (0.005 at 1000 steps)."""
    model = small_rsgm()
    target = roma.random_rotmat(19).view(1, 19, 3, 3)
    monkeypatch.setattr(model, "score", lambda x, t: -roma.rotmat_to_rotvec(target.transpose(-1, -2) @ x)
                        / model.s(t)[:, None, None])
    observed = roma.random_rotmat(3 * 19).view(3, 19, 3, 3)
    visible = torch.zeros(19, dtype=torch.bool)
    visible[:10] = True
    torch.manual_seed(1)
    out = rsgm_inpaint(model, observed, visible, steps=200, schedule="edm")
    assert angle(out[:, ~visible], target[:, ~visible].expand(3, -1, -1, -1)).max() < 0.01
    assert torch.allclose(out[:, visible], observed[:, visible])


def test_the_network_is_called_steps_times_resample(monkeypatch):
    model = small_rsgm()
    calls = []
    real = model.score
    monkeypatch.setattr(model, "score", lambda x, t: calls.append(1) or real(x, t))
    observed = roma.random_rotmat(2 * 19).view(2, 19, 3, 3)
    visible = torch.arange(19) < 5
    rsgm_inpaint(model, observed, visible, steps=4, resample=3)
    assert len(calls) == 12


def test_the_guidance_step_lowers_the_error_of_the_visible_joints(monkeypatch):
    """With a zero score x0_hat = x_t, so the guidance gradient is that of the squared angles of the visible joints
    to the observation; a small step against it must lower them and leave the hidden joints."""
    model = small_rsgm()
    monkeypatch.setattr(model, "score", lambda x, t: torch.zeros(*x.shape[:2], 3, dtype=x.dtype))
    x = roma.random_rotmat(2 * 19).view(2, 19, 3, 3)
    y = roma.random_rotmat(2 * 19).view(2, 19, 3, 3)
    vis = torch.arange(19) < 7
    score, grad = _guidance(model, x, torch.full((2,), 0.3), y, vis)
    assert torch.equal(score, torch.zeros_like(score))
    assert grad[:, ~vis].abs().max() == 0
    moved = exp_frame(x, -1e-2 * grad)
    assert (angle(moved[:, vis], y[:, vis]) <= angle(x[:, vis], y[:, vis]) + 1e-6).all()
    assert (angle(moved[:, vis], y[:, vis]).sum() < angle(x[:, vis], y[:, vis]).sum())


def test_inpainting_needs_replacement_or_guidance_and_an_rsgm():
    observed = roma.random_rotmat(19).view(1, 19, 3, 3)
    with pytest.raises(ValueError):
        rsgm_inpaint(small_rsgm(), observed, torch.arange(19) < 5, steps=2, replace=False)
    with pytest.raises(TypeError):
        run(torch.nn.Linear(1, 1), roma.random_rotmat(21).view(1, 21, 3, 3), "arms", k=1, method="inpaint",
            progress=False)


def test_without_visible_joints_inpainting_is_sampling():
    model = small_rsgm()
    observed = roma.random_rotmat(2 * 19).view(2, 19, 3, 3)
    out = rsgm_inpaint(model, observed, torch.zeros(19, dtype=torch.bool), steps=3)
    assert out.shape == observed.shape
    assert (out.transpose(-1, -2) @ out - torch.eye(3)).abs().max() < 1e-4


# ---- DPoser ----

def small_dposer():
    torch.manual_seed(0)
    return DPoser(21, hidden=64, embed=32, blocks=1).eval()


def test_dposer_inpainting_keeps_the_visible_joints_and_gives_rotations():
    gt = roma.random_rotmat(2 * 21).view(2, 21, 3, 3)
    pred = run(small_dposer(), gt, "legs", k=2, method="inpaint", progress=False,
               prior=dict(steps=5, resample=2, guidance=0.1, schedule="edm"))
    visible = [j for j in range(21) if j not in OCCLUSIONS["legs"]]
    assert torch.allclose(pred[:, :, visible], gt[:, None, visible].expand(2, 2, len(visible), 3, 3), atol=1e-5)
    assert (pred.transpose(-1, -2) @ pred - torch.eye(3)).abs().max() < 1e-4
    hidden = list(OCCLUSIONS["legs"])
    assert angle(pred[:, 0, hidden], pred[:, 1, hidden]).mean() > 1e-3


def test_dposer_times():
    model = small_dposer()
    for schedule in ("uniform", "edm"):
        times = dposer_times(model, 50, schedule)
        assert times[0] == 1.0 and times[-1] == model.sample_eps and len(times) == 51
        assert (times[1:] < times[:-1]).all()
    edm = dposer_times(model, 50, "edm")
    assert (edm[-2] - edm[-1]) < (edm[0] - edm[1])  # dense at small t
    with pytest.raises(ValueError):
        dposer_times(model, 5, "cosine")


def test_dposer_renoising_is_the_forward_transition():
    """x_0 noised to t_next by the marginal and then to t by the transition must have the marginal at t."""
    model = small_dposer()
    torch.manual_seed(2)
    x0 = torch.full((200_000, 1), 0.7)
    t, t_next = torch.tensor(0.6, dtype=torch.float64), torch.tensor(0.3, dtype=torch.float64)
    alpha_next, std_next = model.marginal(t_next)
    x = _dposer_renoise(model, alpha_next.float() * x0 + std_next.float() * torch.randn_like(x0), t, t_next)
    alpha, std = model.marginal(t)
    assert abs(x.mean().item() - 0.7 * alpha.item()) < 5e-3
    assert abs(x.std().item() - std.item()) < 5e-3


def test_dposer_with_the_score_of_a_point_mass_the_hidden_joints_reach_it(monkeypatch):
    """A prior concentrated at x* in the normalized space: the score of N(alpha x*, std^2) is -(x - alpha x*) / std^2.
    The reverse process must bring the hidden joints to x*, whatever the visible ones are."""
    model = small_dposer()
    target = 0.5 * torch.randn(1, 63)
    monkeypatch.setattr(model, "score", lambda x, t: -(x - model.marginal(t)[0][:, None] * target)
                        / model.marginal(t)[1][:, None] ** 2)
    observed = roma.random_rotmat(3 * 21).view(3, 21, 3, 3)
    visible = torch.arange(21) < 10
    torch.manual_seed(1)
    out = dposer_inpaint(model, observed, visible, steps=200, schedule="edm")
    goal = model.denormalize(target)
    assert angle(out[:, ~visible], goal[:, ~visible].expand(3, -1, -1, -1)).max() < 0.01
    assert torch.allclose(out[:, visible], observed[:, visible])


def test_dposer_guidance_lowers_the_error_of_the_visible_joints(monkeypatch):
    model = small_dposer()
    monkeypatch.setattr(model, "noise_prediction", lambda x, t: torch.zeros_like(x))
    torch.manual_seed(3)
    x = 0.3 * torch.randn(2, 63)
    y = roma.random_rotmat(2 * 21).view(2, 21, 3, 3)
    vis = torch.arange(21) < 7
    t = torch.full((2,), 0.01)
    score, grad = _dposer_guidance(model, x, t, y, vis)
    assert torch.equal(score, torch.zeros_like(score))
    assert grad.view(2, 21, 3)[:, ~vis].abs().max() == 0
    alpha = model.marginal(t)[0][:, None]

    def error(z):
        return angle(model.denormalize(z / alpha)[:, vis], y[:, vis]).sum()

    assert error(x - 1e-3 * grad) < error(x)


def test_dposer_inpainting_needs_replacement_or_guidance():
    with pytest.raises(ValueError):
        dposer_inpaint(small_dposer(), roma.random_rotmat(21).view(1, 21, 3, 3), torch.arange(21) < 5, steps=2,
                       replace=False)
