"""Inpainting with the reverse process of RSGM (hp/models/inpainting.py) and run(method="inpaint")."""
import pytest
import roma
import torch

from hp.models.inpainting import _guidance, rsgm_inpaint
from hp.models.rsgm import RSGM, exp_frame


def small_rsgm():
    torch.manual_seed(0)
    return RSGM(19, hidden=(32, 32)).eval()


def angle(a, b):
    return roma.rotmat_geodesic_distance(a, b)


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


def test_inpainting_needs_replacement_or_guidance():
    observed = roma.random_rotmat(19).view(1, 19, 3, 3)
    with pytest.raises(ValueError):
        rsgm_inpaint(small_rsgm(), observed, torch.arange(19) < 5, steps=2, replace=False)


def test_without_visible_joints_inpainting_is_sampling():
    model = small_rsgm()
    observed = roma.random_rotmat(2 * 19).view(2, 19, 3, 3)
    out = rsgm_inpaint(model, observed, torch.zeros(19, dtype=torch.bool), steps=3)
    assert out.shape == observed.shape
    assert (out.transpose(-1, -2) @ out - torch.eye(3)).abs().max() < 1e-4
