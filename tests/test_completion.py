"""Pose completion (hp/evaluation/completion.py) and D-Flow (hp/models/dflow.py)."""
import roma
import torch

from hp.evaluation.completion import KEEP, OCCLUSIONS, at_rest, run, summary
from hp.models.dflow import complete, geodesic_loss, prior_of
from hp.models.rsgm import RSGM


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


def test_a_19_joint_model_fills_its_joints_and_rests_the_hidden_toes():
    gt = roma.random_rotmat(2 * 21).view(2, 21, 3, 3)
    pred = run(small_rsgm(), gt, "left_leg", k=2, steps=1, ode_steps=2, progress=False)
    assert pred.shape == (2, 2, 21, 3, 3)
    assert torch.allclose(pred[:, :, 9], torch.eye(3).expand(2, 2, 3, 3))  # the hidden left toe
    assert torch.allclose(pred[:, :, 10], gt[:, None, 10].expand(2, 2, 3, 3))  # the visible right toe
    visible = [j for j in range(21) if j not in OCCLUSIONS["left_leg"]]
    assert torch.allclose(pred[:, :, visible], gt[:, None, visible].expand(2, 2, len(visible), 3, 3))


def test_the_summary_takes_mean_std_and_min_over_the_hypotheses():
    m = {"mpvpe": torch.tensor([[1.0, 3.0], [5.0, 5.0]]).numpy(), "mpjpe": torch.zeros(2, 2).numpy(),
         "mgeo": torch.zeros(2, 2).numpy(), "apd": torch.tensor([1.0, 3.0]).numpy()}
    s = summary(m)
    assert (s["mpvpe_mean"], s["mpvpe_std"], s["mpvpe_min"], s["apd"]) == (3.5, 0.5, 3.0, 2.0)


def test_the_occlusions_and_toes_fit_the_joints():
    assert all(max(j) < 21 for j in OCCLUSIONS.values()) and len(KEEP) == 19
    assert torch.equal(at_rest(torch.zeros(1, 21, 3, 3), (9, 10))[0, 9], torch.eye(3))
