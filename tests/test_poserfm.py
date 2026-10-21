"""PoseRFM, Riemannian flow matching on SO(3)^J (hp/models/poserfm.py)."""
import math

import pytest
import roma
import torch

from hp.models import build_model
from hp.models.poserfm import PoseRFM, log_frame, random_rotations
from hp.models.rsgm import exp_frame

torch.set_default_dtype(torch.float32)


def model(joints=1, hidden=(32, 32), **kwargs):
    torch.manual_seed(0)
    return build_model("poserfm", num_joints=joints, hidden=list(hidden), **kwargs).double().eval()


def constant_field(m, c):
    """Make the network output the constant coordinates c [J, 3] everywhere (a left-invariant field)."""
    with torch.no_grad():
        for p in m.net.parameters():
            p.zero_()
        m.net[-1].bias.copy_(torch.as_tensor(c, dtype=m.dtype).reshape(-1))
    return m


def pairs(n=64, joints=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return random_rotations((n, joints), g, dtype=torch.float64), random_rotations((n, joints), g, dtype=torch.float64)


def test_the_path_goes_from_the_source_to_the_data():
    x0, x1 = pairs()
    start, _ = PoseRFM.conditional_path(x0, x1, torch.zeros(len(x0), dtype=torch.float64))
    end, _ = PoseRFM.conditional_path(x0, x1, torch.ones(len(x0), dtype=torch.float64))
    assert torch.allclose(start, x0, atol=1e-12) and torch.allclose(end, x1, atol=1e-10)


def test_the_target_is_the_velocity_of_the_path_and_the_field_of_the_paper():
    x0, x1 = pairs()
    t = torch.rand(len(x0), generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.9
    xt, target = PoseRFM.conditional_path(x0, x1, t)
    h = 1e-6
    later, _ = PoseRFM.conditional_path(x0, x1, t + h)
    assert torch.allclose(log_frame(xt, later) / h, target, atol=1e-5)  # d/dt x_t = x_t hat(target)
    # the conditional field of the paper, Log_{x_t}(x1) / (1 - t), is the same vector
    assert torch.allclose(log_frame(xt, x1) / (1 - t)[:, None, None], target, atol=1e-8)


def test_the_cut_locus_gives_finite_values():
    x0, _ = pairs(n=8)
    x1 = x0 @ roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, math.pi], dtype=torch.float64))  # rotation by pi
    xt, target = PoseRFM.conditional_path(x0, x1, torch.full((8,), 0.5, dtype=torch.float64))
    assert torch.isfinite(xt).all() and torch.allclose(target.norm(dim=-1), torch.full((8, 3), math.pi, dtype=torch.float64))
    m = model(joints=3)
    loss = m.train().loss(x1)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_a_zero_field_keeps_the_uniform_distribution():
    m = constant_field(model(joints=2), torch.zeros(2, 3))
    torch.manual_seed(3)
    x0 = random_rotations((5, 2), dtype=torch.float64)
    torch.manual_seed(3)
    assert torch.allclose(m.sample(5), x0, atol=1e-12)  # the samples are the uniform source points
    assert torch.allclose(m.log_prob(x0), torch.zeros(5, dtype=torch.float64), atol=1e-12)


def test_a_constant_field_rotates_without_changing_the_density():
    """x(t) = x0 exp(t c): right multiplication preserves the Haar measure, so the divergence and log_prob are 0."""
    c = torch.tensor([[0.3, -0.2, 0.5], [1.0, 0.4, -0.7]], dtype=torch.float64)
    m = constant_field(model(joints=2, eps=0.0), c)
    torch.manual_seed(4)
    x0 = random_rotations((5, 2), dtype=torch.float64)
    torch.manual_seed(4)
    assert torch.allclose(m.sample(5), exp_frame(x0, c.expand(5, -1, -1)), atol=1e-10)
    _, divergence = m.velocity_and_divergence(x0, torch.full((5,), 0.3, dtype=torch.float64))
    assert torch.allclose(divergence, torch.zeros(5, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(m.log_prob(x0), torch.zeros(5, dtype=torch.float64), atol=1e-10)


def nonuniform(joints=1, scale=3.0, **kwargs):
    """A random network with a larger output: a field that moves the mass, so the density is far from uniform."""
    m = model(joints=joints, eps=0.0, **kwargs)
    with torch.no_grad():
        m.net[-1].weight.mul_(scale)
        m.net[-1].bias.mul_(scale)
    return m


def test_the_density_of_a_field_integrates_to_one():
    """E over uniform rotations of exp(log_prob) = the integral of the density = 1 (one joint, Monte Carlo)."""
    m = nonuniform(scale=15.0, likelihood_steps=200)  # log p from about -1.1 to 1.4: the density varies 12-fold
    x = random_rotations((40_000, 1), torch.Generator().manual_seed(5), dtype=torch.float64)
    log_p = m.log_prob(x)
    assert log_p.std() > 0.3  # far from uniform
    density = log_p.exp()
    assert density.mean().item() == pytest.approx(1.0, abs=4 * density.std().item() / math.sqrt(len(x)) + 0.01)


def test_sampling_and_the_likelihood_are_the_same_flow():
    """Integrating a sample back gives its source point; and E over samples of 1 / p = the volume = 1."""
    m = nonuniform(joints=2, sample_steps=200, likelihood_steps=200)
    torch.manual_seed(6)
    x0 = random_rotations((2000, 2), dtype=torch.float64)
    torch.manual_seed(6)
    samples = m.sample(2000)
    back, _ = m.integrate_back(samples)
    assert roma.rotmat_geodesic_distance(back, x0).max() < 1e-4
    inverse_density = (-m.log_prob(samples)).exp()
    assert inverse_density.mean().item() == pytest.approx(1.0, abs=4 * inverse_density.std().item() / math.sqrt(2000) + 0.01)


def test_the_likelihood_converges_with_the_number_of_steps():
    m = nonuniform(joints=2)
    x = random_rotations((200, 2), torch.Generator().manual_seed(7), dtype=torch.float64)
    m.likelihood_steps = 400
    fine = m.log_prob(x)
    m.likelihood_steps = 50
    assert (m.log_prob(x) - fine).abs().max() < 0.05 * (1 + fine.abs().max())


def test_the_validation_loss_is_the_same_every_time_and_the_training_loss_is_not():
    m = model(joints=3)
    x1, _ = pairs(n=32)
    assert torch.equal(m.eval().loss(x1), m.loss(x1))
    m.train()
    assert not torch.equal(m.loss(x1), m.loss(x1))


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="eps"):
        PoseRFM(num_joints=2, eps=1.0)
    with pytest.raises(ValueError, match="matrix"):
        PoseRFM(num_joints=2, representation="6d")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
def test_the_model_runs_on_mps():
    m = build_model("poserfm", num_joints=19, hidden=[64, 64], sample_steps=5, likelihood_steps=5).to("mps")
    x1 = random_rotations((64, 19))
    loss = m.train().loss(x1)
    loss.backward()
    assert torch.isfinite(loss)
    samples = m.eval().sample(16)
    assert samples.device.type == "mps" and torch.allclose(torch.linalg.det(samples.cpu()), torch.ones(16, 19), atol=1e-4)
    assert torch.isfinite(m.log_prob(x1[:8])).all()
