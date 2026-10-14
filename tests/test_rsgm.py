"""RSGM on SO(3)^J (hp/models/rsgm.py).

The main checks put the exact score of a known distribution into the model (an "oracle"): data from IGSO(3) around
a mean mu with time s0, so p_t is IGSO(3) around mu with time s0 + s(t). Then log_prob must give the analytic
density and sample() must draw from it. That tests the ODE, the divergence, the base measure and the sampler
without any training.
"""
import math

import pytest
import roma
import torch

from hp.models import build_model, igso3
from hp.models.rsgm import RSGM, exp_frame
from test_igso3 import ks_distance


def log_map(r: torch.Tensor) -> torch.Tensor:
    """Rotation vectors of rotations [..., 3, 3] with angles in (0, pi), in plain ops (torch.func can vmap it)."""
    angle = torch.arccos(((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    axis = torch.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1)
    return axis / axis.norm(dim=-1, keepdim=True) * angle[..., None]


class Oracle(RSGM):
    """RSGM with the exact score of data from IGSO(3) around mu with time s0."""

    s0 = 0.1

    def __init__(self, num_joints, **kwargs):
        super().__init__(num_joints, hidden=[4], **kwargs)
        self.double()
        self.mu = roma.random_rotmat(num_joints, generator=torch.Generator().manual_seed(1)).double()

    def score(self, x, t):
        v = log_map(self.mu.transpose(-1, -2) @ x)
        angle = v.norm(dim=-1)
        return (igso3.d_log_density(angle, self.s0 + self.s(t)[:, None]) / angle)[..., None] * v

    def data(self, n, seed=0):
        """n poses from the oracle's distribution and their angles to mu."""
        generator = torch.Generator().manual_seed(seed)
        angles = igso3.AngleSampler(1e-3, 1.0).sample(torch.full((n, self.num_joints), self.s0), generator)
        poses = exp_frame(self.mu.expand(n, -1, 3, 3), angles[..., None] * igso3.random_axes((n, self.num_joints), generator))
        return poses, angles


def test_log_prob_of_the_oracle_is_the_analytic_density():
    # beta_f = 20: p_1 is uniform to 1e-8, so the only error left is the ODE solver's
    model = Oracle(2, beta_f=20.0, likelihood_steps=100)
    poses, angles = model.data(300)
    exact = igso3.log_density(angles, model.s0 + model.s(torch.tensor(model.eps))).sum(-1)
    error = model.log_prob(poses) - exact
    assert error.mean().abs() < 0.01 and error.abs().max() < 0.1


def test_with_the_default_schedule_the_likelihood_is_off_only_by_the_base_mismatch():
    """beta_f = 6 (the JAX so3 experiment): p_1 is not quite uniform, which costs about 0.01 nats per joint."""
    model = Oracle(2, likelihood_steps=100)
    poses, angles = model.data(300)
    exact = igso3.log_density(angles, model.s0 + model.s(torch.tensor(model.eps))).sum(-1)
    assert (model.log_prob(poses) - exact).mean().abs() < 0.05


def test_samples_of_the_oracle_follow_its_distribution():
    model = Oracle(2, beta_f=20.0, sample_steps=200)
    torch.manual_seed(0)
    angles = log_map(model.mu.transpose(-1, -2) @ model.sample(10_000)).norm(dim=-1)
    for joint in range(2):
        assert ks_distance(angles[:, joint], model.s0) < 1.63 / math.sqrt(10_000)


def test_the_target_of_the_loss_is_the_riemannian_gradient_of_the_heat_kernel():
    """d log f / d omega * u, in the frame of x_t, against autograd of log f(angle(x0^T x_t exp(hat(delta))))."""
    torch.manual_seed(0)
    s = torch.tensor(0.3, dtype=torch.float64)
    x0 = roma.random_rotmat(5).double()
    omega = torch.rand(5, dtype=torch.float64) * 2 + 0.2
    axis = igso3.random_axes((5,))
    xt = exp_frame(x0, omega[:, None] * axis)
    delta = torch.zeros(5, 3, dtype=torch.float64, requires_grad=True)
    angle = log_map(x0.transpose(-1, -2) @ exp_frame(xt, delta)).norm(dim=-1)
    igso3.log_density(angle, s).sum().backward()
    assert torch.allclose(delta.grad, igso3.d_log_density(omega, s)[:, None] * axis, atol=1e-8)


def test_the_divergence_is_the_one_of_finite_differences():
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=3, hidden=[32, 32]).double()
    x, t = roma.random_rotmat(6 * 3).double().reshape(6, 3, 3, 3), torch.rand(6, dtype=torch.float64)
    with torch.no_grad():
        _, divergence = model.velocity_and_divergence(x, t)
        numeric = torch.zeros(6, dtype=torch.float64)
        for joint in range(3):
            for i in range(3):
                e = torch.zeros(6, 3, 3, dtype=torch.float64)
                e[:, joint, i] = 1e-5
                numeric += (model.velocity(exp_frame(x, e), t) - model.velocity(exp_frame(x, -e), t))[:, joint, i] / 2e-5
    assert torch.allclose(divergence, numeric, atol=1e-6)


def test_the_validation_loss_is_the_same_every_time_and_the_training_loss_is_not():
    model = build_model("rsgm", num_joints=2, hidden=[16])
    pose = roma.random_rotmat(16 * 2).reshape(16, 2, 3, 3)
    model.eval()
    assert model.loss(pose).item() == model.loss(pose).item()
    model.train()
    assert model.loss(pose).item() != model.loss(pose).item()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs mps")
def test_the_training_loss_stays_finite_on_mps():
    """On mps, a copy that changes device and dtype in one .to() sometimes returns garbage (torch 2.13)."""
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=19, hidden=[64]).to("mps")
    pose = roma.random_rotmat(256 * 19).reshape(256, 19, 3, 3).to("mps")
    assert all(torch.isfinite(model.loss(pose)) for _ in range(100))
