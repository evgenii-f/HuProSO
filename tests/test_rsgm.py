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


def test_likelihood_weighting_draws_the_times_with_density_beta_over_std_squared():
    model = build_model("rsgm", num_joints=1, hidden=[8], beta_f=10.0, weighting="likelihood")
    t = model.loss_times(torch.rand(50_000, dtype=torch.float64, generator=torch.Generator().manual_seed(0))).sort().values
    assert t.min() >= model.loss_t_min - 1e-12 and t.max() <= 1 + 1e-12
    # the CDF of q(t) ~ beta / std^2 is log(exp(s(t)) - 1), normalized
    log_em1 = lambda tt: torch.log(torch.expm1(model.s(torch.as_tensor(tt, dtype=torch.float64))))
    cdf = (log_em1(t) - log_em1(model.loss_t_min)) / (log_em1(1.0) - log_em1(model.loss_t_min))
    empirical = torch.arange(1, len(t) + 1, dtype=torch.float64) / len(t)
    assert (cdf - empirical).abs().max() < 1.63 / math.sqrt(len(t))
    std = build_model("rsgm", num_joints=1, hidden=[8], beta_f=10.0)
    assert torch.equal(std.loss_times(torch.tensor([0.0, 1.0], dtype=torch.float64)), torch.tensor([std.loss_t_min, 1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="weighting"):
        build_model("rsgm", num_joints=1, weighting="uniform")


# ---- where the noise of the loss is drawn: "cpu" (float64, igso3) or "device" (igso3_gpu) ----

def expected_loss(model, pose, draws=200):
    with torch.no_grad():
        losses = torch.stack([model.loss(pose).cpu() for _ in range(draws)]).double()
    return losses.mean().item(), losses.std().item() / math.sqrt(draws)


def test_both_noise_kernels_give_the_same_expected_loss():
    torch.manual_seed(0)
    cpu = build_model("rsgm", num_joints=3, hidden=[32], noise="cpu")
    device = build_model("rsgm", num_joints=3, hidden=[32], noise="device")
    device.load_state_dict(cpu.state_dict())
    pose = roma.random_rotmat(256 * 3, generator=torch.Generator().manual_seed(1)).reshape(256, 3, 3, 3)
    (a, sem_a), (b, sem_b) = expected_loss(cpu.train(), pose), expected_loss(device.train(), pose)
    assert abs(a - b) < 4 * math.hypot(sem_a, sem_b)


def test_auto_noise_on_the_cpu_is_the_float64_kernel():
    torch.manual_seed(0)
    auto = build_model("rsgm", num_joints=2, hidden=[16]).eval()
    cpu = build_model("rsgm", num_joints=2, hidden=[16], noise="cpu").eval()
    cpu.load_state_dict(auto.state_dict())
    pose = roma.random_rotmat(16 * 2).reshape(16, 2, 3, 3)
    assert auto.loss(pose).item() == cpu.loss(pose).item()


def test_the_device_noise_validation_loss_is_the_same_every_time():
    model = build_model("rsgm", num_joints=2, hidden=[16], noise="device").eval()
    pose = roma.random_rotmat(16 * 2).reshape(16, 2, 3, 3)
    assert model.loss(pose).item() == model.loss(pose).item()


def test_the_angle_table_is_not_in_the_state_dict():
    """Checkpoints stay as they were: only the network is saved, and old checkpoints load."""
    model = build_model("rsgm", num_joints=2, hidden=[16])
    assert all(key.startswith("net.") for key in model.state_dict())


def test_likelihood_weighting_in_float32_matches_float64():
    model = build_model("rsgm", num_joints=1, hidden=[8], beta_f=10.0, weighting="likelihood")
    u = torch.rand(10_000, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    t64, t32 = model.loss_times(u), model.loss_times(u.float())
    assert t32.dtype == torch.float32 and (t32.double() - t64).abs().max() < 1e-5


def test_unknown_noise_is_refused():
    with pytest.raises(ValueError, match="noise"):
        build_model("rsgm", num_joints=1, noise="gpu")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs mps")
def test_on_mps_the_noise_is_drawn_on_the_device():
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=19, hidden=[64], weighting="likelihood").to("mps")
    pose = roma.random_rotmat(256 * 19).reshape(256, 19, 3, 3).to("mps")
    xt, target, t = model.noised_device(pose)
    assert {v.device.type for v in (xt, target, t)} == {"mps"} and model.device_angles.table.device.type == "mps"
    assert all(torch.isfinite(model.loss(pose)) for _ in range(100))
    # the same expected loss as the float64 CPU kernel, on the same weights and poses
    cpu = build_model("rsgm", num_joints=19, hidden=[64], weighting="likelihood", noise="cpu").to("mps")
    cpu.load_state_dict(model.state_dict())
    (a, sem_a), (b, sem_b) = expected_loss(cpu.train(), pose), expected_loss(model.train(), pose)
    assert abs(a - b) < 4 * math.hypot(sem_a, sem_b)
