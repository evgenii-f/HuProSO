"""The heat kernel of SO(3) that RSGM diffuses with (hp/models/igso3.py)."""
import math

import pytest
import roma
import torch

from hp.models import igso3


def ks_distance(angles: torch.Tensor, s: float) -> float:
    """Kolmogorov-Smirnov distance of a sample of angles to the IGSO(3) angle distribution."""
    angles = angles.sort().values
    empirical = torch.arange(1, len(angles) + 1, dtype=torch.float64) / len(angles)
    every = max(1, len(angles) // 500)
    return float((igso3.angle_cdf(angles[::every], s) - empirical[::every]).abs().max())


@pytest.mark.parametrize("s", [0.5, 1.0, 3.0, 6.0])
def test_the_image_sum_equals_the_eigenfunction_series(s):
    omega = torch.linspace(0.01, math.pi - 0.01, 50, dtype=torch.float64)
    assert torch.allclose(igso3.log_density(omega, s), igso3.log_density_series(omega, s), atol=1e-10)


@pytest.mark.parametrize("s", [1e-6, 1e-3, 0.3, 3.0])
def test_the_derivative_is_the_derivative_of_the_log_density(s):
    omega = torch.linspace(0.05, 3, 30, dtype=torch.float64) * min(1.0, 5 * math.sqrt(s))
    h = 1e-6 * omega
    numeric = (igso3.log_density(omega + h, s) - igso3.log_density(omega - h, s)) / (2 * h)
    assert torch.allclose(igso3.d_log_density(omega, s), numeric, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("s", [0.2, 3.0])
def test_the_density_integrates_to_one_over_uniform_rotations(s):
    torch.manual_seed(0)
    omega = roma.rotmat_to_rotvec(roma.random_rotmat(200_000).double()).norm(dim=-1)
    assert igso3.log_density(omega, s).exp().mean().item() == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("s", [1e-7, 1e-4, 0.07, 3.0])
def test_the_angle_sampler_draws_from_igso3(s):
    sampler = igso3.AngleSampler(1e-7, 6.5)
    angles = sampler.sample(torch.full((20_000,), s), torch.Generator().manual_seed(0))
    assert ks_distance(angles, s) < 1.63 / math.sqrt(20_000)  # the 1% level


def test_a_geodesic_random_walk_has_the_igso3_time_convention():
    """Steps x @ exp(hat(sqrt(dt) z)) for a total time s end in IGSO(3) with that s: the generator is Laplace/2."""
    torch.manual_seed(0)
    s, steps, n = 0.5, 200, 20_000
    x = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
    for _ in range(steps):
        x = x @ roma.rotvec_to_rotmat(math.sqrt(s / steps) * torch.randn(n, 3, dtype=torch.float64))
    angles = roma.rotmat_to_rotvec(x).norm(dim=-1)
    assert ks_distance(angles, s) < 1.63 / math.sqrt(n)


def test_the_sampler_refuses_times_outside_its_table():
    sampler = igso3.AngleSampler(1e-3, 1.0, levels=16, angles=64, quantiles=65)
    with pytest.raises(ValueError, match="s must be in"):
        sampler.sample(torch.tensor([2.0]))
