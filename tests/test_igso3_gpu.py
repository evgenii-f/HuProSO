"""The float32 heat kernel of SO(3) (hp/models/igso3_gpu.py) against a 50-digit reference and against hp/models/igso3.py.

The reference is the image sum in mpmath, evaluated at the inputs as the tested dtype rounds them: near omega = pi
for small s the functions are so steep that rounding omega to float32 alone moves them by 1e-2, which no formula
can avoid. Errors of d log f are measured as std(s) * error, the scale of the training target of RSGM.
"""
import math

import pytest
import torch

from hp.models import igso3, igso3_gpu

mp = pytest.importorskip("mpmath")

S = [4e-7, 1e-5, 1e-3, 0.05, 0.5, 0.99, 1.0, 1.5, 5.0, 12.0]
OMEGA = [0.0, 1e-6, 1e-4, 1e-2, 0.3, 1.0, 2.5, math.pi - 1e-3]


def reference(omega: float, s: float):
    """(log f, d log f / d omega) to 50 digits."""
    with mp.workdps(50):
        w, s = mp.mpf(omega), mp.mpf(s)

        def log_f(x):
            if x == 0:  # the limit: bracket / sin(x/2) -> 2 d bracket / d x
                br = 2 * mp.fsum((-1) ** abs(k) * (1 - (2 * mp.pi * k) ** 2 / s) * mp.exp(-(2 * mp.pi * k) ** 2 / (2 * s))
                                 for k in range(-12, 13))
                return s / 8 + mp.log(mp.sqrt(2 * mp.pi / s)) - mp.log(s) + mp.log(br)
            br = mp.fsum((-1) ** abs(k) * (x + 2 * mp.pi * k) * mp.exp(-((x + 2 * mp.pi * k) ** 2 - x**2) / (2 * s))
                         for k in range(-12, 13))
            return s / 8 + mp.log(mp.sqrt(2 * mp.pi / s)) - mp.log(s) - x**2 / (2 * s) + mp.log(br) - mp.log(mp.sin(x / 2))

        d = mp.mpf(0) if w == 0 else mp.diff(log_f, w)  # f is even in omega
        return float(log_f(w)), float(d)


def grid(dtype):
    omega = torch.tensor(OMEGA, dtype=dtype).expand(len(S), -1)
    s = torch.tensor(S, dtype=dtype)[:, None].expand_as(omega)
    ref = torch.tensor([[reference(float(w), float(t)) for w, t in zip(wr, sr)] for wr, sr in zip(omega, s)], dtype=torch.float64)
    return omega, s, ref[..., 0], ref[..., 1]


@pytest.fixture(scope="module")
def grid64():
    return grid(torch.float64)


@pytest.fixture(scope="module")
def grid32():
    return grid(torch.float32)


def std(s):
    return torch.sqrt(-torch.expm1(-s.double()))


def test_float64_matches_the_reference_to_double_precision(grid64):
    omega, s, log_f, d_log_f = grid64
    assert (igso3_gpu.log_density(omega, s) - log_f).abs().max() < 1e-8  # log f is up to -1e7: that is 1e-15 relative
    assert ((igso3_gpu.d_log_density(omega, s) - d_log_f).abs() * std(s)).max() < 1e-11


def test_float32_matches_the_reference(grid32):
    omega, s, log_f, d_log_f = grid32
    got = igso3_gpu.log_density(omega, s).double()
    assert ((got - log_f).abs() / log_f.abs().clamp_min(1)).max() < 3e-6
    d_got = igso3_gpu.d_log_density(omega, s).double()
    assert torch.isfinite(d_got).all()
    assert ((d_got - d_log_f).abs() / d_log_f.abs().clamp_min(1e-3)).max() < 3e-4
    # where the training target lives (omega up to 6 standard deviations), its error is far below float32 noise of O(1)
    typical = omega.double() <= 6 * s.double().sqrt()
    assert ((d_got - d_log_f).abs() * std(s))[typical].max() < 2e-6


def test_the_derivative_is_zero_at_omega_zero():
    s = torch.tensor(S)
    assert torch.equal(igso3_gpu.d_log_density(torch.zeros_like(s), s), torch.zeros_like(s))


@pytest.mark.parametrize("s", [0.5, 1.0, 3.0])
def test_it_agrees_with_the_float64_cpu_kernel(s):
    omega = torch.linspace(0.01, math.pi - 0.01, 50, dtype=torch.float64)
    s = torch.full_like(omega, s)
    assert torch.allclose(igso3_gpu.log_density(omega, s), igso3.log_density(omega, s), atol=1e-10)
    assert torch.allclose(igso3_gpu.d_log_density(omega, s), igso3.d_log_density(omega, s), atol=1e-8)


def test_both_forms_agree_at_the_switch():
    """The image sum (just below S_SWITCH) and the cosine series (at it) give the same values."""
    omega = torch.linspace(0, math.pi, 200, dtype=torch.float64)
    below = torch.full_like(omega, igso3_gpu.S_SWITCH * (1 - 1e-12))
    at = torch.full_like(omega, igso3_gpu.S_SWITCH)
    assert torch.allclose(igso3_gpu.log_density(omega, below), igso3_gpu.log_density(omega, at), atol=1e-10)
    assert torch.allclose(igso3_gpu.d_log_density(omega, below), igso3_gpu.d_log_density(omega, at), atol=1e-10)


@pytest.mark.parametrize("s", [1e-7, 1e-4, 0.07, 3.0])
def test_the_float32_angle_sampler_draws_from_igso3(s):
    from tests.test_igso3 import ks_distance

    sampler = igso3_gpu.AngleSampler(1e-7, 6.5)
    angles = sampler.sample(torch.full((20_000,), s), torch.Generator().manual_seed(0))
    assert angles.dtype == torch.float32
    assert ks_distance(angles.double(), s) < 1.63 / math.sqrt(20_000)  # the 1% level


def test_the_sampler_table_matches_the_float64_one():
    """Same grids: the float32 table differs from igso3's by its rounding and the 1 - cos fix, relative to the angle scale.
    The quantile u = 1 is left out: there the CDF is 1 to rounding over a whole plateau of the tail, and either table
    picks some point of it."""
    old, new = igso3.AngleSampler(1e-6, 5.0, levels=64), igso3_gpu.AngleSampler(1e-6, 5.0, levels=64)
    scale = torch.clamp(old.log_s.exp().sqrt(), max=1.0)[:, None]
    assert ((new.table.double() - old.table).abs() / scale)[:, :-1].max() < 1e-4


def test_the_sampler_table_moves_with_the_module_and_is_not_saved():
    sampler = igso3_gpu.AngleSampler(1e-3, 1.0, levels=16, angles=64, quantiles=65)
    assert "table" in dict(sampler.named_buffers()) and sampler.state_dict() == {}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs mps")
def test_it_runs_on_mps_and_matches_the_cpu():
    omega, s = torch.rand(10_000) * math.pi, torch.exp(torch.rand(10_000) * 16 - 14.7)
    for fn in (igso3_gpu.log_density, igso3_gpu.d_log_density):
        cpu, gpu = fn(omega, s), fn(omega.to("mps"), s.to("mps")).cpu()
        assert torch.allclose(gpu, cpu, rtol=1e-5, atol=1e-5)
    sampler = igso3_gpu.AngleSampler(1e-7, 6.5).to("mps")
    angles = sampler.sample(torch.full((1000,), 0.1, device="mps"))
    assert angles.device.type == "mps" and torch.isfinite(angles).all()
