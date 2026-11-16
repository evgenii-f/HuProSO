"""Brownian motion on SO(3): the isotropic Gaussian IGSO(3), the heat kernel that RSGM diffuses with.

The metric is the one where the geodesic distance of two rotations is the angle between them (in radians), so a
tangent vector at R is R @ hat(v) with |v| its length, and Brownian motion takes steps R <- R @ exp(hat(sqrt(dt) z))
with z ~ N(0, I_3): its generator is Laplace/2. After a time s the rotation is R0 @ exp(hat(omega * u)) with a
uniform axis u and an angle omega whose density, relative to the uniform distribution on SO(3), is

    f(omega; s) = sum_l (2l + 1) exp(-l(l+1) s / 2) sin((l + 1/2) omega) / sin(omega / 2).

The series converges slowly for small s. By Poisson summation it equals a sum over the "images" of a Gaussian,

    f(omega; s) = exp(s/8) sqrt(2 pi / s) / s * sum_k (-1)^k (omega + 2 pi k) exp(-(omega + 2 pi k)^2 / (2 s)) / sin(omega / 2),

which converges fast for every s that matters here (|k| <= 4 is exact to double precision for s < 20). Both forms are
in this module, and tests/test_igso3.py checks that they agree.

The angle of a uniform rotation has the density (1 - cos omega) / pi on [0, pi]; the angle of IGSO(3) has f times it.

The densities and scores work in float64 on the CPU: the score near s = 0 is 1/s large and the image terms underflow
in float32. Drawing angles (AngleSampler) only interpolates a table of quantiles and works on any device and dtype.
"""
import math

import torch

IMAGES = torch.arange(-4, 5, dtype=torch.float64)  # k of the image sum


def log_density_series(omega: torch.Tensor, s: torch.Tensor, l_max: int = 2000) -> torch.Tensor:
    """log f(omega; s) from the eigenfunction series (reference implementation, for tests)."""
    omega, s = torch.broadcast_tensors(torch.as_tensor(omega, dtype=torch.float64), torch.as_tensor(s, dtype=torch.float64))
    l = torch.arange(l_max + 1, dtype=torch.float64)
    terms = (2 * l + 1) * torch.exp(-l * (l + 1) * s[..., None] / 2) * torch.sin((l + 0.5) * omega[..., None])
    return torch.log(terms.sum(-1) / torch.sin(omega / 2))


def _images(omega: torch.Tensor, s: torch.Tensor):
    """The image sum without the k = 0 Gaussian factor: (bracket, d bracket / d omega), where
    f = exp(s/8) sqrt(2 pi / s) / s * exp(-omega^2 / (2 s)) * bracket / sin(omega / 2). The exponents
    ((omega + 2 pi k)^2 - omega^2) / (2 s) are >= 0 on [0, pi], so nothing overflows."""
    k = IMAGES.to(omega.device)
    shifted = omega[..., None] + 2 * math.pi * k
    weight = (-1) ** k.abs() * torch.exp(-(shifted**2 - omega[..., None] ** 2) / (2 * s[..., None]))
    bracket = (weight * shifted).sum(-1)
    # d/d omega of weight * shifted, with d weight / d omega = -weight * 2 pi k / s
    d_bracket = (weight * (1 - shifted * 2 * math.pi * k / s[..., None])).sum(-1)
    return bracket, d_bracket


def log_density(omega: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """log f(omega; s), the log density of IGSO(3) relative to the uniform distribution on SO(3). float64."""
    omega, s = torch.broadcast_tensors(torch.as_tensor(omega, dtype=torch.float64), torch.as_tensor(s, dtype=torch.float64))
    bracket, _ = _images(omega, s)
    return (s / 8 + 0.5 * torch.log(2 * math.pi / s) - torch.log(s) - omega**2 / (2 * s)
            + torch.log(bracket) - torch.log(torch.sin(omega / 2)))


def d_log_density(omega: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """d/d omega log f(omega; s). The Riemannian gradient of log f at R0 @ exp(hat(omega u)) is this times u
    (in the frame of the rotation, i.e. the tangent vector R @ hat(u * this)). float64."""
    omega, s = torch.broadcast_tensors(torch.as_tensor(omega, dtype=torch.float64), torch.as_tensor(s, dtype=torch.float64))
    bracket, d_bracket = _images(omega, s)
    return -omega / s + d_bracket / bracket - 0.5 / torch.tan(omega / 2)


def angle_cdf(omega: torch.Tensor, s: torch.Tensor, points: int = 4096) -> torch.Tensor:
    """P(angle <= omega) under IGSO(3) with time s, by the trapezoidal rule (reference, for tests). float64."""
    omega = torch.as_tensor(omega, dtype=torch.float64)
    grid = torch.linspace(0, 1, points, dtype=torch.float64)[1:]
    full = grid * min(math.pi, 14 * math.sqrt(float(s)))  # the mass beyond 14 standard deviations is nil
    pdf_full = torch.exp(log_density(full, torch.as_tensor(s, dtype=torch.float64))) * (1 - torch.cos(full)) / math.pi
    total = torch.trapezoid(torch.cat([torch.zeros(1, dtype=torch.float64), pdf_full]), torch.cat([torch.zeros(1, dtype=torch.float64), full]))
    part = omega[..., None] * grid
    pdf = torch.exp(log_density(part, torch.as_tensor(s, dtype=torch.float64))) * (1 - torch.cos(part)) / math.pi
    zero = torch.zeros(*part.shape[:-1], 1, dtype=torch.float64)
    return torch.trapezoid(torch.cat([zero, pdf], -1), torch.cat([zero, part], -1)) / total


class AngleSampler:
    """Samples the angle of IGSO(3) for any s in [s_min, s_max] by inverting tabulated CDFs.

    The quantile function is tabulated on a log-spaced grid of s and a grid of probabilities, and interpolated
    linearly in both (in log s). The angle of a level is tabulated up to 14 standard deviations (or pi), so the
    small-s levels are resolved as finely as the large ones.
    """

    def __init__(self, s_min: float, s_max: float, levels: int = 1024, angles: int = 2048, quantiles: int = 4097):
        self.log_s = torch.linspace(math.log(s_min), math.log(s_max), levels, dtype=torch.float64)
        s = self.log_s.exp()
        top = torch.clamp(14 * s.sqrt(), max=math.pi)
        omega = top[:, None] * torch.linspace(0, 1, angles, dtype=torch.float64)[1:]  # [levels, angles - 1]
        pdf = torch.exp(log_density(omega, s[:, None])) * (1 - torch.cos(omega)) / math.pi
        omega = torch.cat([torch.zeros(levels, 1, dtype=torch.float64), omega], 1)
        pdf = torch.cat([torch.zeros(levels, 1, dtype=torch.float64), pdf], 1)
        cdf = torch.cat([torch.zeros(levels, 1, dtype=torch.float64),
                         torch.cumsum((pdf[:, 1:] + pdf[:, :-1]) / 2 * omega.diff(dim=1), 1)], 1)
        cdf = cdf / cdf[:, -1:]
        self.u = torch.linspace(0, 1, quantiles, dtype=torch.float64)
        # invert every level: omega at the probabilities self.u
        idx = torch.searchsorted(cdf, self.u.expand(levels, -1).contiguous(), right=False).clamp(1, angles - 1)
        c0, c1 = cdf.gather(1, idx - 1), cdf.gather(1, idx)
        w0, w1 = omega.gather(1, idx - 1), omega.gather(1, idx)
        frac = ((self.u - c0) / (c1 - c0).clamp_min(1e-300)).clamp(0, 1)
        self.table = w0 + frac * (w1 - w0)  # [levels, quantiles]
        self.s_min, self.s_max = s_min, s_max
        self._tables = {}

    def table_on(self, device, dtype) -> torch.Tensor:
        """The quantile table on `device` as `dtype` (the float64 one on the CPU is the master; copies are cached)."""
        key = (torch.device(device), dtype)
        if key not in self._tables:
            self._tables[key] = self.table.to(dtype).to(device)  # dtype on the CPU side: mps has no float64
        return self._tables[key]

    def sample(self, s: torch.Tensor, generator: torch.Generator = None, dtype=torch.float64) -> torch.Tensor:
        """One angle for every entry of s, on the device of s (the CPU for a number or a list), as `dtype` (float32
        on mps, which has no float64)."""
        s = torch.as_tensor(s).to(dtype)
        if (s < self.s_min * (1 - 1e-9)).any() or (s > self.s_max * (1 + 1e-9)).any():
            raise ValueError(f"s must be in [{self.s_min}, {self.s_max}].")
        position = (s.log() - self.log_s[0].item()) / (self.log_s[1] - self.log_s[0]).item()
        lo = position.floor().long().clamp(0, len(self.log_s) - 2)
        return self._draw(lo, (position - lo).clamp(0, 1), s.shape, s.device, dtype, generator)

    def sample_at(self, s: float, shape, device="cpu", dtype=torch.float64,
                  generator: torch.Generator = None) -> torch.Tensor:
        """Angles [*shape] of one Brownian time s for all, on `device` as `dtype`. The level of s is found on the CPU,
        so on a GPU nothing waits for the device (sample() checks the range of a tensor s, a synchronization)."""
        s = float(s)
        if not self.s_min * (1 - 1e-9) <= s <= self.s_max * (1 + 1e-9):
            raise ValueError(f"s must be in [{self.s_min}, {self.s_max}].")
        position = (math.log(s) - self.log_s[0].item()) / (self.log_s[1] - self.log_s[0]).item()
        lo = min(max(math.floor(position), 0), len(self.log_s) - 2)
        return self._draw(lo, min(max(position - lo, 0.0), 1.0), tuple(shape), device, dtype, generator)

    def _draw(self, lo, a, shape, device, dtype, generator) -> torch.Tensor:
        """Interpolation of the table between the levels lo and lo + 1 (weight a of the upper one) at uniform
        probabilities: the inverse CDF method."""
        table = self.table_on(device, dtype)
        u = torch.rand(shape, device=device, dtype=dtype, generator=generator) * (len(self.u) - 1)
        j = u.floor().long().clamp(0, len(self.u) - 2)
        b = u - j

        def at(level):
            return table[level, j] * (1 - b) + table[level, j + 1] * b

        return at(lo) * (1 - a) + at(lo + 1) * a


def random_axes(shape, generator: torch.Generator = None, device="cpu", dtype=torch.float64) -> torch.Tensor:
    """Uniform unit vectors [*shape, 3] (float64 on the CPU by default)."""
    v = torch.randn(*shape, 3, device=device, dtype=dtype, generator=generator)
    return v / v.norm(dim=-1, keepdim=True)
