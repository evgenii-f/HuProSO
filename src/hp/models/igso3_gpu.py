"""IGSO(3), the heat kernel that RSGM diffuses with, on any device and in float32 (for mps and cuda).

hp/models/igso3.py computes it in float64 on the CPU, so RSGM copies every training batch to the CPU and back. This
module computes the same functions (same conventions, see igso3.py) with formulas that stay accurate in float32:

- d/d omega log f has a removable 1/omega singularity: the 1/omega of the image sum cancels the -cot(omega/2)/2 of
  the Jacobian 1/sin(omega/2). At omega = 1e-6 both terms are 1e6 large and float32 keeps nothing of their difference.
  Here the cancellation is done analytically: g(omega) = 1/omega - cot(omega/2)/2 is smooth, and computed by its
  Taylor series at small omega.
- The images k and -k are summed in pairs, in closed form (cosh, sinh of u = 2 pi k omega / s). Term by term they
  cancel at small omega (the sum is odd in omega), which is what broke the image sum for large s in float32.
  The exponents are combined before exp, so nothing overflows for small s.
- For s >= S_SWITCH the eigenfunction series is used, as a cosine series: with the Dirichlet kernel
  sin((l + 1/2) omega) / sin(omega / 2) = 1 + 2 sum_{m <= l} cos(m omega) it becomes f = sum_m c_m cos(m omega),
  which has no singularity at all. It converges fast where the image sum needs many terms.
- The angle density of the sampler uses 2 sin^2(omega / 2) instead of 1 - cos(omega) (0 in float32 at omega = 1e-4).

Both forms are evaluated for every entry and one is picked by torch.where: there is no data-dependent control flow,
so nothing waits for the device (no host syncs). Every function works in the dtype and on the device of its input.
"""
import math

import torch
from torch import nn

S_SWITCH = 1.0  # the image sum below, the cosine series from here on
PAIRS = (1, 2)  # image pairs +-k; for s < 1 the pair k = 3 is below exp(-118) for every omega in [0, pi]
SERIES_TERMS = 10  # l <= 10 of the series; for s >= 1 the first term left out is below exp(-66)
SMALL_U = 0.5  # below it the pair terms use their Taylor series
SMALL_OMEGA = 0.5  # below it g(omega) uses its Taylor series


def _g(omega: torch.Tensor) -> torch.Tensor:
    """1/omega - cot(omega/2)/2 = omega/12 + omega^3/720 + omega^5/30240 + omega^7/1209600 + ..., smooth at 0."""
    w2 = omega * omega
    series = omega * (1 / 12 + w2 * (1 / 720 + w2 * (1 / 30240 + w2 / 1209600)))
    safe = omega.clamp_min(SMALL_OMEGA)
    direct = 1 / safe - 0.5 / torch.tan(safe / 2)
    return torch.where(omega < SMALL_OMEGA, series, direct)


def _pairs(omega: torch.Tensor, s: torch.Tensor):
    """The image sum relative to its k = 0 term, (B, N): with bracket = sum_k (-1)^k (omega + 2 pi k)
    exp(-((omega + 2 pi k)^2 - omega^2) / (2 s)) as in igso3._images,
        B = bracket / omega,   N = (omega * d bracket / d omega - bracket) / omega^2,
    both smooth in omega (N is odd, B even). For the pair +-k, with A = 2 pi k and u = A omega / s:
        B_k = 2 (-1)^k e^{-A^2/2s} (cosh u - (A^2/s) sinh(u)/u),
        N_k = 2 (-1)^k e^{-A^2/2s} A ((A/s)^2 (sinh u - u cosh u)/u^2 + sinh(u)/s).
    e^{-A^2/2s} cosh u = (E+ + E-)/2 with E+- = exp(A (+-omega - A/2) / s) <= 1: no overflow for small s."""
    k = torch.tensor(PAIRS, device=omega.device, dtype=omega.dtype)
    A = 2 * math.pi * k
    w, s_ = omega[..., None], s[..., None]
    u = A * w / s_
    e_plus, e_minus = torch.exp(A * (w - A / 2) / s_), torch.exp(-A * (w + A / 2) / s_)
    ch, sh = (e_plus + e_minus) / 2, (e_plus - e_minus) / 2  # times e^{-A^2/2s}
    eg = torch.exp(-A * A / (2 * s_))
    small = u < SMALL_U
    u_safe = u.clamp_min(SMALL_U)
    u2 = u * u
    sinhc = torch.where(small, eg * (1 + u2 * (1 / 6 + u2 * (1 / 120 + u2 / 5040))), sh / u_safe)
    # (sinh u - u cosh u) / u^2 = -u/3 - u^3/30 - u^5/840 - u^7/45360 - ...
    q = torch.where(small, -eg * u * (1 / 3 + u2 * (1 / 30 + u2 * (1 / 840 + u2 / 45360))), (sh - u * ch) / u_safe**2)
    sign = 2 * (1 - 2 * (k % 2))  # 2 (-1)^k
    b = 1 + (sign * (ch - A * A / s_ * sinhc)).sum(-1)
    n = (sign * A * ((A / s_) ** 2 * q + sh / s_)).sum(-1)
    return b, n


def _series(omega: torch.Tensor, s: torch.Tensor):
    """f and d f / d omega from the cosine series f = sum_m c_m cos(m omega), with w_l = (2l + 1) e^{-l(l+1)s/2},
    c_0 = sum_l w_l and c_m = 2 sum_{l >= m} w_l."""
    l = torch.arange(SERIES_TERMS + 1, device=omega.device, dtype=omega.dtype)
    w = (2 * l + 1) * torch.exp(-l * (l + 1) * s[..., None] / 2)
    tail = w.flip(-1).cumsum(-1).flip(-1)
    c = torch.cat([tail[..., :1], 2 * tail[..., 1:]], -1)
    mw = l * omega[..., None]
    return (c * torch.cos(mw)).sum(-1), -(c * l * torch.sin(mw)).sum(-1)


def _split(omega, s):
    omega, s = torch.broadcast_tensors(omega, s)
    switch = torch.as_tensor(S_SWITCH, dtype=s.dtype, device=s.device)
    return omega, s, s < S_SWITCH, torch.minimum(s, switch), torch.maximum(s, switch)


def log_density(omega: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """log f(omega; s), the log density of IGSO(3) relative to the uniform distribution on SO(3)."""
    omega, s, low, s_img, s_ser = _split(omega, s)
    b, _ = _pairs(omega, s_img)
    # log(bracket) - log(sin(omega/2)) = log B + log(omega / sin(omega/2)), and sinc(x) = sin(pi x) / (pi x)
    images = (s_img / 8 + 0.5 * torch.log(2 * math.pi / s_img) - torch.log(s_img) - omega**2 / (2 * s_img)
              + torch.log(b) + math.log(2) - torch.log(torch.sinc(omega / (2 * math.pi))))
    f, _ = _series(omega, s_ser)
    return torch.where(low, images, torch.log(f))


def d_log_density(omega: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """d/d omega log f(omega; s): -omega/s + g(omega) + N/B from the images, f'/f from the series."""
    omega, s, low, s_img, s_ser = _split(omega, s)
    b, n = _pairs(omega, s_img)
    images = -omega / s_img + _g(omega) + n / b
    f, df = _series(omega, s_ser)
    return torch.where(low, images, df / f)


class AngleSampler(nn.Module):
    """Samples the angle of IGSO(3) for any s in [s_min, s_max], like igso3.AngleSampler, on the device of the module.

    The quantile table is built once in float64 on the CPU and kept as a non-persistent buffer in `dtype`, so it
    follows .to(device) and is not part of the state dict. sample() checks nothing on the host: s outside the table
    is clamped to its ends (igso3.AngleSampler raises, which needs a host sync)."""

    def __init__(self, s_min: float, s_max: float, levels: int = 1024, angles: int = 2048, quantiles: int = 4097,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        log_s = torch.linspace(math.log(s_min), math.log(s_max), levels, dtype=torch.float64)
        s = log_s.exp()
        top = torch.clamp(14 * s.sqrt(), max=math.pi)
        omega = top[:, None] * torch.linspace(0, 1, angles, dtype=torch.float64)  # [levels, angles], from 0
        pdf = torch.exp(log_density(omega, s[:, None].expand_as(omega))) * 2 * torch.sin(omega / 2) ** 2 / math.pi
        cdf = torch.cat([torch.zeros(levels, 1, dtype=torch.float64),
                         torch.cumsum((pdf[:, 1:] + pdf[:, :-1]) / 2 * omega.diff(dim=1), 1)], 1)
        cdf = cdf / cdf[:, -1:]
        u = torch.linspace(0, 1, quantiles, dtype=torch.float64)
        idx = torch.searchsorted(cdf, u.expand(levels, -1).contiguous(), right=False).clamp(1, angles - 1)
        c0, c1 = cdf.gather(1, idx - 1), cdf.gather(1, idx)
        w0, w1 = omega.gather(1, idx - 1), omega.gather(1, idx)
        frac = ((u - c0) / (c1 - c0).clamp_min(1e-300)).clamp(0, 1)
        self.register_buffer("table", (w0 + frac * (w1 - w0)).to(dtype), persistent=False)  # [levels, quantiles]
        self.log_s_min, self.log_s_step = math.log(s_min), (math.log(s_max) - math.log(s_min)) / (levels - 1)
        self.s_min, self.s_max = s_min, s_max

    def sample(self, s: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        """One angle for every entry of s, on the device and in the dtype of the table."""
        levels, quantiles = self.table.shape
        s = s.to(self.table)
        position = ((s.log() - self.log_s_min) / self.log_s_step).clamp(0, levels - 1)
        lo = position.floor().long().clamp(0, levels - 2)
        a = position - lo
        u = torch.rand(s.shape, generator=generator, device=s.device, dtype=s.dtype) * (quantiles - 1)
        j = u.floor().long().clamp(0, quantiles - 2)
        b = u - j

        def at(level):
            return self.table[level, j] * (1 - b) + self.table[level, j + 1] * b

        return at(lo) * (1 - a) + at(lo + 1) * a


def random_axes(shape, generator: torch.Generator = None, device=None, dtype=torch.float32) -> torch.Tensor:
    """Uniform unit vectors [*shape, 3] on `device`."""
    v = torch.randn(*shape, 3, generator=generator, device=device, dtype=dtype)
    return v / v.norm(dim=-1, keepdim=True)
