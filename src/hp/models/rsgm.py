"""Riemannian score-based generative model (RSGM, De Bortoli et al. 2022) on SO(3)^J, as a PoseModel.

A port of the Brownian-motion RSGM of github.com/FlorianPfaff/riemannian-score-sde (JAX) to torch, with the
settings of its SO(3) experiment (config/experiment/so3.yaml): every joint diffuses by Brownian motion on SO(3)
with a linear beta schedule, the score network is the "Concat" MLP (5 x 512, sin) whose output is read as
Lie algebra coordinates (LieAlgebraGenerator), scaled by 1/std (std trick), and trained with denoising score
matching against the exact heat kernel (dsm0, like_w=false).

One difference: the JAX code has no SO(3) heat kernel, so there the target falls back to the small-time
(Varadhan) approximation. Here the target is the exact IGSO(3) score (hp/models/igso3.py) for every t.

Conventions. Rotations act with the metric where the distance of two rotations is their angle. A tangent vector
at x is given by its coordinates v in the frame of x: the vector x @ hat(v). The noising process is
dx = sqrt(beta(t)) dB, so x_t | x_0 is IGSO(3) around x_0 with time s(t) = int_0^t beta. At t = 1 it is close to
uniform (for beta_f = 6 the l=1 mode is down to exp(-s(1)) = 5%), and the likelihood treats it as uniform.

log_prob is the probability flow ODE dx/dt = -beta(t)/2 * score(x, t) integrated from eps to 1, plus the integral
of the exact divergence (torch.func, one forward-mode pass per coordinate). It is a density relative to the
uniform distribution on SO(3)^J, the same space as so3_flow, so the numbers are comparable.
"""
import math
from typing import Optional, Sequence

import roma
import torch
from torch import nn
from torch.func import jacfwd, vmap

from hp.models import igso3
from hp.models.base import PoseModel
from hp.models.registry import register_model


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


ACTIVATIONS = {"sin": Sin, "silu": nn.SiLU, "relu": nn.ReLU}


def move(x: torch.Tensor, device, dtype) -> torch.Tensor:
    """x on `device` as `dtype`, in two steps: on mps, changing device and dtype in one .to() sometimes
    returns garbage (torch 2.13), so the dtype is changed on the CPU side of the copy."""
    if torch.device(device).type == "cpu":
        return x.to("cpu").to(dtype)
    return x.to(dtype).to(device)


def exp_frame(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Move the rotations x [..., 3, 3] along the tangent vectors with coordinates v [..., 3]: x @ exp(hat(v))."""
    return x @ roma.rotvec_to_rotmat(v)


@register_model("rsgm")
class RSGM(PoseModel):
    has_log_prob = True
    density_space = "SO(3)^J"

    def __init__(
        self,
        num_joints: int,
        hidden: Sequence[int] = (512, 512, 512, 512, 512),
        activation: str = "sin",
        beta_0: float = 0.001,
        beta_f: float = 6.0,
        eps: float = 2e-4,  # the time the sampler stops and the likelihood starts at
        loss_t_min: float = 1e-3,  # the smallest time of the training loss (t0 + 1e-3 in the JAX code)
        sample_steps: int = 100,
        likelihood_steps: int = 100,
        weighting: str = "std",  # of the loss over time: "std" (std(t)^2, as the JAX code) or "likelihood" (beta(t))
        representation: str = "matrix",
        cond_dim: int = 0,
    ):
        if representation != "matrix":
            raise ValueError("RSGM works on rotation matrices directly; use representation='matrix'.")
        super().__init__(num_joints, "matrix", cond_dim)
        self.beta_0, self.beta_f, self.eps, self.loss_t_min = beta_0, beta_f, eps, loss_t_min
        self.sample_steps, self.likelihood_steps = sample_steps, likelihood_steps
        if weighting not in ("std", "likelihood"):
            raise ValueError(f"Unknown weighting '{weighting}'. Use 'std' or 'likelihood'.")
        self.weighting = weighting
        layers, width = [], 9 * num_joints + 1
        for h in hidden:
            layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
            width = h
        layers.append(nn.Linear(width, 3 * num_joints))
        self.net = nn.Sequential(*layers)
        s_lo, s_hi = self.s(torch.tensor(min(eps, loss_t_min))), self.s(torch.tensor(1.0))
        self.angles = igso3.AngleSampler(float(s_lo) * 0.999, float(s_hi) * 1.001)

    # ---- the noising process ----

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_0 + (self.beta_f - self.beta_0) * t

    def s(self, t: torch.Tensor) -> torch.Tensor:
        """Brownian time int_0^t beta (rescale_t of the JAX code)."""
        return self.beta_0 * t + 0.5 * (self.beta_f - self.beta_0) * t**2

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """The scale of the std trick: sqrt(1 - exp(-s)), as marginal_prob of the JAX code."""
        return torch.sqrt(-torch.expm1(-self.s(t)))

    def loss_times(self, u: torch.Tensor) -> torch.Tensor:
        """Times of the training loss from uniform numbers u in [0, 1] (float64).

        "std": t uniform on [loss_t_min, 1]. "likelihood": the loss weighted by beta(t) instead of std(t)^2, which
        makes it a bound on the negative log-likelihood (Song et al. 2021, "Maximum likelihood training of score-based
        diffusion models"). It is done by importance sampling: t is drawn with the density q(t) proportional to
        beta(t) / std(t)^2, so the loss itself stays the same (up to a constant factor, to which Adam is blind).
        In the Brownian time s the density is 1 / (1 - exp(-s)), with the CDF log(exp(s) - 1), inverted in closed form."""
        if self.weighting == "std":
            return self.loss_t_min + (1 - self.loss_t_min) * u
        t_lo = torch.tensor(self.loss_t_min, dtype=torch.float64)
        lo, hi = torch.log(torch.expm1(self.s(t_lo))), torch.log(torch.expm1(self.s(torch.ones_like(t_lo))))
        s = torch.log1p(torch.exp(lo + u * (hi - lo)))
        # s = beta_0 t + (beta_f - beta_0) t^2 / 2, solved for t
        return (-self.beta_0 + torch.sqrt(self.beta_0**2 + 2 * (self.beta_f - self.beta_0) * s)) / (self.beta_f - self.beta_0)

    # ---- the score ----

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score of p_t at x [B, J, 3, 3] and times t [B], as coordinates [B, J, 3] in the frame of x."""
        features = torch.cat([x.reshape(x.shape[0], -1), t[:, None].to(x.dtype)], dim=1)
        out = self.net(features) / self.std(t)[:, None].to(x.dtype)
        return out.reshape(x.shape[0], self.num_joints, 3)

    def loss(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Denoising score matching: E || std(t) * (score(x_t, t) - grad log p_t(x_t | x_0)) ||^2.

        In eval mode the times and the noise are drawn from a fixed seed, so that a validation loss is the same
        number every time (and two copies of a model give the same loss)."""
        self.check_cond(cond, len(pose))
        generator = None if self.training else torch.Generator().manual_seed(0)
        batch, joints = pose.shape[0], self.num_joints
        t = self.loss_times(torch.rand(batch, dtype=torch.float64, generator=generator))
        s = self.s(t)[:, None].expand(batch, joints)
        omega = self.angles.sample(s, generator)
        axis = igso3.random_axes((batch, joints), generator)
        x0 = move(pose.detach().reshape(batch, joints, 3, 3), "cpu", torch.float64)
        xt = exp_frame(x0, omega[..., None] * axis)
        target = (self.std(t)[:, None, None] * igso3.d_log_density(omega, s)[..., None] * axis)
        dtype = self.net[0].weight.dtype
        xt, target, t = (move(v, self.device, dtype) for v in (xt, target, t))
        scaled = self.score(xt, t) * self.std(t)[:, None, None]
        return ((scaled - target) ** 2).sum(dim=(1, 2)).mean()

    # ---- sampling: the time reversal, by geodesic Euler-Maruyama ----

    @torch.no_grad()
    def sample(self, n: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_cond(cond, n)
        dtype = self.net[0].weight.dtype
        x = roma.random_rotmat(n * self.num_joints, device=self.device, dtype=dtype).reshape(n, self.num_joints, 3, 3)
        times = torch.linspace(1.0, self.eps, self.sample_steps + 1, device=self.device, dtype=dtype)
        for t, t_next in zip(times[:-1], times[1:]):
            dt, beta = t - t_next, self.beta(t)
            step = beta * self.score(x, t.expand(n)) * dt + torch.sqrt(beta * dt) * torch.randn(n, self.num_joints, 3, device=self.device, dtype=dtype)
            x = exp_frame(x, step)
        return x

    # ---- the likelihood: probability flow ODE with the exact divergence ----

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The probability flow -beta(t)/2 * score, coordinates [B, J, 3] in the frame of x."""
        return -0.5 * self.beta(t)[:, None, None] * self.score(x, t)

    def velocity_and_divergence(self, x: torch.Tensor, t: torch.Tensor):
        """The velocity and its Riemannian divergence [B]. The frame fields x @ hat(e_i) are divergence free
        (Killing fields), so the divergence is sum_i d/de v_i(x @ exp(e * e_i)) at e = 0, the trace of a Jacobian."""

        def one(delta, xi, ti):  # delta [J, 3], xi [J, 3, 3], ti []
            v = self.velocity(exp_frame(xi, delta)[None], ti[None])[0]
            return v, v

        jacobian, v = vmap(jacfwd(one, has_aux=True))(torch.zeros(x.shape[0], self.num_joints, 3, device=x.device, dtype=x.dtype), x, t)
        divergence = jacobian.reshape(x.shape[0], 3 * self.num_joints, 3 * self.num_joints).diagonal(dim1=1, dim2=2).sum(-1)
        return v, divergence

    def log_prob(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """log p(pose) relative to the uniform distribution on SO(3)^J: midpoint steps of the ODE from eps to 1,
        log p_eps(x) = log p_1(x_1) + int div, with p_1 taken as uniform (log p_1 = 0)."""
        self.check_cond(cond, len(pose))
        dtype = self.net[0].weight.dtype
        x = move(pose.reshape(-1, self.num_joints, 3, 3), self.device, dtype)
        n = x.shape[0]
        times = torch.linspace(self.eps, 1.0, self.likelihood_steps + 1, device=self.device, dtype=dtype)
        log_p = torch.zeros(n, device=self.device, dtype=dtype)
        with torch.no_grad():
            for t, t_next in zip(times[:-1], times[1:]):
                h = t_next - t
                x_mid = exp_frame(x, 0.5 * h * self.velocity(x, t.expand(n)))
                v_mid, div_mid = self.velocity_and_divergence(x_mid, (t + 0.5 * h).expand(n))
                x = exp_frame(x, h * v_mid)
                log_p = log_p + h * div_mid
        return log_p
