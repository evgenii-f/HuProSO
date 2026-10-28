"""The pretrained prior of PoseD-Flow (Nadar, Foti, Birdal, CVPR 2026; github.com/circle-group/PoseDFlow), as a PoseModel.

Their model is manifm (Chen & Lipman 2024) on SO(3)^21, the SMPL body joints 1 ... 21 (ours are the same without the
toes, SMPL joints 10 and 11). This file rebuilds its network without their environment and loads the weights of
their Lightning checkpoint (tensors only, torch.load(weights_only=True)):

    u = tMLP(t, x)                 x: the 21 rotation matrices flattened [B, 189]; 4 layers of 512, ConcatLinear_v2
                                   (a linear layer plus a bias linear in t) and TimeDependentSwish (x sigmoid(beta(t) x))
    v = x @ skew(x^T u)            ProjectToTangent of SO3Power: a tangent vector at x, in the ambient R^189

Time runs from the noise (t = 0) to the data (t = 1), as PoseRFM. Note the noise of their code: SO3.random draws
exp(n), n ~ N(0, I) in axis-angle per joint (a wrapped normal of 1 rad), not the uniform distribution, although their
log-likelihood uses the uniform base density -21 log(8 pi^2). log_prob here uses the density of the noise they
actually sampled from (base="wrapped_normal", the right one), or the uniform as their code (base="uniform").

Sampling is theirs (pose_generation.py, ManifoldFMLitModule.sample): Euler or midpoint steps in the ambient space
from t = 0 to 1, without projection on the way, and a projection onto SO(3) (QR) at the end. Their script defaults to
Euler with 1000 steps; the paper reports the midpoint solver with 100 steps.

`sample` returns our 19 joints (their joints without the toes). The density of a CNF cannot be marginalized over the
toes, so the model has no log_prob on 19 joints; `log_prob_21` gives the density of a pose of all 21 joints relative to
the uniform distribution on SO(3)^21 (comparable only with other models on these 21 joints).
"""
import math
from pathlib import Path
from typing import Optional

import roma
import torch
from torch import nn
from torch.func import jacfwd, vmap

from hp.models.base import PoseModel
from hp.models.rsgm import exp_frame

NUM_JOINTS = 21  # SMPL joints 1 ... 21
TOES = (9, 10)  # LeftToeBase (SMPL 10) and RightToeBase (SMPL 11) among the 21
KEEP = [j for j in range(NUM_JOINTS) if j not in TOES]  # our 19 joints (AmassJoints4Learning)


class ConcatLinear(nn.Module):
    """ConcatLinear_v2 of manifm: W x + b + w t."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))


class TimeSwish(nn.Module):
    """TimeDependentSwish of manifm: x sigmoid(beta(t) x)."""

    def __init__(self, dim):
        super().__init__()
        hidden = min(64, dim * 4)
        self.beta = nn.Sequential(nn.Linear(1, hidden), nn.Softplus(), nn.Linear(hidden, dim), nn.Softplus())

    def forward(self, t, x):
        return x * torch.sigmoid(x * self.beta(t.reshape(-1, 1)))


class Wrapped(nn.Module):
    """DiffEqWrapper: keeps the key names of their state dict (layers.i.module....)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, t, x):
        return self.module(t, x)


class TMLP(nn.Module):
    """tMLP of manifm (without Fourier features)."""

    def __init__(self, dim, d_model=512, num_layers=4):
        super().__init__()
        layers = [ConcatLinear(dim, d_model)]
        for _ in range(num_layers - 2):
            layers += [TimeSwish(d_model), ConcatLinear(d_model, d_model)]
        layers += [TimeSwish(d_model), ConcatLinear(d_model, dim)]
        self.layers = nn.ModuleList(Wrapped(layer) for layer in layers)

    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x


def skew(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a - a.transpose(-1, -2))


def vee(a: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] skew-symmetric -> [..., 3]."""
    return torch.stack([a[..., 2, 1], a[..., 0, 2], a[..., 1, 0]], dim=-1)


def projx(x: torch.Tensor) -> torch.Tensor:
    """Their projection onto SO(3) (SO3.projx): the Q of a QR decomposition with the signs of diag(R) taken out."""
    q, r = torch.linalg.qr(x)
    s = torch.diagonal(r, dim1=-2, dim2=-1).clone()
    s[s == 0] = 1
    return q * (s / s.abs()).unsqueeze(-2)


def wrapped_normal_log_density(rotations: torch.Tensor, terms: int = 2) -> torch.Tensor:
    """log density [...] of exp(n), n ~ N(0, I_3), relative to the uniform distribution on SO(3), at rotations [..., 3, 3].

    In exponential coordinates the normalized Haar measure is 2 (1 - cos r) / r^2 d^3v / (8 pi^2), r = |v|. The
    preimages of a rotation by theta about u are v_k = (theta + 2 pi k) u, k in Z, so

        p(R) = 8 pi^2 sum_k N(v_k) (theta + 2 pi k)^2 / (4 sin^2(theta / 2)).
    """
    theta = roma.rotmat_to_rotvec(rotations.double()).norm(dim=-1).clamp_min(1e-6)
    k = torch.arange(-terms, terms + 1, dtype=theta.dtype, device=theta.device)
    r = theta[..., None] + 2 * math.pi * k  # signed magnitudes of the preimages
    log_terms = -0.5 * r**2 - 1.5 * math.log(2 * math.pi) + 2 * torch.log(r.abs().clamp_min(1e-300))
    log_jac = math.log(8 * math.pi**2) - math.log(4.0) - 2 * torch.log(torch.sin(theta / 2))
    return (torch.logsumexp(log_terms, dim=-1) + log_jac).to(rotations.dtype)


class PoseDFlowPretrained(PoseModel):
    """Their checkpoint as a PoseModel of our 19 joints (sample only; log_prob_21 on their 21 joints)."""

    name = "posedflow_pretrained"  # not in the registry: it is loaded from their checkpoint, not trained here
    has_log_prob = False  # on our 19 joints: no (see log_prob_21)
    density_space = "SO(3)^21"

    def __init__(
        self,
        checkpoint: str,
        weights: str = "ema",  # "ema" (the shadow parameters) or "raw" (what their pose_generation.py samples with,
                               # as it never calls eval(), which would swap in the EMA)
        method: str = "midpoint",  # their solvers: "euler" or "midpoint"
        sample_steps: int = 100,
        likelihood_steps: int = 100,
        base: str = "wrapped_normal",  # log_prob_21: the density of their noise, or "uniform" as their code
        d_model: int = 512,
        num_layers: int = 4,
    ):
        if weights not in ("ema", "raw") or method not in ("euler", "midpoint") or base not in ("wrapped_normal", "uniform"):
            raise ValueError(f"Bad option: weights={weights}, method={method}, base={base}.")
        super().__init__(len(KEEP), "matrix")
        self.method, self.sample_steps, self.likelihood_steps, self.base = method, sample_steps, likelihood_steps, base
        self.net = TMLP(9 * NUM_JOINTS, d_model, num_layers)
        state = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)["state_dict"]
        prefix = "model.model.vecfield.vecfield."
        self.net.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
        if weights == "ema":
            shadow = [state[f"model.shadow_params.{i}"] for i in range(len(list(self.net.parameters())))]
            with torch.no_grad():
                for p, s in zip(self.net.parameters(), shadow, strict=True):
                    p.copy_(s)
        self.requires_grad_(False)
        self.build_args = dict(checkpoint=str(checkpoint), weights=weights, method=method, sample_steps=sample_steps,
                               likelihood_steps=likelihood_steps, base=base, d_model=d_model, num_layers=num_layers)

    @property
    def dtype(self) -> torch.dtype:
        return self.net.layers[0].module._layer.weight.dtype

    def loss(self, pose, cond=None):
        raise NotImplementedError("The pretrained PoseD-Flow model is not trained here.")

    # ---- the field ----

    def field_ambient(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Their vector field at x [B, 189] (not necessarily on SO(3)) and t [] or [B], in R^189."""
        b = x.shape[0]
        u = self.net(t.expand(b) if t.ndim == 0 else t, x)
        xm = x.view(b, NUM_JOINTS, 3, 3)
        return (xm @ skew(xm.transpose(-1, -2) @ u.view(b, NUM_JOINTS, 3, 3))).reshape(b, -1)

    def field_frame(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The field at rotations x [B, 21, 3, 3], as coordinates [B, 21, 3] in the frame of x (x @ hat(v))."""
        b = x.shape[0]
        u = self.net(t, x.reshape(b, -1)).view(b, NUM_JOINTS, 3, 3)
        return vee(skew(x.transpose(-1, -2) @ u))

    # ---- sampling, as theirs ----

    def base_sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Their noise: exp(N(0, I)) per joint, [n, 21, 3, 3]."""
        noise = torch.randn(n, NUM_JOINTS, 3, generator=generator, device=self.device, dtype=self.dtype)
        return roma.rotvec_to_rotmat(noise)

    @torch.no_grad()
    def sample_21(self, n: int, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """n poses of all 21 joints [n, 21, 3, 3]: their ambient solver from t = 0 to 1, projx at the end."""
        x = (self.base_sample(n) if x0 is None else x0.to(self.device, self.dtype)).reshape(n, -1)
        times = torch.linspace(0.0, 1.0, self.sample_steps + 1, device=self.device, dtype=self.dtype)
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            v = self.field_ambient(t0, x)
            if self.method == "euler":
                x = x + dt * v
            else:
                x = x + dt * self.field_ambient(t0 + 0.5 * dt, x + 0.5 * dt * v)
        return projx(x.view(n, NUM_JOINTS, 3, 3))

    def sample(self, n: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_cond(cond, n)
        return self.sample_21(n)[:, KEEP]

    # ---- the likelihood on SO(3)^21 ----

    def velocity_and_divergence(self, x: torch.Tensor, t: torch.Tensor):
        def one(delta, xi, ti):
            v = self.field_frame(exp_frame(xi, delta)[None], ti[None])[0]
            return v, v

        zeros = torch.zeros(x.shape[0], NUM_JOINTS, 3, device=x.device, dtype=x.dtype)
        jacobian, v = vmap(jacfwd(one, has_aux=True))(zeros, x, t)
        n = 3 * NUM_JOINTS
        return v, jacobian.reshape(x.shape[0], n, n).diagonal(dim1=1, dim2=2).sum(-1)

    @torch.no_grad()
    def log_prob_21(self, pose: torch.Tensor) -> torch.Tensor:
        """log p [B] of poses of all 21 joints [B, 21, 3, 3] relative to the uniform distribution on SO(3)^21:
        the ODE backwards from t = 1 with geodesic midpoint steps and the exact divergence (as PoseRFM)."""
        x = pose.reshape(-1, NUM_JOINTS, 3, 3).to(self.device, self.dtype)
        n = x.shape[0]
        times = torch.linspace(1.0, 0.0, self.likelihood_steps + 1, device=self.device, dtype=self.dtype)
        integral = torch.zeros(n, device=self.device, dtype=self.dtype)
        for t, t_next in zip(times[:-1], times[1:]):
            h = t_next - t
            x_mid = exp_frame(x, 0.5 * h * self.field_frame(x, t.expand(n)))
            v_mid, div_mid = self.velocity_and_divergence(x_mid, (t + 0.5 * h).expand(n))
            x = exp_frame(x, h * v_mid)
            integral = integral - h * div_mid
        base = wrapped_normal_log_density(x).sum(-1) if self.base == "wrapped_normal" else torch.zeros_like(integral)
        return base - integral
