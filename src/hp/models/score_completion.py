"""Pose completion by the optimization of DPoser (Lu et al., DPoser-X, ICCV 2025; their lib/algorithms/completion.py,
DPoserComp.optimize with its defaults, the method of the DPoser rows of PoseD-Flow's Tab. 2): the pose itself is
optimized, not a latent.

    x: the z-scored axis-angle vector [B, 3J]; the visible entries start at the observation, the hidden ones at
       N(0, I) (their create_mask, observation_type="noise")
    for step in 0 ... 199 (2 iterations `it` of 100 steps), with t linear from t_max = 0.15 down to t_min = 0.05:
        x_t = alpha(t) x + std(t) z,  z ~ N(0, I) fresh every step
        x0_hat = (x_t + std^2 score(x_t, t)) / alpha,  without gradient (one-step denoising)
        loss = 100 / (1 + it) * mean((x - y)^2 on the visible entries, 0 elsewhere)
             + 0.1 * (it + 1) * mean(0.5 sqrt(1 + alpha / std) (x - x0_hat)^2)
        Adam (lr 0.1, betas (0.9, 0.99)), one optimizer for all steps
    x on the visible entries replaced by the observation.

The prior term is score distillation: its gradient is proportional to the noise prediction minus z. Only the small
times of the diffusion are used (their truncated schedule), so the prior refines the pose locally; one forward pass
of the network per step, no gradient through it. A batch shares t (a scalar per step), as theirs.
"""
from typing import Optional

import roma
import torch

from hp.models import igso3
from hp.models.dposer import DPoser
from hp.models.rsgm import RSGM, exp_frame


@torch.no_grad()
def one_step_denoise(model: DPoser, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Their one_step_denoise: x0_hat = (x_t + std^2 score(x_t, t)) / alpha (Tweedie), x_t [B, 3J], t [B]."""
    alpha, std = model.marginal(t)
    return (x_t + std[:, None] ** 2 * model.score(x_t, t)) / alpha[:, None]


def dposer_complete(model: DPoser, observed: torch.Tensor, visible: torch.Tensor, init: torch.Tensor,
                    iterations: int = 2, steps_per_iter: int = 100, lr: float = 0.1, t_max: float = 0.15,
                    t_min: float = 0.05) -> torch.Tensor:
    """Poses [B, J, 3, 3] that keep the visible joints (bool [J]) of `observed` [B, J, 3, 3] and fill the others;
    `init` [B, 3J]: the start of the hidden entries (their N(0, I) noise), in the normalized space."""
    device, dtype = model.device, model.dtype
    n = observed.shape[0]
    y = model.normalize(observed.to(device, dtype))
    mask = visible.to(device).repeat_interleave(3)[None].expand_as(y)
    x = torch.where(mask, y, init.to(device, dtype)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
    total = iterations * steps_per_iter
    with torch.enable_grad():
        for it in range(iterations):
            for i in range(steps_per_iter):
                step = it * steps_per_iter + i
                t = torch.full((n,), t_min + (total - step - 1) / total * (t_max - t_min), device=device, dtype=dtype)
                alpha, std = model.marginal(t)
                x0_hat = one_step_denoise(model, alpha[:, None] * x.detach() + std[:, None] * torch.randn_like(x), t)
                weight = 0.5 * torch.sqrt(1 + alpha / std)[:, None]
                prior = (weight * (x - x0_hat) ** 2).mean()
                data = ((x * mask - y * mask) ** 2).mean()
                loss = 100 / (1 + it) * data + 0.1 * (it + 1) * prior
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return model.denormalize(torch.where(mask, y, x.detach()))


@torch.no_grad()
def rsgm_denoise(model: RSGM, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """One-step denoising on SO(3)^J, the Tweedie step of the heat kernel at small times:
    x0_hat = x_t exp(s(t) score(x_t, t)), s(t) the Brownian time (the variance of a coordinate; per joint with
    joint_scale), x_t [B, J, 3, 3]."""
    return exp_frame(x_t, model.joint_s(t)[..., None] * model.score(x_t, t))


def rsgm_complete(model: RSGM, observed: torch.Tensor, visible: torch.Tensor, init: torch.Tensor,
                  iterations: int = 2, steps_per_iter: int = 100, lr: float = 0.1, t_max: float = 0.15,
                  t_min: float = 0.05, scale: Optional[torch.Tensor] = None, data_weight: float = 100.0,
                  prior_weight: float = 0.1) -> torch.Tensor:
    """DPoser's optimization (dposer_complete) on SO(3)^J for RSGM: the pose x = start exp(scale * u), the visible
    joints starting at the observation and the hidden ones at `init` [B, J, 3, 3]; Adam on u [B, J, 3]. `scale` [J, 3]
    (None: 1) plays the part of DPoser's z-scoring: with the spread of the data per joint, a step of Adam moves every
    joint by a similar fraction of its own range. Every step:
        x_t: x noised by the IGSO(3) of Brownian time s(t) per joint (the forward process of RSGM)
        x0_hat = rsgm_denoise(x_t), without gradient
        loss = data_weight / (1 + it) * sum over the visible joints of d(x, y)^2 / (B J)
             + prior_weight * (it + 1) * mean over the joints of 0.5 sqrt(1 + 1 / std(t)) d(x, x0_hat)^2
    (their weights: data_weight 100, prior_weight 0.1)
    d the geodesic distance (the angle), so both terms are squared angles, as theirs are squared normalized
    coordinates; their weight 0.5 sqrt(1 + alpha / sigma) with alpha ~ 1 at small t and RSGM's std. The times t
    are RSGM's (their scale differs from DPoser's), linear from t_max down to t_min over the steps."""
    device, dtype = model.device, model.dtype
    n, joints = observed.shape[:2]
    y = observed.to(device, dtype)
    seen = visible.to(device)[None, :, None, None]
    start = torch.where(seen, y, init.to(device, dtype)).detach()
    delta = start.new_zeros(n, joints, 3).requires_grad_(True)
    scale = torch.ones(joints, 3) if scale is None else torch.as_tensor(scale).reshape(joints, 3)
    scale = scale.to(device, dtype)
    optimizer = torch.optim.Adam([delta], lr=lr, betas=(0.9, 0.99))
    total = iterations * steps_per_iter
    vis = visible.to(device)
    with torch.enable_grad():
        for it in range(iterations):
            for i in range(steps_per_iter):
                step = it * steps_per_iter + i
                t = torch.full((n,), t_min + (total - step - 1) / total * (t_max - t_min), dtype=torch.float64)
                x = exp_frame(start, scale * delta)
                s = model.joint_s(t)
                noise = model.angles.sample(s)[..., None] * igso3.random_axes((n, joints))
                t = t.to(device, dtype)
                x0_hat = rsgm_denoise(model, exp_frame(x.detach(), noise.to(device, dtype)), t)
                weight = 0.5 * torch.sqrt(1 + 1 / model.joint_std(t))
                prior = (weight * _angle2(x, x0_hat)).mean()
                data = (_angle2(x[:, vis], y[:, vis])).sum() / (n * joints)
                loss = data_weight / (1 + it) * data + prior_weight * (it + 1) * prior
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return torch.where(seen, y, exp_frame(start, scale * delta.detach()))


def _angle2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The squared rotation angle between a and b [..., 3, 3], smooth at 0 (|log(a^T b)|^2)."""
    return roma.rotmat_to_rotvec(a.transpose(-1, -2) @ b).pow(2).sum(-1)
