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
import torch

from hp.models.dposer import DPoser


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
