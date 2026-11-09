"""Riemannian D-Flow for the models of the zoo that sample by an ODE: a pose problem is solved without task-specific
training by optimizing the start x_0 of the ODE so that its end x_1 = F(x_0) makes a loss small (Ben-Hamu et al. 2024;
on SO(3)^J as PoseD-Flow, Nadar, Foti, Birdal, CVPR 2026, their pose_completion.py):

    loss(x_1) + running_weight * mean_t sum_joints (3 - tr x_t),

the gradients going through all ODE steps; x_0 = start exp(hat(delta)) with Adam on delta. The start mixes noise with
a guess of the latent of the observations, as theirs. A prior wraps the ODE of one model:
  - FlowPrior: the pretrained PoseD-Flow (hp/models/posedflow.py), 21 joints: their ambient Euler steps from t = 0
    to 1 without projection, their projx at the end; the start projx(sqrt(1/4) F(y_0) + sqrt(3/4) noise), F(y_0) the
    flow run from the observed pose y_0 (their code does exactly this, from the data side);
  - ScorePrior: RSGM (hp/models/rsgm.py), its 19 joints: the probability flow ODE from t = 1 to eps by geodesic
    Euler steps; the start the same mix with uniform rotations, and as theirs the sampling ODE run from y_0 as if it
    were noise (invert=True: the latent of y_0 instead, the ODE from eps to 1).
The gradients go back through the steps by autograd (no adjoint ODE), so the memory grows with the number of steps.
Their RiemannianAdam (weight decay 1e-4 on the point) is here Adam on the tangent coordinates at the start.
"""
import math
from typing import Callable, Tuple

import roma
import torch

from hp.models.posedflow import PoseDFlowPretrained, projx
from hp.models.rsgm import RSGM, exp_frame


def trace_cost(x: torch.Tensor) -> torch.Tensor:
    """Their running cost of one point of a trajectory x [B, J, 3, 3]: sum over the joints of 3 - tr, mean over B."""
    return (3 - x.diagonal(dim1=-2, dim2=-1).sum(-1)).sum(-1).mean()


class FlowPrior:
    def __init__(self, model: PoseDFlowPretrained, ode_steps: int = 100):
        self.model, self.ode_steps, self.joints = model, ode_steps, 21

    def _times(self):
        return torch.linspace(0.0, 1.0, self.ode_steps + 1, device=self.model.device, dtype=self.model.dtype)

    def noise(self, n: int) -> torch.Tensor:
        return self.model.base_sample(n)

    @torch.no_grad()
    def guess(self, y0: torch.Tensor) -> torch.Tensor:
        n, times, x = y0.shape[0], self._times(), y0.reshape(y0.shape[0], -1)
        for t0, t1 in zip(times[:-1], times[1:]):
            x = x + (t1 - t0) * self.model.field_ambient(t0, x)
        return x.view(n, 21, 3, 3)  # not projected, as theirs

    def generate(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, times, x = x0.shape[0], self._times(), x0.reshape(x0.shape[0], -1)
        running = trace_cost(x0)
        for t0, t1 in zip(times[:-1], times[1:]):
            x = x + (t1 - t0) * self.model.field_ambient(t0, x)
            running = running + trace_cost(x.view(n, 21, 3, 3))
        return projx(x.view(n, 21, 3, 3)), running / (self.ode_steps + 1)


class ScorePrior:
    def __init__(self, model: RSGM, ode_steps: int = 100, invert: bool = False):
        self.model, self.ode_steps, self.joints, self.invert = model, ode_steps, model.num_joints, invert

    @property
    def dtype(self):
        return self.model.net[0].weight.dtype

    def _times(self, start: float, end: float):
        return torch.linspace(start, end, self.ode_steps + 1, device=self.model.device, dtype=self.dtype)

    def noise(self, n: int) -> torch.Tensor:
        return roma.random_rotmat(n * self.joints, device=self.model.device, dtype=self.dtype).view(n, self.joints, 3, 3)

    def _run(self, x: torch.Tensor, times, running: bool):
        n, cost = x.shape[0], trace_cost(x) if running else None
        for t0, t1 in zip(times[:-1], times[1:]):
            x = exp_frame(x, (t1 - t0) * self.model.velocity(x, t0.expand(n)))
            if running:
                cost = cost + trace_cost(x)
        return x, cost

    @torch.no_grad()
    def guess(self, y0: torch.Tensor) -> torch.Tensor:
        times = self._times(self.model.eps, 1.0) if self.invert else self._times(1.0, self.model.eps)
        return self._run(y0.to(self.model.device, self.dtype), times, False)[0]

    def generate(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, cost = self._run(x0, self._times(1.0, self.model.eps), True)
        return x1, cost / (self.ode_steps + 1)


def prior_of(model, ode_steps: int = 100):
    if isinstance(model, PoseDFlowPretrained):
        return FlowPrior(model, ode_steps)
    if isinstance(model, RSGM):
        return ScorePrior(model, ode_steps)
    raise TypeError(f"No D-Flow prior for {type(model).__name__}.")


def geodesic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Their geodesic_loss: the mean rotation angle between pred and target [..., 3, 3]."""
    trace = (pred.transpose(-1, -2) @ target).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps)).mean()


def dflow(prior, loss_fn: Callable[[torch.Tensor], torch.Tensor], start: torch.Tensor, steps: int = 200,
          lr: float = 0.1, running_weight: float = 1e-3) -> torch.Tensor:
    """x_1 [B, J, 3, 3] of the last forward pass (as theirs, before the last step of the optimizer)."""
    start = start.detach()
    delta = torch.zeros(*start.shape[:2], 3, device=start.device, dtype=start.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    x1 = None
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            x1, running = prior.generate(exp_frame(start, delta))
            (loss_fn(x1) + running_weight * running).backward()
            optimizer.step()
    return x1.detach()


def complete(prior, observed: torch.Tensor, visible: torch.Tensor, steps: int = 200, lr: float = 0.1,
             running_weight: float = 1e-3) -> torch.Tensor:
    """Poses [B, J, 3, 3] that keep the visible joints (bool [J]) of `observed` [B, J, 3, 3] (in the joints of the
    prior) and fill the others, as their pose completion: the start from the observed joints with the others at rest,
    the geodesic loss of the visible joints, the visible joints of the result replaced by the observed ones."""
    n, device = observed.shape[0], prior.model.device
    observed = observed.to(device, prior.noise(1).dtype)
    eye = torch.eye(3, device=device, dtype=observed.dtype).expand_as(observed)
    y0 = torch.where(visible.to(device)[None, :, None, None], observed, eye)
    start = projx(math.sqrt(0.25) * prior.guess(y0) + math.sqrt(0.75) * prior.noise(n))
    x1 = dflow(prior, lambda x: geodesic_loss(x[:, visible], observed[:, visible]), start, steps, lr, running_weight)
    return torch.where(visible.to(device)[None, :, None, None], observed, x1)
