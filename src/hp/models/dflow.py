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
    were noise (invert=True: the latent of y_0 instead, the ODE from eps to 1). The ODE steps can also be those of
    the RSGM samplers: sampler="heun" (two network calls per step) and schedule="edm" (the times of RSGM.sample_times);
    the running cost is taken at the ends of the steps either way;
  - DPoserPrior: DPoser (hp/models/dposer.py), a Euclidean prior: the probability flow ODE of its sub-VP SDE on the
    z-scored axis-angle vectors R^(3J), from t = 1 to sample_eps by Euler steps. The latent lives in R^(3J): x_0 =
    start + delta, N(0, I) noise, and the start sqrt(1/4) guess + sqrt(3/4) noise without a projection (the mix of
    D-Flow, which keeps a standard normal); the guess is the sampling ODE run from the normalized y_0
    (invert=True: the latent of y_0). The loss and the running cost see the trajectory decoded to rotations.
The gradients go back through the steps by autograd (no adjoint ODE), so the memory grows with the number of steps.
Their RiemannianAdam (weight decay 1e-4 on the point) is here Adam on the tangent coordinates at the start (on delta in
R^(3J) for DPoser).
"""
import math
from typing import Callable, Tuple

import roma
import torch

from hp.models.dposer import DPoser
from hp.models.posedflow import PoseDFlowPretrained, projx
from hp.models.rsgm import RSGM, exp_frame


def trace_cost(x: torch.Tensor) -> torch.Tensor:
    """Their running cost of one point of a trajectory x [B, J, 3, 3]: sum over the joints of 3 - tr, mean over B."""
    return (3 - x.diagonal(dim1=-2, dim2=-1).sum(-1)).sum(-1).mean()


class _RotationLatent:
    """The latent of the priors on SO(3)^J: their start projx(sqrt(1/4) guess + sqrt(3/4) noise), steps x exp(delta)."""

    def start(self, y0: torch.Tensor) -> torch.Tensor:
        return projx(math.sqrt(0.25) * self.guess(y0) + math.sqrt(0.75) * self.noise(y0.shape[0]))

    @staticmethod
    def zero_delta(start: torch.Tensor) -> torch.Tensor:
        return start.new_zeros(*start.shape[:2], 3)

    @staticmethod
    def shift(start: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return exp_frame(start, delta)


class FlowPrior(_RotationLatent):
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


class ScorePrior(_RotationLatent):
    def __init__(self, model: RSGM, ode_steps: int = 100, invert: bool = False, sampler: str = "euler",
                 schedule: str = "uniform"):
        if sampler not in ("euler", "heun"):
            raise ValueError(f"Unknown sampler '{sampler}' for D-Flow. Use 'euler' or 'heun' (an ODE).")
        if schedule not in ("uniform", "edm"):
            raise ValueError(f"Unknown schedule '{schedule}'. Use 'uniform' or 'edm'.")
        self.model, self.ode_steps, self.joints, self.invert = model, ode_steps, model.num_joints, invert
        self.sampler, self.schedule = sampler, schedule

    @property
    def dtype(self):
        return self.model.net[0].weight.dtype

    def _times(self, backwards: bool = True):
        """ode_steps + 1 times from 1 to eps (backwards, the sampling direction) or from eps to 1."""
        if self.schedule == "uniform":
            ends = (1.0, self.model.eps) if backwards else (self.model.eps, 1.0)
            return torch.linspace(*ends, self.ode_steps + 1, device=self.model.device, dtype=self.dtype)
        times = self.model.sample_times(self.ode_steps, schedule=self.schedule).to(self.model.device, self.dtype)
        return times if backwards else times.flip(0)

    def noise(self, n: int) -> torch.Tensor:
        return roma.random_rotmat(n * self.joints, device=self.model.device, dtype=self.dtype).view(n, self.joints, 3, 3)

    def _run(self, x: torch.Tensor, times, running: bool):
        n, cost = x.shape[0], trace_cost(x) if running else None
        for t0, t1 in zip(times[:-1], times[1:]):
            h = t1 - t0
            v = self.model.velocity(x, t0.expand(n))
            if self.sampler == "heun":
                v = 0.5 * (v + self.model.velocity(exp_frame(x, h * v), t1.expand(n)))
            x = exp_frame(x, h * v)
            if running:
                cost = cost + trace_cost(x)
        return x, cost

    @torch.no_grad()
    def guess(self, y0: torch.Tensor) -> torch.Tensor:
        return self._run(y0.to(self.model.device, self.dtype), self._times(not self.invert), False)[0]

    def generate(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, cost = self._run(x0, self._times(), True)
        return x1, cost / (self.ode_steps + 1)


class DPoserPrior:
    def __init__(self, model: DPoser, ode_steps: int = 100, invert: bool = False):
        self.model, self.ode_steps, self.joints, self.invert = model, ode_steps, model.num_joints, invert

    @property
    def dtype(self):
        return self.model.dtype

    def _times(self, backwards: bool = True):
        ends = (1.0, self.model.sample_eps) if backwards else (self.model.sample_eps, 1.0)
        return torch.linspace(*ends, self.ode_steps + 1, device=self.model.device, dtype=self.dtype)

    def _velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The probability flow dx/dt = -beta x / 2 - g^2 score / 2 of the sub-VP SDE."""
        m = self.model
        return -0.5 * m.beta(t)[:, None] * x - 0.5 * m.diffusion(t)[:, None] ** 2 * m.score(x, t)

    def noise(self, n: int) -> torch.Tensor:
        return torch.randn(n, 3 * self.joints, device=self.model.device, dtype=self.dtype)

    def _run(self, x: torch.Tensor, times, running: bool):
        n = x.shape[0]
        cost = trace_cost(self.model.denormalize(x)) if running else None
        for t0, t1 in zip(times[:-1], times[1:]):
            x = x + (t1 - t0) * self._velocity(x, t0.expand(n))
            if running:
                cost = cost + trace_cost(self.model.denormalize(x))
        return x, cost

    @torch.no_grad()
    def guess(self, y0: torch.Tensor) -> torch.Tensor:
        x = self.model.normalize(y0.to(self.model.device, self.dtype))
        return self._run(x, self._times(not self.invert), False)[0]

    def start(self, y0: torch.Tensor) -> torch.Tensor:
        return math.sqrt(0.25) * self.guess(y0) + math.sqrt(0.75) * self.noise(y0.shape[0])

    @staticmethod
    def zero_delta(start: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(start)

    @staticmethod
    def shift(start: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return start + delta

    def generate(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, cost = self._run(x0, self._times(), True)
        return self.model.denormalize(x1), cost / (self.ode_steps + 1)


def prior_of(model, ode_steps: int = 100, **options):
    """The D-Flow prior of a model; options go to its prior class (ScorePrior: sampler, schedule, invert;
    DPoserPrior: invert)."""
    if isinstance(model, PoseDFlowPretrained):
        if options:
            raise ValueError(f"The PoseD-Flow prior takes no options, got {sorted(options)}.")
        return FlowPrior(model, ode_steps)
    if isinstance(model, RSGM):
        return ScorePrior(model, ode_steps, **options)
    if isinstance(model, DPoser):
        return DPoserPrior(model, ode_steps, **options)
    raise TypeError(f"No D-Flow prior for {type(model).__name__}.")


def geodesic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Their geodesic_loss: the mean rotation angle between pred and target [..., 3, 3]."""
    trace = (pred.transpose(-1, -2) @ target).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps)).mean()


def dflow(prior, loss_fn: Callable[[torch.Tensor], torch.Tensor], start: torch.Tensor, steps: int = 200,
          lr: float = 0.1, running_weight: float = 1e-3) -> torch.Tensor:
    """x_1 [B, J, 3, 3] of the last forward pass (as theirs, before the last step of the optimizer)."""
    start = start.detach()
    delta = prior.zero_delta(start).requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    x1 = None
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            x1, running = prior.generate(prior.shift(start, delta))
            (loss_fn(x1) + running_weight * running).backward()
            optimizer.step()
    return x1.detach()


def complete(prior, observed: torch.Tensor, visible: torch.Tensor, steps: int = 200, lr: float = 0.1,
             running_weight: float = 1e-3) -> torch.Tensor:
    """Poses [B, J, 3, 3] that keep the visible joints (bool [J]) of `observed` [B, J, 3, 3] (in the joints of the
    prior) and fill the others, as their pose completion: the start from the observed joints with the others at rest,
    the geodesic loss of the visible joints, the visible joints of the result replaced by the observed ones."""
    device = prior.model.device
    observed = observed.to(device, prior.noise(1).dtype)
    eye = torch.eye(3, device=device, dtype=observed.dtype).expand_as(observed)
    y0 = torch.where(visible.to(device)[None, :, None, None], observed, eye)
    start = prior.start(y0)
    x1 = dflow(prior, lambda x: geodesic_loss(x[:, visible], observed[:, visible]), start, steps, lr, running_weight)
    return torch.where(visible.to(device)[None, :, None, None], observed, x1)
