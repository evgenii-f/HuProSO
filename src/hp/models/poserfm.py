"""PoseRFM: Riemannian flow matching on SO(3)^J (the prior of PoseD-Flow, Nadar, Foti, Birdal, CVPR 2026), as a PoseModel.

Flow matching on general geometries (Chen & Lipman 2024) with the geodesic conditional path. Time runs from noise to
data: t = 0 is the uniform distribution on SO(3)^J, t = 1 the data (the opposite of RSGM). For a source rotation x0
and a data rotation x1 of a joint, the path is the geodesic

    x_t = x0 exp(t v0),    v0 = log(x0^T x1)   (axis-angle, angle in [0, pi]),

and the conditional vector field is u_t(x_t | x1) = Log_{x_t}(x1) / (1 - t). In the frame of x_t (a tangent vector
at x is x @ hat(v), given by v) this is exactly v0 for every t: the field is constant along the geodesic, and the
apparent singularity 1 / (1 - t) at t = 1 is not there. The network v(x, t) regresses v0 with the squared norm of the
angle metric (as RSGM), summed over the joints:

    loss = E_{t, x0, x1} || v(x_t, t) - v0 ||^2.

Where x0^T x1 is a rotation by pi (the cut locus) the geodesic is not unique and v0 jumps; such pairs have measure
zero, and the loss stays finite there.

Sampling integrates dx/dt = v(x, t) from uniform x0 to t = 1 with geodesic midpoint steps; every step is a rotation,
so the samples are valid rotations. The paper uses the midpoint solver with 100 steps too.

log_prob integrates the same ODE backwards, from the pose at t = 1 - eps to t = 0, with the exact divergence (one
forward-mode pass per coordinate; the frame fields are divergence free, as in RSGM):

    log p_{1-eps}(x) = log p_0(x_0) - int_0^{1-eps} div v(x_t, t) dt,   log p_0 = 0 (uniform).

It is a density relative to the uniform distribution on SO(3)^J, the same space as so3_flow and RSGM, so the numbers
are comparable. Near t = 1 the learned field contracts towards the data and its divergence grows, so the density is
that of the data smoothed by the path at 1 - eps (a geodesic step of at most pi * eps per joint, 0.18 degrees for
eps = 1e-3), as RSGM gives the density at its eps.
"""
from typing import Optional, Sequence

import roma
import torch
from torch import nn
from torch.func import jacfwd, vmap

from hp.models.base import PoseModel
from hp.models.registry import register_model
from hp.models.rsgm import ACTIVATIONS, exp_frame, move


def random_rotations(shape, generator: Optional[torch.Generator] = None, device=None, dtype=torch.float32) -> torch.Tensor:
    """Uniform (Haar) rotations [*shape, 3, 3]: normalized Gaussian quaternions."""
    q = torch.randn(*shape, 4, generator=generator, device=device, dtype=dtype)
    return roma.unitquat_to_rotmat(q / q.norm(dim=-1, keepdim=True))


def log_frame(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The tangent vector at x of the geodesic to y, as coordinates [..., 3] in the frame of x: log(x^T y)."""
    return roma.rotmat_to_rotvec(x.transpose(-1, -2) @ y)


@register_model("poserfm")
class PoseRFM(PoseModel):
    has_log_prob = True
    density_space = "SO(3)^J"

    def __init__(
        self,
        num_joints: int,
        hidden: Sequence[int] = (512, 512, 512, 512),
        activation: str = "silu",
        eps: float = 1e-3,  # the likelihood is that of the path at t = 1 - eps
        sample_steps: int = 100,
        likelihood_steps: int = 100,
        representation: str = "matrix",
        cond_dim: int = 0,
    ):
        if representation != "matrix":
            raise ValueError("PoseRFM works on rotation matrices directly; use representation='matrix'.")
        if not 0 <= eps < 1:
            raise ValueError(f"eps must be in [0, 1), got {eps}.")
        super().__init__(num_joints, "matrix", cond_dim)
        self.eps, self.sample_steps, self.likelihood_steps = eps, sample_steps, likelihood_steps
        layers, width = [], 9 * num_joints + 1
        for h in hidden:
            layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
            width = h
        layers.append(nn.Linear(width, 3 * num_joints))
        self.net = nn.Sequential(*layers)

    @property
    def dtype(self) -> torch.dtype:
        return self.net[0].weight.dtype

    # ---- the vector field and the conditional path ----

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The learned field at x [B, J, 3, 3] and times t [B], as coordinates [B, J, 3] in the frame of x."""
        features = torch.cat([x.reshape(x.shape[0], -1), t[:, None].to(x.dtype)], dim=1)
        return self.net(features).reshape(x.shape[0], self.num_joints, 3)

    @staticmethod
    def conditional_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        """(x_t, u_t): the point at time t [B] on the geodesic from x0 to x1 [B, J, 3, 3], and the conditional
        field there in the frame of x_t, which is log(x0^T x1) for every t."""
        v0 = log_frame(x0, x1)
        return exp_frame(x0, t[:, None, None] * v0), v0

    def loss(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Riemannian conditional flow matching: E || v(x_t, t) - log(x0^T x1) ||^2, summed over the joints.

        In eval mode the times and the source rotations come from a fixed seed, so that a validation loss is the
        same number every time (and two copies of a model give the same loss)."""
        self.check_cond(cond, len(pose))
        device, dtype = self.device, self.dtype
        generator = None if self.training else torch.Generator(device=device).manual_seed(0)
        x1 = move(pose.detach().reshape(-1, self.num_joints, 3, 3), device, dtype)
        t = torch.rand(x1.shape[0], generator=generator, device=device, dtype=dtype)
        x0 = random_rotations(x1.shape[:2], generator, device, dtype)
        xt, target = self.conditional_path(x0, x1, t)
        return ((self.velocity(xt, t) - target) ** 2).sum(dim=(1, 2)).mean()

    # ---- sampling: the ODE from the uniform distribution to the data ----

    def _step(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One geodesic midpoint step of dx/dt = v from time t by h (h < 0: backwards)."""
        n = x.shape[0]
        x_mid = exp_frame(x, 0.5 * h * self.velocity(x, t.expand(n)))
        return exp_frame(x, h * self.velocity(x_mid, (t + 0.5 * h).expand(n)))

    @torch.no_grad()
    def sample(self, n: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_cond(cond, n)
        x = random_rotations((n, self.num_joints), device=self.device, dtype=self.dtype)
        times = torch.linspace(0.0, 1.0, self.sample_steps + 1, device=self.device, dtype=self.dtype)
        for t, t_next in zip(times[:-1], times[1:]):
            x = self._step(x, t, t_next - t)
        return x

    # ---- the likelihood: the ODE backwards with the exact divergence ----

    def velocity_and_divergence(self, x: torch.Tensor, t: torch.Tensor):
        """The field and its Riemannian divergence [B]: sum_i d/de v_i(x @ exp(e * e_i)) at e = 0 (the frame fields
        x @ hat(e_i) are Killing fields, so this trace of a Jacobian is the divergence)."""

        def one(delta, xi, ti):  # delta [J, 3], xi [J, 3, 3], ti []
            v = self.velocity(exp_frame(xi, delta)[None], ti[None])[0]
            return v, v

        zeros = torch.zeros(x.shape[0], self.num_joints, 3, device=x.device, dtype=x.dtype)
        jacobian, v = vmap(jacfwd(one, has_aux=True))(zeros, x, t)
        divergence = jacobian.reshape(x.shape[0], 3 * self.num_joints, 3 * self.num_joints).diagonal(dim1=1, dim2=2).sum(-1)
        return v, divergence

    @torch.no_grad()
    def integrate_back(self, pose: torch.Tensor):
        """(x_0, int div): the source point of `pose` [B, J, 3, 3] taken at t = 1 - eps, and the integral of the
        divergence along its trajectory (geodesic midpoint steps backwards, the divergence at the midpoints)."""
        x = move(pose.reshape(-1, self.num_joints, 3, 3), self.device, self.dtype)
        n = x.shape[0]
        times = torch.linspace(1.0 - self.eps, 0.0, self.likelihood_steps + 1, device=self.device, dtype=self.dtype)
        integral = torch.zeros(n, device=self.device, dtype=self.dtype)
        for t, t_next in zip(times[:-1], times[1:]):
            h = t_next - t  # < 0
            x_mid = exp_frame(x, 0.5 * h * self.velocity(x, t.expand(n)))
            v_mid, div_mid = self.velocity_and_divergence(x_mid, (t + 0.5 * h).expand(n))
            x = exp_frame(x, h * v_mid)
            integral = integral - h * div_mid  # int_0^{1-eps} div dt, accumulated from the end
        return x, integral

    def log_prob(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """log p(pose) relative to the uniform distribution on SO(3)^J (the density of the path at t = 1 - eps)."""
        self.check_cond(cond, len(pose))
        _, integral = self.integrate_back(pose)
        return -integral
