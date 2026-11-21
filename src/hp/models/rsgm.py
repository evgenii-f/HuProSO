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

sample() integrates the time reversal from t = 1 to eps over `sample_steps` steps. `sampler`: "euler_maruyama" (the
reverse SDE, as the JAX code), "euler" or "heun" (the probability flow ODE, first and second order; Heun calls the
network twice per step). On SO(3) the velocities are coordinates in the frame of x (left trivialization), and averaging
them as Heun does is second order without commutator terms (Runge-Kutta-Munthe-Kaas of order 2). `schedule`: the
times of the steps, "uniform" in t, or "edm" (Karras et al. 2022): uniform in sigma^(1/rho) for the noise scale
sigma = sqrt(s(t)), which puts the steps where the noise is small and the score changes fast.

log_prob is the probability flow ODE dx/dt = -beta(t)/2 * score(x, t) integrated from eps to 1, plus the integral
of the exact divergence (torch.func, one forward-mode pass per coordinate). It is a density relative to the
uniform distribution on SO(3)^J, the same space as so3_flow, so the numbers are comparable.

Conditioning (`condition` = "3D" or "2D", see hp.models.conditioning): the model learns p(pose | c) for keypoints c.
The noising process does not touch c, so the target of the loss is the same and only the network sees c: the features
of the condition MLP are concatenated to the input of the score network. Sampling and the likelihood are the same with
score(x, t, c); log_prob is log p(pose | c). cond_mask = "random" hides joints at random in the loss, "none": all
joints are observed. Without `condition`, cond_dim > 0 makes a model with a generic condition vector [B, cond_dim].

Per-joint noise (`joint_scale` = c [J]): joint j diffuses with the Brownian time s_j(t) = c_j s(t), i.e. with
beta_j = c_j beta, so x_t | x_0 is still IGSO(3) per joint and the target of the loss stays exact. With c_j proportional
to the spread of the joint over the data (fit_joint_scale), every joint has the same signal-to-noise ratio at a given
t, as in DPoser, whose noise is added to z-scored coordinates; the isotropic model (c = 1) drowns the joints of small
spread (the spine, the shoulders) at times where the wide ones (the knees, the elbows) are still clear. The std trick,
the weighting std_j(t)^2 of the loss, the sampler and the probability flow use s_j and beta_j joint by joint; the
network still gets the one time t. c is normalized to min c_j = 1, so that at t = 1 every joint is at least as close
to uniform as in the isotropic model. s_j stops at S_CAP (beta_j = 0 from there on): IGSO(3) of time 20 is uniform to
about exp(-20), and the numerics of the heat kernel fail far above it (c_j s(1) reaches about 160 for the knees).

Per-joint, per-axis noise (`joint_cov`, aniso_x; needs joint_scale): within a joint the spread is far from isotropic
(the knee bends about one axis: 89% of the variance of log(M_j^T x_j) on its first principal axis), and DPoser's
z-scoring of each coordinate gives its noise that shape. With joint_cov (the covariance of log(M_j^T x_j), J x 3 x 3)
the noise of joint j at Brownian time s_j is x0 exp(eps), eps ~ N(0, s_j K_j(s_j)) in the frame of x0: K_j has the
principal axes V_j of the covariance and the eigenvalues k = 1 + w (lambda - 1), lambda its eigenvalues over their mean
(so tr K = 3 and c_j keeps its meaning), w = 1 for s_j <= S_ANISO and 0 for s_j >= S_ISO, log-linear in between.
Below S_ISO this is the wrapped normal (the push-forward of the Gaussian by exp); its score, the target of the loss, is
taken by autograd from its density, a sum over the preimages of exp (wrapped_normal_score). Above S_ISO the noise is
the exact IGSO(3) of time s_j as without joint_cov (the wrapped normal of a large variance is singular at the
identity). The network outputs the scaled score in whitened coordinates: score = K^(-1/2) out / std_j. These kernels
are not the transitions of one SDE, so the samplers (the reverse SDE and the ODE with beta_j K_j) are approximate and
there is no exact likelihood (has_log_prob is False); the loss, the one-step denoising x_t exp(s_j K_j score) and
DPoser's optimization (score distillation) are what it is for.

Inputs of the score network. The rotations of a pose are relative to the parent joints, so the orientation of a
hand is a product along the chain root-spine-shoulder-arm-elbow-hand, which a plain MLP has to learn to multiply.
inputs = "local+global" also gives it these products (the forward kinematics along the SMPL tree, no root rotation),
19 x 9 numbers more; time_features adds sin and cos of t at several frequencies. Both are functions of x and t only,
so the sampler, the loss and the exact divergence of log_prob are unchanged.
inputs = "local+z" gives it, next to the matrices, the z-scored tangent coordinates of every joint around its mean, as
DPoser's input: z_j = log(M_j^T x_j) / sigma_j, with M_j the Karcher mean of the joint over the training poses and
sigma_j the std of each of the 3 coordinates of log(M_j^T x_j) (z_statistics; the arguments z_mean, the rotation
vectors of M_j, and z_std, kept as buffers). Joints of small spread (the wrists, the spine) move their 9 matrix numbers
little and their z by about 1. The matrices stay in the input: near angle pi from M_j the log jumps (large t), they
do not. A function of x only, like "local+global".
"""
import math
from typing import Optional, Sequence

import roma
import torch
from torch import nn
from torch.func import jacfwd, vmap

from hp.data.amass_cfg import AmassJoints4Learning
from hp.models import conditioning, igso3
from hp.models.score_nets import TimeResNet
from hp.models.base import PoseModel
from hp.models.conditioning import move  # noqa: F401  (also imported from here by other models)
from hp.models.registry import register_model


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


ACTIVATIONS = {"sin": Sin, "silu": nn.SiLU, "relu": nn.ReLU}
S_CAP = 20.0  # the largest Brownian time of a joint with joint_scale (see the module doc)
S_ANISO, S_ISO = 0.05, 1.0  # joint_cov: the full shape of K_j below S_ANISO, IGSO(3) from S_ISO on (module doc)
IMAGES = 3  # the preimages of exp on each side in the density of the wrapped normal
SAMPLERS = {"euler_maruyama": 1, "euler": 1, "heun": 2}  # and the network calls of a step


def exp_frame(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Move the rotations x [..., 3, 3] along the tangent vectors with coordinates v [..., 3]: x @ exp(hat(v))."""
    return x @ roma.rotvec_to_rotmat(v)


def log_rotations(r: torch.Tensor) -> torch.Tensor:
    """Rotation vectors [..., 3] of rotations r [..., 3, 3] in plain out-of-place ops, so that torch.func can vmap and
    differentiate it (the exact divergence of log_prob; roma's log map writes in place). The angle is atan2(|w|, c) of
    w = vee(r - r^T) / 2 = sin(angle) axis and c = (trace - 1) / 2 = cos(angle): accurate near 0; near pi the axis of w
    is still right while |w| is far above the rounding error (about 1e-3 rad from pi in float32), and at pi it is lost."""
    w = torch.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1) / 2
    c = (r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    n = w.pow(2).sum(-1).clamp_min(1e-20).sqrt()
    return (torch.atan2(n, c) / n)[..., None] * w


# parents of the 22 body joints of SMPL (AmassJoints order; -1: none), as body_models/smpl_skeleton.npz
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
# for each joint of AmassJoints4Learning, the index of its parent among them; -1: the parent is the root (identity)
_LEARNED = [joint.value for joint in AmassJoints4Learning]
PARENTS19 = tuple(_LEARNED.index(SMPL_PARENTS[j]) if SMPL_PARENTS[j] in _LEARNED else -1 for j in _LEARNED)


def global_rotations(x: torch.Tensor) -> torch.Tensor:
    """Rotations of the joints relative to the root [..., 19, 3, 3] from the poses x [..., 19, 3, 3] (each joint
    relative to its parent, AmassJoints4Learning): the forward kinematics G_j = G_parent(j) @ x_j along the tree."""
    out = []
    for j, p in enumerate(PARENTS19):  # a parent comes before its children
        out.append(x[..., j, :, :] if p < 0 else out[p] @ x[..., j, :, :])
    return torch.stack(out, dim=-3)


@torch.no_grad()
def z_statistics(poses: torch.Tensor, iterations: int = 50, chunk: int = 1_000_000):
    """(z_mean, z_std) [3J] of inputs="local+z" from training poses [N, J, 3, 3], as lists for the model's arguments:
    the rotation vectors of the Karcher (geodesic L2) mean M_j of every joint, and the std (N - 1) of each coordinate of
    log(M_j^T x_j). The mean starts at the projection of the mean matrix onto SO(3) and is moved by the mean of the
    logs until the step is below 1e-10 rad; in float64."""
    def logs_sum(m, power):
        total = torch.zeros(m.shape[0], 3, dtype=torch.float64)
        for i in range(0, len(poses), chunk):
            v = roma.rotmat_to_rotvec(m.transpose(-1, -2) @ poses[i:i + chunk].double())
            total += (v ** power).sum(0)
        return total

    n = len(poses)
    total = torch.zeros(poses.shape[1], 3, 3, dtype=torch.float64)
    for i in range(0, n, chunk):
        total += poses[i:i + chunk].double().sum(0)
    m = roma.special_procrustes(total / n)
    for _ in range(iterations):
        step = logs_sum(m, 1) / n
        m = m @ roma.rotvec_to_rotmat(step)
        if step.norm(dim=-1).max() < 1e-10:
            break
    mean = logs_sum(m, 1) / n  # ~0 at the Karcher mean
    std = ((logs_sum(m, 2) - n * mean ** 2) / (n - 1)).sqrt()
    return roma.rotmat_to_rotvec(m).reshape(-1).tolist(), std.reshape(-1).tolist()


def joint_covariance(poses: torch.Tensor, z_mean, chunk: int = 500_000) -> list:
    """The covariance (N - 1) [J * 9] of log(M_j^T x_j) of every joint over training poses [N, J, 3, 3], M_j the rotation
    vectors z_mean [3J] of z_statistics (the Karcher means): the argument joint_cov of RSGM (aniso_x)."""
    m = roma.rotvec_to_rotmat(torch.as_tensor(z_mean, dtype=torch.float64).reshape(-1, 3))
    n, joints = len(poses), m.shape[0]
    first = torch.zeros(joints, 3, dtype=torch.float64)
    second = torch.zeros(joints, 3, 3, dtype=torch.float64)
    for i in range(0, n, chunk):
        v = roma.rotmat_to_rotvec(m.transpose(-1, -2) @ poses[i:i + chunk].double())
        first += v.sum(0)
        second += torch.einsum("nja,njb->jab", v, v)
    mean = first / n
    return ((second - n * mean[:, :, None] * mean[:, None, :]) / (n - 1)).reshape(-1).tolist()


def wrapped_normal_score(eps: torch.Tensor, s: torch.Tensor, axes: torch.Tensor, k: torch.Tensor,
                         images: int = IMAGES) -> torch.Tensor:
    """The score [..., 3] (coordinates in the frame of x_t) of the wrapped normal at x_t = x0 exp(eps): the density of
    exp(e), e ~ N(0, s V diag(k) V^T) (s [...], the axes V [..., 3, 3] as columns, k [..., 3]), with respect to the
    Haar measure, sum over the preimages e_m = u (theta + 2 pi m) of exp of N(e_m) / J(e_m), J(e) = 4 sin^2(theta / 2)
    / |e|^2, theta and u the angle and axis of x_t relative to x0. In float64; the gradient d/d delta of the log density
    at x_t exp(delta), by autograd."""
    with torch.enable_grad():
        delta = torch.zeros_like(eps, requires_grad=True)
        v = log_rotations(roma.rotvec_to_rotmat(eps) @ roma.rotvec_to_rotmat(delta))
        theta = v.norm(dim=-1).clamp_min(1e-9)
        u = v / theta[..., None]
        var = s[..., None] * k
        terms = []
        for m in range(-images, images + 1):
            phi = theta + 2 * math.pi * m
            y = (axes.transpose(-1, -2) @ (u * phi[..., None])[..., None])[..., 0]  # coordinates on the principal axes
            if m == 0:  # J -> 1 at theta -> 0: 2 log sinc
                log_j = 2 * torch.log(torch.sinc(theta / (2 * math.pi)))
            else:
                log_j = torch.log(4 * torch.sin(theta / 2) ** 2) - torch.log(phi ** 2)
            terms.append(-0.5 * (y ** 2 / var).sum(-1) - log_j)
        log_p = torch.logsumexp(torch.stack(terms), dim=0)
        (grad,) = torch.autograd.grad(log_p.sum(), delta)
    return grad


def fit_joint_scale(z_std, alpha: float = 1.0) -> list:
    """joint_scale c [J] from the per-coordinate std [3J] of z_statistics: c_j = (sigma_j^2 / min_k sigma_k^2)^alpha,
    sigma_j the RMS of the three stds of joint j (the RMS angle of the joint around its mean). alpha = 1: every joint at
    the same signal-to-noise ratio (as DPoser's z-scoring); 0: isotropic; in between or beyond: the family of the
    alpha study (scripts/train_rsgm_posedflow.py, ALPHA)."""
    sigma2 = torch.as_tensor(z_std, dtype=torch.float64).reshape(-1, 3).pow(2).mean(-1)
    return ((sigma2 / sigma2.min()) ** alpha).tolist()


def joint_cov_power(joint_cov, alpha: float = 1.0) -> list:
    """The covariances [J * 9] raised to the power alpha (the same axes, the eigenvalues to the power alpha): the shape
    of the per-axis noise (joint_cov) of the alpha study, with the eigenvalue ratios lambda^alpha."""
    cov = torch.as_tensor(joint_cov, dtype=torch.float64).reshape(-1, 3, 3)
    lam, vec = torch.linalg.eigh(0.5 * (cov + cov.transpose(-1, -2)))
    return (vec @ torch.diag_embed(lam ** alpha) @ vec.transpose(-1, -2)).reshape(-1).tolist()


@register_model("rsgm")
class RSGM(PoseModel):
    has_log_prob = True
    supports_cond = True
    density_space = "SO(3)^J"
    # True: draw the IGSO(3) noise of the loss and of inpainting on the CPU in float64 and copy it to the device, the
    # behaviour before it was drawn on the device; for scripts/benchmark_igso_noise.py (not saved with the model)
    noise_on_cpu = False

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
        sampler: str = "euler_maruyama",  # "euler_maruyama" (reverse SDE), "euler" or "heun" (probability flow ODE)
        schedule: str = "uniform",  # the times of the sampler steps: "uniform" in t, or "edm" (see the module doc)
        schedule_rho: float = 7.0,  # rho of the "edm" schedule
        likelihood_steps: int = 100,
        weighting: str = "std",  # of the loss over time: "std" (std(t)^2, as the JAX code) or "likelihood" (beta(t))
        # or "lognormal": the std loss at log s(t) ~ N(loss_log_s_mean, loss_log_s_std^2), truncated (loss_times)
        loss_log_s_mean: float = -3.0,
        loss_log_s_std: float = 2.0,
        condition: Optional[str] = None,  # None (unconditional), "3D" or "2D": keypoints of hp.data.conditions
        cond_hidden: int = 64,  # the hidden layer of the condition MLP
        cond_features: int = 64,  # its output, concatenated to the input of the score network
        cond_mask: str = "none",  # joints hidden in the loss: "none" or "random" (see the module doc)
        representation: str = "matrix",
        cond_dim: int = 0,  # set by `condition`; without one, a generic condition vector of this length
        inputs: str = "local",  # of the score network: "local" (the rotations), "local+global" or "local+z" (module doc)
        z_mean: Optional[Sequence[float]] = None,  # of inputs="local+z": the rotation vectors of the means M_j [3J],
        z_std: Optional[Sequence[float]] = None,  # and the std of log(M_j^T x_j) [3J] (z_statistics)
        time_features: int = 0,  # sin and cos of t at this many frequencies, next to t itself; 0: t alone
        net: str = "mlp",  # "mlp": Linear-activation layers of `hidden`, t at the input; "timeres": score_nets.TimeResNet
        net_width: int = 1024,  # of net="timeres": the width,
        net_blocks: int = 2,  # its residual blocks (of two layers),
        net_embed: int = 256,  # the size of its embedding of the noise level,
        dropout: float = 0.0,  # and its dropout;
        net_level: str = "log_s",  # its noise level: "log_s" (log of the Brownian time, / 4) or "t"
        net_time_in: str = "layers",  # the ablation switches of score_nets.TimeResNet: time_in,
        net_time_embed: str = "sinusoidal",  # time_embed,
        net_residual: bool = True,  # residual,
        net_norm: bool = True,  # norm,
        net_zero_init: bool = True,  # zero_init
        joint_scale: Optional[Sequence[float]] = None,  # c [J]: the Brownian time c_j s(t) per joint (module doc)
        joint_cov: Optional[Sequence[float]] = None,  # [J * 9]: the noise of each joint shaped by it (aniso_x, module doc)
    ):
        if representation != "matrix":
            raise ValueError("RSGM works on rotation matrices directly; use representation='matrix'.")
        if net not in ("mlp", "timeres"):
            raise ValueError(f"Unknown net '{net}'. Use 'mlp' or 'timeres'.")
        if net_level not in ("log_s", "t"):
            raise ValueError(f"Unknown net_level '{net_level}'. Use 'log_s' or 't'.")
        if net == "timeres" and time_features:
            raise ValueError("net='timeres' embeds the noise level itself; use time_features=0.")
        if inputs not in ("local", "local+global", "local+z"):
            raise ValueError(f"Unknown inputs '{inputs}'. Use 'local', 'local+global' or 'local+z'.")
        if (inputs == "local+z") != (z_mean is not None and z_std is not None):
            raise ValueError("inputs='local+z' needs z_mean and z_std (z_statistics of the training poses), and only it.")
        if inputs == "local+global" and num_joints != len(PARENTS19):
            raise ValueError(f"inputs='local+global' needs the {len(PARENTS19)} joints of AmassJoints4Learning, got {num_joints}.")
        cond_dim = conditioning.resolve(condition, cond_dim, cond_mask)
        super().__init__(num_joints, "matrix", cond_dim)
        self.condition, self.cond_mask = condition, cond_mask
        self.beta_0, self.beta_f, self.eps, self.loss_t_min = beta_0, beta_f, eps, loss_t_min
        self.sample_steps, self.likelihood_steps = sample_steps, likelihood_steps
        if weighting not in ("std", "likelihood", "lognormal"):
            raise ValueError(f"Unknown weighting '{weighting}'. Use 'std', 'likelihood' or 'lognormal'.")
        if loss_log_s_std <= 0:
            raise ValueError(f"loss_log_s_std must be > 0, got {loss_log_s_std}.")
        self.weighting = weighting
        self.loss_log_s_mean, self.loss_log_s_std = float(loss_log_s_mean), float(loss_log_s_std)
        self.sampler, self.schedule, self.schedule_rho = sampler, schedule, schedule_rho
        self.check_sampler()
        self.inputs, self.time_features = inputs, time_features
        self.cond_net = conditioning.make_net(cond_dim, condition, cond_hidden, cond_features)
        if inputs == "local+z":  # buffers: saved with the weights, never trained
            z_mean = torch.as_tensor(z_mean, dtype=torch.float32).reshape(-1)
            z_std = torch.as_tensor(z_std, dtype=torch.float32).reshape(-1)
            if z_mean.shape != (3 * num_joints,) or z_std.shape != (3 * num_joints,):
                raise ValueError(f"z_mean and z_std must have {3 * num_joints} entries (3 per joint).")
            if not (z_std > 0).all():
                raise ValueError("z_std must be positive.")
            self.register_buffer("z_rot", roma.rotvec_to_rotmat(z_mean.reshape(num_joints, 3)))
            self.register_buffer("z_std", z_std.reshape(num_joints, 3))
        rotations = 9 * num_joints * (2 if inputs == "local+global" else 1) + (3 * num_joints if inputs == "local+z" else 0)
        self.net_kind, self.net_level = net, net_level
        if net == "timeres":
            self.net = TimeResNet(rotations + (cond_features if cond_dim else 0), 3 * num_joints, net_width, net_blocks,
                                  net_embed, dropout, net_time_in, net_time_embed, net_residual, net_norm,
                                  net_zero_init)
        else:
            layers, width = [], rotations + 1 + 2 * time_features + (cond_features if cond_dim else 0)
            for h in hidden:
                layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
                width = h
            layers.append(nn.Linear(width, 3 * num_joints))
            self.net = nn.Sequential(*layers)
        c = None  # a buffer when given (saved with the weights, never trained), else no buffer, as before
        if joint_scale is not None:
            c = torch.as_tensor(joint_scale, dtype=torch.float64).reshape(-1)
            if c.shape != (num_joints,) or not (c > 0).all():
                raise ValueError(f"joint_scale must have {num_joints} positive entries, got {c.tolist()}.")
            c = c.float()
        self.register_buffer("joint_scale", c)
        axes = shape = None  # joint_cov: buffers of the principal axes [J, 3, 3] and the eigenvalues over their mean [J, 3]
        if joint_cov is not None:
            if joint_scale is None:
                raise ValueError("joint_cov needs joint_scale (its eigenvalues give the shape, c_j the size).")
            cov = torch.as_tensor(joint_cov, dtype=torch.float64).reshape(-1, 3, 3)
            if cov.shape[0] != num_joints:
                raise ValueError(f"joint_cov must have {9 * num_joints} entries (3 x 3 per joint), got {cov.numel()}.")
            lam, vec = torch.linalg.eigh(0.5 * (cov + cov.transpose(-1, -2)))
            if not (lam > 0).all():
                raise ValueError("joint_cov must be positive definite.")
            axes, shape = vec.float(), (lam / lam.mean(-1, keepdim=True)).float()
            self.has_log_prob = False  # this instance: the kernels are not the transitions of one SDE (module doc)
        self.register_buffer("joint_axes", axes)
        self.register_buffer("joint_shape", shape)
        c_lo, c_hi = (1.0, 1.0) if joint_scale is None else (float(c.min()), float(c.max()))
        s_lo, s_hi = c_lo * self.s(torch.tensor(min(eps, loss_t_min))), c_hi * self.s(torch.tensor(1.0))
        s_hi = s_hi if joint_scale is None else min(s_hi, torch.tensor(S_CAP))
        self.angles = igso3.AngleSampler(float(s_lo) * 0.999, float(s_hi) * 1.001)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the score network's parameters."""
        return next(self.net.parameters()).dtype

    # ---- the noising process ----

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_0 + (self.beta_f - self.beta_0) * t

    def s(self, t: torch.Tensor) -> torch.Tensor:
        """Brownian time int_0^t beta (rescale_t of the JAX code)."""
        return self.beta_0 * t + 0.5 * (self.beta_f - self.beta_0) * t**2

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """The scale of the std trick: sqrt(1 - exp(-s)), as marginal_prob of the JAX code."""
        return torch.sqrt(-torch.expm1(-self.s(t)))

    def _c(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        return None if self.joint_scale is None else self.joint_scale.to(like.device).to(like.dtype)  # mps: no float64

    def joint_s(self, t: torch.Tensor) -> torch.Tensor:
        """The Brownian time of every joint [B, J] at the times t [B]: min(c_j s(t), S_CAP) (s(t) for the isotropic
        model)."""
        s = self.s(t)[:, None]
        c = self._c(s)
        return s.expand(-1, self.num_joints) if c is None else (s * c).clamp_max(S_CAP)

    def joint_std(self, t: torch.Tensor) -> torch.Tensor:
        """The scale of the std trick per joint [B, J]: sqrt(1 - exp(-s_j(t)))."""
        return torch.sqrt(-torch.expm1(-self.joint_s(t)))

    def joint_beta(self, t: torch.Tensor) -> torch.Tensor:
        """beta_j(t) = d s_j / dt [B, J]: c_j beta(t), and 0 where s_j has reached S_CAP."""
        b = self.beta(t)[:, None]
        c = self._c(b)
        if c is None:
            return b.expand(-1, self.num_joints)
        return torch.where(self.s(t)[:, None] * c < S_CAP, b * c, torch.zeros_like(b * c))

    def joint_k(self, s: torch.Tensor) -> torch.Tensor:
        """The eigenvalues k [B, J, 3] of the shape K_j(s_j) of the noise at the Brownian times s [B, J] (joint_cov)."""
        w = ((math.log(S_ISO) - torch.log(s)) / math.log(S_ISO / S_ANISO)).clamp(0, 1)
        return 1 + w[..., None] * (self.joint_shape.to(s.device).to(s.dtype) - 1)

    def shape_apply(self, v: torch.Tensor, s: torch.Tensor, power: float = 1.0) -> torch.Tensor:
        """K_j(s_j)^power v for tangent coordinates v [B, J, 3] and Brownian times s [B, J]; v itself without
        joint_cov."""
        if self.joint_axes is None:
            return v
        axes = self.joint_axes.to(v.device).to(v.dtype)
        y = torch.einsum("jca,bjc->bja", axes, v)
        return torch.einsum("jca,bja->bjc", axes, self.joint_k(s.to(v.dtype)) ** power * y)

    def tangent_noise(self, s: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        """Noise eps [B, J, 3] (float64, on the device of s) of the Brownian times s [B, J]: x_t = x0 exp(eps), IGSO(3)
        without joint_cov, the wrapped normal of covariance s K_j(s) below S_ISO with it (module doc)."""
        # with joint_cov IGSO(3) is used only from S_ISO on: drawn there (also for times below the range of `angles`)
        omega = self.angles.sample(s if self.joint_axes is None else s.clamp_min(S_ISO), generator)
        iso = omega[..., None] * igso3.random_axes(s.shape, generator, device=s.device)
        if self.joint_axes is None:
            return iso
        g = torch.randn(*s.shape, 3, generator=generator, device=s.device, dtype=torch.float64)
        gauss = self.shape_apply(s.sqrt()[..., None] * g, s, 0.5)
        return torch.where((s < S_ISO)[..., None], gauss, iso)

    def noise_score(self, eps: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """The score [B, J, 3] of the noise kernel at x_t = x0 exp(eps) in the frame of x_t, for eps of tangent_noise
        (float64)."""
        omega = eps.norm(dim=-1)
        iso_s = s if self.joint_axes is None else s.clamp_min(S_ISO)  # with joint_cov only used from S_ISO on
        iso = igso3.d_log_density(omega, iso_s)[..., None] * eps / omega.clamp_min(1e-12)[..., None]
        if self.joint_axes is None:
            return iso
        axes = self.joint_axes.to(eps.device).double().expand(*s.shape, 3, 3)
        small = s < S_ISO
        wrapped = wrapped_normal_score(eps, s.clamp_max(S_ISO), axes, self.joint_k(s))
        return torch.where(small[..., None], wrapped, iso)

    def t_of_s(self, s: torch.Tensor) -> torch.Tensor:
        """The time t of the Brownian time s: s = beta_0 t + (beta_f - beta_0) t^2 / 2, solved for t."""
        a = self.beta_f - self.beta_0
        return (-self.beta_0 + torch.sqrt(self.beta_0**2 + 2 * a * s)) / a

    def loss_times(self, u: torch.Tensor) -> torch.Tensor:
        """Times of the training loss from uniform numbers u in [0, 1] (float64).

        "std": t uniform on [loss_t_min, 1]. "likelihood": the loss weighted by beta(t) instead of std(t)^2, which
        makes it a bound on the negative log-likelihood (Song et al. 2021, "Maximum likelihood training of score-based
        diffusion models"). It is done by importance sampling: t is drawn with the density q(t) proportional to
        beta(t) / std(t)^2, so the loss itself stays the same (up to a constant factor, to which Adam is blind).
        In the Brownian time s the density is 1 / (1 - exp(-s)), with the CDF log(exp(s) - 1), inverted in closed form.
        "lognormal": log s(t) ~ N(loss_log_s_mean, loss_log_s_std^2) truncated to [s(loss_t_min), s(1)], by the inverse
        CDF (EDM's log-normal noise levels, Karras et al. 2022); the loss keeps the std(t)^2 weighting, so only where
        it is evaluated changes. The default N(-3, 2^2) puts 21% at s < 0.01 and 7% at s > 1 (std: 6% and 42%)."""
        if self.weighting == "std":
            return self.loss_t_min + (1 - self.loss_t_min) * u
        t_lo = torch.tensor(self.loss_t_min, dtype=torch.float64)
        if self.weighting == "lognormal":
            m, sd = self.loss_log_s_mean, self.loss_log_s_std
            ends = torch.log(torch.stack([self.s(t_lo), self.s(torch.ones_like(t_lo))]))
            p_lo, p_hi = torch.special.ndtr((ends - m) / sd)
            p = (p_lo + u * (p_hi - p_lo)).clamp(1e-300, 1 - 1e-16)
            t = self.t_of_s(torch.exp(m + sd * torch.special.ndtri(p)))
            return t.clamp(self.loss_t_min, 1.0)
        lo, hi = torch.log(torch.expm1(self.s(t_lo))), torch.log(torch.expm1(self.s(torch.ones_like(t_lo))))
        return self.t_of_s(torch.log1p(torch.exp(lo + u * (hi - lo))))

    # ---- the score ----

    def cond_features(self, cond: Optional[torch.Tensor], batch: int) -> torch.Tensor:
        """Features [B, cond_features] of the condition [B, cond_dim] (NaN: joint not observed), on the model's device
        in its dtype; [B, 0] for an unconditional model."""
        return conditioning.features(self.cond_net, self.condition, cond, batch, self.device, self.dtype)

    def score(self, x: torch.Tensor, t: torch.Tensor, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score of p_t at x [B, J, 3, 3] and times t [B], as coordinates [B, J, 3] in the frame of x; f [B, F] are
        the features of the condition (cond_features; None or [B, 0] for an unconditional model)."""
        b, t_col = x.shape[0], t[:, None].to(x.dtype)
        inputs = [x.reshape(b, -1)]
        if self.inputs == "local+global":
            inputs.append(global_rotations(x).reshape(b, -1))
        elif self.inputs == "local+z":
            z = log_rotations(self.z_rot.to(x.dtype).transpose(-1, -2) @ x) / self.z_std.to(x.dtype)
            inputs.append(z.reshape(b, -1))
        if self.net_kind == "timeres":  # the time goes to every layer, as the noise level log(s) / 4
            features = torch.cat(inputs + ([] if f is None else [f]), dim=1)
            if self.net_level == "log_s":
                level = torch.log(self.s(t.to(x.dtype))) / 4  # s(t) >= beta_0 eps ~ 2e-7: fine in float32
            else:
                level = t.to(x.dtype)
            return self._unscale(self.net(features, level), t, x.dtype)
        inputs.append(t_col)
        if self.time_features:  # frequencies pi, 2 pi, 4 pi, ...: periods from 2 down to 2^(2 - K) in t
            angles = t_col * (math.pi * 2.0 ** torch.arange(self.time_features, device=x.device, dtype=x.dtype))
            inputs += [torch.sin(angles), torch.cos(angles)]
        features = torch.cat(inputs + ([] if f is None else [f]), dim=1)
        return self._unscale(self.net(features), t, x.dtype)

    def _unscale(self, out: torch.Tensor, t: torch.Tensor, dtype) -> torch.Tensor:
        """The output [B, 3J] of the network divided by std (per joint with joint_scale), as [B, J, 3]; with joint_cov
        it is read in whitened coordinates: K_j^(-1/2) out / std_j."""
        if self.joint_scale is None:
            return (out / self.std(t)[:, None].to(dtype)).reshape(out.shape[0], self.num_joints, 3)
        out = out.reshape(out.shape[0], self.num_joints, 3)
        if self.joint_axes is not None:
            out = self.shape_apply(out, self.joint_s(t.to(dtype)), -0.5)
        return out / self.joint_std(t.to(dtype))[..., None]

    def loss(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Denoising score matching: E || std(t) * (score(x_t, t) - grad log p_t(x_t | x_0)) ||^2.

        In eval mode the times and the noise are drawn from a fixed seed, so that a validation loss is the same
        number every time (and two copies of a model give the same loss). A conditional model computes the keypoints
        of the poses if `cond` is None, and hides joints of it if cond_mask is "random"."""
        generator = None if self.training else torch.Generator().manual_seed(0)
        batch, joints = pose.shape[0], self.num_joints
        cond = conditioning.of_poses(pose, cond, self.condition)
        self.check_cond(cond, batch)
        if self.cond_mask == "random":
            cond = conditioning.hide_random(cond, generator)
        # the noise and its target in float64 (the score of IGSO(3) is ~1/s large near s = 0), on the model's device
        # where it has float64 (cuda), else on the CPU (mps); validation (fixed generator) on the CPU, as before
        on_cpu = generator is not None or self.device.type == "mps" or self.noise_on_cpu
        work = torch.device("cpu") if on_cpu else self.device
        t = self.loss_times(torch.rand(batch, dtype=torch.float64, device=work, generator=generator))
        s = self.joint_s(t)  # [B, J], float64
        std = self.joint_std(t)
        x0 = move(pose.detach().reshape(batch, joints, 3, 3), work, torch.float64)
        if self.joint_axes is None:
            omega = self.angles.sample(s, generator)
            axis = igso3.random_axes((batch, joints), generator, device=work)
            xt = exp_frame(x0, omega[..., None] * axis)
            target = std[..., None] * igso3.d_log_density(omega, s)[..., None] * axis
        else:  # whitened: std K^(1/2) (score - target), the weighting of the covariance (module doc)
            eps = self.tangent_noise(s, generator)
            xt = exp_frame(x0, eps)
            target = std[..., None] * self.shape_apply(self.noise_score(eps, s), s, 0.5)
        dtype = self.dtype
        xt, target, t, std, s = (move(v, self.device, dtype) for v in (xt, target, t, std, s))
        scaled = self.score(xt, t, self.cond_features(cond, batch)) * std[..., None]
        if self.joint_axes is not None:
            scaled = self.shape_apply(scaled, s, 0.5)
        return ((scaled - target) ** 2).sum(dim=(1, 2)).mean()

    # ---- sampling: the time reversal, by geodesic Euler-Maruyama or the probability flow ODE ----

    def check_sampler(self) -> None:
        if self.sampler not in SAMPLERS:
            raise ValueError(f"Unknown sampler '{self.sampler}'. Use one of {', '.join(SAMPLERS)}.")
        if self.schedule not in ("uniform", "edm"):
            raise ValueError(f"Unknown schedule '{self.schedule}'. Use 'uniform' or 'edm'.")

    @property
    def sample_nfe(self) -> int:
        """Calls of the score network in one sample()."""
        return self.sample_steps * SAMPLERS[self.sampler]

    def sample_times(self, steps: int, schedule: Optional[str] = None) -> torch.Tensor:
        """steps + 1 times from 1 down to eps (float64): the grid of `schedule` (default: the model's)."""
        schedule = schedule or self.schedule
        if schedule == "uniform":
            return torch.linspace(1.0, self.eps, steps + 1, dtype=torch.float64)
        ends = self.s(torch.tensor([1.0, self.eps], dtype=torch.float64)).sqrt() ** (1 / self.schedule_rho)
        sigma = (ends[0] + torch.linspace(0, 1, steps + 1, dtype=torch.float64) * (ends[1] - ends[0])) ** self.schedule_rho
        times = self.t_of_s(sigma**2)
        times[0], times[-1] = 1.0, self.eps  # exactly, not up to rounding
        return times

    @torch.no_grad()
    def sample(self, n: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_cond(cond, n)
        self.check_sampler()
        dtype = self.dtype
        f = self.cond_features(cond, n)
        x = roma.random_rotmat(n * self.num_joints, device=self.device, dtype=dtype).reshape(n, self.num_joints, 3, 3)
        times = self.sample_times(self.sample_steps).to(self.device, dtype)
        for t, t_next in zip(times[:-1], times[1:]):
            h = t_next - t  # negative: the ODE runs backwards in time
            if self.sampler == "euler_maruyama":
                dt, beta = -h, self.joint_beta(t.expand(n))[..., None]
                s_j = self.joint_s(t.expand(n))  # joint_cov: K_j(s_j) shapes the drift and the noise (approximate)
                noise = self.shape_apply(torch.randn(n, self.num_joints, 3, device=self.device, dtype=dtype), s_j, 0.5)
                step = beta * self.shape_apply(self.score(x, t.expand(n), f), s_j) * dt + torch.sqrt(beta * dt) * noise
                x = exp_frame(x, step)
                continue
            v = self.velocity(x, t.expand(n), f)
            if self.sampler == "heun":
                v = 0.5 * (v + self.velocity(exp_frame(x, h * v), t_next.expand(n), f))
            x = exp_frame(x, h * v)
        return x

    @torch.no_grad()
    def sample_ode(self, n: int, cond: Optional[torch.Tensor] = None, start: Optional[torch.Tensor] = None,
                   steps: Optional[int] = None) -> torch.Tensor:
        """n poses by the probability flow ODE from t = 1 to eps (geodesic midpoint steps): the same distribution as
        sample(), but a deterministic, continuous map of the start [n, J, 3, 3] (uniform rotations if None) and of
        the condition. So the same start with slowly changing conditions gives slowly changing poses."""
        self.check_cond(cond, n)
        dtype = self.dtype
        f = self.cond_features(cond, n)
        if start is None:
            start = roma.random_rotmat(n * self.num_joints, device=self.device, dtype=dtype)
        x = move(start.reshape(n, self.num_joints, 3, 3), self.device, dtype)
        times = torch.linspace(1.0, self.eps, (steps or self.sample_steps) + 1, device=self.device, dtype=dtype)
        for t, t_next in zip(times[:-1], times[1:]):
            h = t_next - t  # < 0: backwards in time
            x_mid = exp_frame(x, 0.5 * h * self.velocity(x, t.expand(n), f))
            x = exp_frame(x, h * self.velocity(x_mid, (t + 0.5 * h).expand(n), f))
        return x

    # ---- the likelihood: probability flow ODE with the exact divergence ----

    def velocity(self, x: torch.Tensor, t: torch.Tensor, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The probability flow -beta_j(t)/2 * score (K_j score with joint_cov, approximate), coordinates [B, J, 3] in
        the frame of x."""
        return -0.5 * self.joint_beta(t)[..., None] * self.shape_apply(self.score(x, t, f), self.joint_s(t))

    def velocity_and_divergence(self, x: torch.Tensor, t: torch.Tensor, f: Optional[torch.Tensor] = None):
        """The velocity and its Riemannian divergence [B]. The frame fields x @ hat(e_i) are divergence free
        (Killing fields), so the divergence is sum_i d/de v_i(x @ exp(e * e_i)) at e = 0, the trace of a Jacobian."""

        if f is None:
            f = torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)

        def one(delta, xi, ti, fi):  # delta [J, 3], xi [J, 3, 3], ti [], fi [F]
            v = self.velocity(exp_frame(xi, delta)[None], ti[None], fi[None] if fi.numel() else None)[0]
            return v, v

        zeros = torch.zeros(x.shape[0], self.num_joints, 3, device=x.device, dtype=x.dtype)
        jacobian, v = vmap(jacfwd(one, has_aux=True))(zeros, x, t, f)
        divergence = jacobian.reshape(x.shape[0], 3 * self.num_joints, 3 * self.num_joints).diagonal(dim1=1, dim2=2).sum(-1)
        return v, divergence

    def log_prob(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """log p(pose) relative to the uniform distribution on SO(3)^J (log p(pose | cond) for a conditional model):
        midpoint steps of the ODE from eps to 1, log p_eps(x) = log p_1(x_1) + int div, with p_1 taken as uniform
        (log p_1 = 0)."""
        if self.joint_axes is not None:
            raise NotImplementedError("no exact likelihood with joint_cov: its kernels are not one SDE (module doc).")
        self.check_cond(cond, len(pose))
        dtype = self.dtype
        x = move(pose.reshape(-1, self.num_joints, 3, 3), self.device, dtype)
        n = x.shape[0]
        with torch.no_grad():
            f = self.cond_features(cond, n)
        times = torch.linspace(self.eps, 1.0, self.likelihood_steps + 1, device=self.device, dtype=dtype)
        log_p = torch.zeros(n, device=self.device, dtype=dtype)
        with torch.no_grad():
            for t, t_next in zip(times[:-1], times[1:]):
                h = t_next - t
                x_mid = exp_frame(x, 0.5 * h * self.velocity(x, t.expand(n), f))
                v_mid, div_mid = self.velocity_and_divergence(x_mid, (t + 0.5 * h).expand(n), f)
                x = exp_frame(x, h * v_mid)
                log_p = log_p + h * div_mid
        return log_p
