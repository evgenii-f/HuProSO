"""DPoser: a score-based diffusion prior in Euclidean space (Lu et al., ICCV 2025; github.com/moonbow721/DPoser-X), as a
PoseModel, to be trained from scratch on our data.

The model of DPoser-body (configs/body/subvp/timefc.py there), rebuilt without their environment:

    data       the axis-angle vectors of the joints [B, 3J], z-scored per coordinate with `mean` and `std` of the training
               data (their Posenormalizer, min_max=False); our axis-angle has angles in [0, pi] (roma.rotmat_to_rotvec)
    SDE        sub-VP (Song et al. 2021) with beta from beta_min = 0.1 to beta_max = 20 on t in [0, 1]:
               x_t = alpha(t) x_0 + std(t) z,  alpha = exp(-t^2 (beta_max - beta_min) / 4 - t beta_min / 2),
               std = 1 - alpha^2 (the sub-VP variance is std^2, not 1 - alpha^2)
    network    TimeFC: a dense layer, then `blocks` residual blocks of two dense layers, each with an additive projection of
               the time embedding, GroupNorm(32), swish and dropout; the time embedding is sinusoidal of labels = 999 t
               (embedding_type 'positional') followed by a shared dense + swish
    output     divided by sigmas[int(999 t)] (scale_by_sigma), sigmas the geometric series from sigma_max = 50 down to
               sigma_min = 0.01 of the VE-SDE (a quirk of score_sde_pytorch that DPoser keeps for its VP models); the
               score is -output / std(t), so the output, after this division, predicts the noise z
    loss       continuous denoising score matching, ||score std + z||^2 = ||z - output||^2 averaged over the coordinates
               (reduce_mean) and the batch, t ~ U(loss_eps, 1)
    sampling   the PC sampler with the Euler-Maruyama predictor and no corrector: sample_steps steps on
               linspace(1, sample_eps, sample_steps) with dt = -1 / sample_steps (as theirs, although the grid spacing is
               (1 - sample_eps) / (sample_steps - 1)), from N(0, I), returning the mean of the last step (noise_removal)

The model has no log_prob in the sense of the zoo, so has_log_prob is False: its probability flow ODE gives a density
on R^(3J) of the normalized vectors (log_prob_normalized), which log_prob_so3 carries over to SO(3)^J (relative to the
uniform distribution, as RSGM) by the z-scoring and the volume of the axis-angle chart, up to the mass outside it.
"""
import math
from typing import Optional, Sequence

import torch
from torch import nn
from torch.func import jacfwd, vmap

from hp.models.base import PoseModel
from hp.models.registry import register_model
from hp.models.rsgm import move


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_positions: int = 10_000) -> torch.Tensor:
    """The sinusoidal embedding [B, dim] of (real) timesteps [B], as get_timestep_embedding of DPoser (and DDPM)."""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=timesteps.device) * -(math.log(max_positions) / (half - 1)))
    angles = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return nn.functional.pad(emb, (0, 1)) if dim % 2 else emb


class TimeFC(nn.Module):
    """The network of DPoser-body (lib/algorithms/advanced/model.py there), its layers named as theirs."""

    def __init__(self, dim: int, hidden: int, embed: int, blocks: int, dropout: float):
        super().__init__()
        self.embed, self.blocks = embed, blocks
        self.act = nn.SiLU()  # 'swish'
        self.dropout = nn.Dropout(p=dropout)
        self.pre_dense = nn.Linear(dim, hidden)
        self.pre_dense_t = nn.Linear(embed, hidden)
        self.pre_gnorm = nn.GroupNorm(32, num_channels=hidden)
        self.shared_time_embed = nn.Sequential(nn.Linear(embed, embed), nn.SiLU())
        for i in range(1, blocks + 1):
            for k in (1, 2):
                setattr(self, f"b{i}_dense{k}", nn.Linear(hidden, hidden))
                setattr(self, f"b{i}_dense{k}_t", nn.Linear(embed, hidden))
                setattr(self, f"b{i}_gnorm{k}", nn.GroupNorm(32, num_channels=hidden))
        self.post_dense = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        temb = self.shared_time_embed(timestep_embedding(labels, self.embed).to(x.dtype))
        h = self.dropout(self.act(self.pre_gnorm(self.pre_dense(x) + self.pre_dense_t(temb))))
        for i in range(1, self.blocks + 1):
            h1 = h
            for k in (1, 2):
                h1 = getattr(self, f"b{i}_dense{k}")(h1) + getattr(self, f"b{i}_dense{k}_t")(temb)
                h1 = self.dropout(self.act(getattr(self, f"b{i}_gnorm{k}")(h1)))
            h = h + h1
        return self.post_dense(h)


@register_model("dposer")
class DPoser(PoseModel):
    has_log_prob = False

    def __init__(
        self,
        num_joints: int,
        hidden: int = 1024,  # HIDDEN_DIM
        embed: int = 512,  # EMBED_DIM
        blocks: int = 2,  # N_BLOCKS
        dropout: float = 0.1,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        num_scales: int = 1000,  # the length of the sigma table of scale_by_sigma
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        scale_by_sigma: bool = True,
        loss_eps: float = 1e-5,  # the smallest time of the loss
        sample_steps: int = 1000,  # their N (num_scales)
        sample_eps: float = 1e-3,  # the last time of the sampler
        mean: Optional[Sequence[float]] = None,  # [3J] of the training data; None: no normalization
        std: Optional[Sequence[float]] = None,
        representation: str = "axis_angle",
        cond_dim: int = 0,
    ):
        if representation != "axis_angle":
            raise ValueError("DPoser works on axis-angle vectors; use representation='axis_angle'.")
        if (mean is None) != (std is None):
            raise ValueError("Give both mean and std of the training data, or neither.")
        if hidden % 32:
            raise ValueError(f"hidden must be a multiple of 32 (GroupNorm with 32 groups), got {hidden}.")
        super().__init__(num_joints, representation, cond_dim)
        dim = 3 * num_joints
        self.beta_min, self.beta_max = beta_min, beta_max
        self.scale_by_sigma, self.loss_eps = scale_by_sigma, loss_eps
        self.sample_steps, self.sample_eps = sample_steps, sample_eps
        self.net = TimeFC(dim, hidden, embed, blocks, dropout)
        sigmas = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), num_scales, dtype=torch.float64))
        self.register_buffer("sigmas", sigmas.float())
        for name, value, default in (("mean", mean, 0.0), ("std", std, 1.0)):
            t = torch.full((dim,), default) if value is None else torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            if t.shape != (dim,):
                raise ValueError(f"{name} must have {dim} entries (3 per joint), got {t.numel()}.")
            self.register_buffer(f"data_{name}", t)
        if not (self.data_std > 0).all():
            raise ValueError("std must be positive.")

    @property
    def dtype(self) -> torch.dtype:
        return self.net.pre_dense.weight.dtype

    # ---- the data space: z-scored axis-angle ----

    def normalize(self, pose: torch.Tensor) -> torch.Tensor:
        """[B, J, 3, 3] -> [B, 3J] z-scored axis-angle."""
        return (self.encode(pose) - self.data_mean) / self.data_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3J] -> [B, J, 3, 3]."""
        return self.decode(x * self.data_std + self.data_mean)

    # ---- the sub-VP SDE ----

    def log_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def marginal(self, t: torch.Tensor):
        """(alpha(t), std(t)) [B]: x_t = alpha x_0 + std z."""
        log_alpha = self.log_alpha(t)
        return torch.exp(log_alpha), 1 - torch.exp(2 * log_alpha)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) of the forward SDE dx = -beta x / 2 dt + g dW."""
        discount = 1 - torch.exp(-2 * self.beta_min * t - (self.beta_max - self.beta_min) * t ** 2)
        return torch.sqrt(self.beta(t) * discount)

    # ---- the network and the score ----

    def noise_prediction(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The network output [B, 3J] at x [B, 3J] and times t [B] (after scale_by_sigma): it predicts z."""
        labels = t * 999
        out = self.net(x, labels)
        if self.scale_by_sigma:
            out = out / self.sigmas[labels.long()][:, None]
        return out

    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -self.noise_prediction(x, t) / self.marginal(t)[1][:, None]

    def loss(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Denoising score matching, ||score std + z||^2 averaged over the coordinates and the batch.

        In eval mode the times and the noise come from a fixed seed, so that a validation loss is the same number every
        time (and two copies of a model give the same loss)."""
        self.check_cond(cond, len(pose))
        device, dtype = self.device, self.dtype
        generator = None if self.training else torch.Generator(device=device).manual_seed(0)
        x0 = self.normalize(move(pose.detach().reshape(-1, self.num_joints, 3, 3), device, dtype))
        t = torch.rand(x0.shape[0], generator=generator, device=device, dtype=dtype) * (1 - self.loss_eps) + self.loss_eps
        z = torch.randn(x0.shape, generator=generator, device=device, dtype=dtype)
        alpha, std = self.marginal(t)
        xt = alpha[:, None] * x0 + std[:, None] * z
        return ((z - self.noise_prediction(xt, t)) ** 2).mean()

    # ---- sampling: the reverse SDE, Euler-Maruyama ----

    @torch.no_grad()
    def sample(self, n: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_cond(cond, n)
        device, dtype = self.device, self.dtype
        x = torch.randn(n, 3 * self.num_joints, device=device, dtype=dtype)
        x_mean = x
        dt = -1.0 / self.sample_steps
        for t in torch.linspace(1.0, self.sample_eps, self.sample_steps, device=device, dtype=dtype):
            vec_t = t.expand(n)
            g = self.diffusion(vec_t)[:, None]
            drift = -0.5 * self.beta(vec_t)[:, None] * x - g ** 2 * self.score(x, vec_t)
            x_mean = x + drift * dt
            x = x_mean + g * math.sqrt(-dt) * torch.randn_like(x)
        return self.denormalize(x_mean)

    # ---- the likelihood: probability flow ODE with the exact divergence ----

    def ode_velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The probability flow dx/dt = -beta x / 2 - g^2 score / 2 at x [B, 3J] and times t [B]."""
        return -0.5 * self.beta(t)[:, None] * x - 0.5 * self.diffusion(t)[:, None] ** 2 * self.score(x, t)

    def velocity_and_divergence(self, x: torch.Tensor, t: torch.Tensor):
        """The velocity [B, 3J] and its divergence [B], the trace of the Jacobian (one forward pass per coordinate)."""

        def one(xi, ti):
            v = self.ode_velocity(xi[None], ti[None])[0]
            return v, v

        jacobian, v = vmap(jacfwd(one, has_aux=True))(x, t)
        return v, jacobian.diagonal(dim1=1, dim2=2).sum(-1)

    @torch.no_grad()
    def log_prob_normalized(self, x: torch.Tensor, steps: int = 100, eps: Optional[float] = None) -> torch.Tensor:
        """log p [B] of z-scored axis-angle vectors x [B, 3J] (a density on R^(3J)): midpoint steps of the ODE from eps
        (default sample_eps) to 1, log p_eps(x) = log N(x_1; 0, I) + int div; at t = 1 the sub-VP marginal is
        N(alpha x_0, std^2) with alpha < 0.01 and std > 0.9999, taken as N(0, I)."""
        x = move(x, self.device, self.dtype)
        n = x.shape[0]
        times = torch.linspace(self.sample_eps if eps is None else eps, 1.0, steps + 1, device=self.device, dtype=self.dtype)
        integral = torch.zeros(n, device=self.device, dtype=self.dtype)
        for t, t_next in zip(times[:-1], times[1:]):
            h = t_next - t
            x_mid = x + 0.5 * h * self.ode_velocity(x, t.expand(n))
            v_mid, div_mid = self.velocity_and_divergence(x_mid, (t + 0.5 * h).expand(n))
            x = x + h * v_mid
            integral = integral + h * div_mid
        base = -0.5 * (x ** 2).sum(-1) - 0.5 * x.shape[1] * math.log(2 * math.pi)
        return base + integral

    @torch.no_grad()
    def log_prob_so3(self, pose: torch.Tensor, steps: int = 100, eps: Optional[float] = None) -> torch.Tensor:
        """log p [B] of poses [B, J, 3, 3] relative to the uniform distribution on SO(3)^J, as RSGM's log_prob: the
        density of the normalized vectors, the z-scoring (- sum log std) and per joint the volume of the axis-angle
        chart. The uniform distribution has the density (1 - cos a) / (4 pi^2 a^2) in axis-angle v (a = |v| <= pi),
        so p_SO(3)(R) = p_v(v) 4 pi^2 a^2 / (1 - cos a) = p_v(v) 2 pi^2 a^2 / sin^2(a / 2) per joint.

        The model's density on R^(3J) also has some mass outside the ball |v| <= pi of every joint, which this
        ignores: the result is a slight underestimate, the same for all poses up to that leakage. Not log_prob, so
        that the harness of the zoo keeps DPoser as a model without a density (has_log_prob is False)."""
        pose = move(pose.reshape(-1, self.num_joints, 3, 3), self.device, self.dtype)
        v = self.encode(pose)
        log_p = self.log_prob_normalized((v - self.data_mean) / self.data_std, steps, eps) - torch.log(self.data_std).sum()
        a = v.reshape(-1, self.num_joints, 3).norm(dim=-1)
        half = torch.sin(a / 2)
        # a^2 / sin^2(a / 2) -> 4 at a = 0
        ratio = torch.where(a > 1e-4, a ** 2 / half.clamp(min=1e-12) ** 2, torch.full_like(a, 4.0))
        return log_p + torch.log(2 * math.pi ** 2 * ratio).sum(-1)


def normalization(poses: torch.Tensor, chunk: int = 1_000_000):
    """(mean, std) [3J] of the axis-angle vectors of training poses [N, J, 3, 3], as lists for the model's arguments.
    std is that of torch.std (N - 1), as their Posenormalizer was fitted."""
    from hp.models.representation import get_representation

    axis_angle = get_representation("axis_angle")
    total, square, n = None, None, 0
    for i in range(0, len(poses), chunk):
        v = axis_angle.pose_to_vector(poses[i:i + chunk]).double()
        total = v.sum(0) if total is None else total + v.sum(0)
        square = (v * v).sum(0) if square is None else square + (v * v).sum(0)
        n += len(v)
    mean = total / n
    std = ((square - n * mean ** 2) / (n - 1)).sqrt()
    return mean.tolist(), std.tolist()
