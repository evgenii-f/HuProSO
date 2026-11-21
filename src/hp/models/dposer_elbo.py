"""DPoser trained on the continuous-time ELBO, with its noise d learned (phase 1 of docs/elbo_noise_plan.md).

A subclass of DPoser (hp/models/dposer.py), which it leaves unchanged: the network, the sub-VP SDE, the score, the
sampler and the likelihood are DPoser's, with the noise covariance std(t)^2 diag(d^2). What differs:

    noise      with learn_noise (the default) d is learned: d^2 = 3J softmax(noise_logit_scale * logits), so that the
               mean of d^2 is 1 (a common scale of d is not determined by the ELBO, only by its boundaries); logits = 0
               is d = 1, DPoser as published; without learn_noise d is DPoser's fixed noise_scale
    loss       -ELBO per coordinate (VDM, Kingma et al. 2021, Eq. 11): the prior KL(q(x_1 | x_0) || N(0, diag d^2)),
               the reconstruction -log p(x_0 | x_t0) with the Gaussian p(x | z) ~ q(z | x) (VDM Eq. 8 normalized over
               R), and the diffusion term of VDM Eq. 17 with the time drawn uniformly in gamma = -log SNR (a constant
               weight) and spread over the batch (VDM App. I.1); from elbo_t0 (default loss_eps) to 1

Its product is d (phase 2 trains DPoser itself with noise_scale = d). The completion paths that assume isotropic noise
(D-Flow, inpainting) refuse it, as they refuse a noise_scale.
"""
import math
from typing import Optional

import torch
from torch import nn

from hp.models.dposer import DPoser
from hp.models.registry import register_model
from hp.models.rsgm import move


@register_model("dposer_elbo")
class DPoserELBO(DPoser):

    def __init__(
        self,
        num_joints: int,
        learn_noise: bool = True,
        noise_logit_scale: float = 1.0,  # multiplies the logits: with Adam their step is this times the learning rate
        elbo_t0: Optional[float] = None,  # the smallest time of the ELBO; None: loss_eps
        **kwargs,
    ):
        if learn_noise and kwargs.get("noise_scale") is not None:
            raise ValueError("Give noise_scale (fixed d) or learn_noise, not both.")
        super().__init__(num_joints, **kwargs)
        self.noise_logit_scale = noise_logit_scale
        self.elbo_t0 = self.loss_eps if elbo_t0 is None else elbo_t0
        if learn_noise:  # DPoser's fixed d (a buffer) gives way to the property below
            del self._buffers["noise_d"]
            self.noise_logits = nn.Parameter(torch.zeros(3 * num_joints))
        else:
            self.register_parameter("noise_logits", None)

    @property
    def noise_d(self) -> torch.Tensor:
        """d [3J]: sqrt(3J softmax(noise_logit_scale * logits)), or without learn_noise DPoser's fixed noise_scale."""
        logits = self.noise_logits  # before __init__ has set it, this raises AttributeError (hasattr is False)
        if logits is None:
            return self._buffers["noise_d"]
        logits = self.noise_logit_scale * logits
        return torch.sqrt(logits.numel() * torch.softmax(logits, dim=0))

    @property
    def isotropic_noise(self) -> bool:
        return self.noise_logits is None and bool((self.noise_d == 1).all())

    # ---- log-SNR ----

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """-log SNR(t) of the noise without d: log(std^2 / alpha^2); coordinate k has gamma + 2 log d_k."""
        log_alpha = self.log_alpha(t)
        return 2 * torch.log(-torch.expm1(2 * log_alpha)) - 2 * log_alpha

    def gamma_prime(self, t: torch.Tensor) -> torch.Tensor:
        """d gamma / dt = beta (1 + alpha^2) / (1 - alpha^2)."""
        alpha2 = torch.exp(2 * self.log_alpha(t))
        return self.beta(t) * (1 + alpha2) / (1 - alpha2)

    def time_of_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        """The inverse of gamma(t): from e = e^(gamma / 2) = (1 - alpha^2) / alpha, alpha = 2 / (sqrt(e^2 + 4) + e),
        so -log alpha = log1p((e + e^2 / (sqrt(e^2 + 4) + 2)) / 2); then t > 0 of the quadratic log alpha = -t^2
        (beta_max - beta_min) / 4 - t beta_min / 2. Both without cancellation (t from 1e-5 to 1 in float32)."""
        e = torch.exp(gamma / 2)
        log_alpha = -torch.log1p((e + e * e / (torch.sqrt(e * e + 4) + 2)) / 2)
        a, b = (self.beta_max - self.beta_min) / 4, self.beta_min / 2
        return -2 * log_alpha / (b + torch.sqrt(b * b - 4 * a * log_alpha))

    # ---- the ELBO ----

    def elbo_terms(self, x0: torch.Tensor, generator: Optional[torch.Generator] = None):
        """The three terms [B] of -ELBO (prior, reconstruction, diffusion) in nats of the density of the z-scored
        vectors x0 [B, 3J]; see the module docstring."""
        n, device, dtype = x0.shape[0], x0.device, x0.dtype
        d = self.noise_d.to(dtype)
        ends = torch.tensor([self.elbo_t0, 1.0], device=device, dtype=dtype)
        g0, g1 = self.gamma(ends)
        alpha1, std1 = (v[1] for v in self.marginal(ends))
        prior = 0.5 * (std1 ** 2 - 1 - torch.log(std1 ** 2) + alpha1 ** 2 * (x0 / d) ** 2).sum(-1)
        # E_q -log N(x_0; z_0 / alpha_0, (std_0 / alpha_0)^2 d^2) = 1/2 log(2 pi e s0^2) + log d_k per coordinate,
        # s0^2 = std_0^2 / alpha_0^2 = e^gamma_0
        rec = (0.5 * (math.log(2 * math.pi) + 1 + g0) + torch.log(d)).sum().expand(n)
        u = (torch.rand((), generator=generator, device=device, dtype=dtype)
             + torch.arange(n, device=device, dtype=dtype) / n) % 1
        t = self.time_of_gamma(g0 + (g1 - g0) * u)
        z = torch.randn(x0.shape, generator=generator, device=device, dtype=dtype)
        alpha, std = self.marginal(t)
        xt = alpha[:, None] * x0 + std[:, None] * d * z
        diff = 0.5 * (g1 - g0) * ((z - self.noise_prediction(xt, t)) ** 2).sum(-1)
        return prior, rec, diff

    def loss(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """-ELBO per coordinate (so that its size is that of DPoser's loss), averaged over the batch. In eval mode the
        times and the noise come from a fixed seed, as DPoser's validation loss."""
        self.check_cond(cond, len(pose))
        device, dtype = self.device, self.dtype
        generator = None if self.training else torch.Generator(device=device).manual_seed(0)
        x0 = self.normalize(move(pose.detach().reshape(-1, self.num_joints, 3, 3), device, dtype))
        return sum(self.elbo_terms(x0, generator)).mean() / x0.shape[1]
