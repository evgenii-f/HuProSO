"""The Fokker-Planck term of an energy's training (embedding.py --fp-weight): the noised densities of RSGM obey the heat
equation of IGSO(3), so E_s = -log p_s obeys

    dE/d log s = s/2 (Laplace E - |grad E|^2) + s c'(s)

(fp_residual.py measures it with exact differences). Denoising score matching trains every level only near the data
noised at that level; far from it (the data noised at a larger s') the levels need not agree, and at small s they do
not (fp_residual.py: hundreds of nats per unit of log s). The term ties them: at points of both kinds,

    R = dE/d log s - s/2 (Laplace E - |grad E|^2) - a(s),    loss = Huber_delta(R),

with a(s) = s c'(s) a learned number per bin of log s (linear between the centres, as the coefficients). Estimates in
float32 that keep the graph of the parameters: grad E from the scaled score; Laplace E by Hutchinson with central
differences, v^T H v ~ (E(x exp(h v)) + E(x exp(-h v)) - 2 E(x)) / h^2, v ~ N(0, I_57), h = H_SCALE sqrt(s) within
[H_MIN, H_MAX]; dE/d log s by central differences of +-DELTA in log s.

Points: the training poses of the step noised at a level s drawn log-uniformly over the range of the loss (not
uniformly in t as the loss: then only 7% of the levels would be below s = 0.02, where the violations are), and as many
noised at s' drawn log-uniformly in [s, S_FAR], then scored at the level s.
"""
import math

import torch

from fit import draw
from hp.models.rsgm import exp_frame

DELTA = 0.02     # step in log s
H_SCALE, H_MIN, H_MAX = 0.2, 1e-3, 0.05
S_FAR = 2.0


class FokkerPlanck(torch.nn.Module):
    def __init__(self, model, ref, probes: int = 2, huber: float = 5.0):
        super().__init__()
        self.model, self.ref, self.probes, self.huber = model, ref, probes, huber
        self.a = torch.nn.Parameter(torch.zeros(len(model.bins.centres)))
        self.model_ref = None

    def offset(self, s: torch.Tensor) -> torch.Tensor:
        """a(s) [B], linear in log s between the bin centres."""
        c = self.model.bins.centres
        pos = ((s.log() - c[0]) / (c[1] - c[0])).clamp(0, len(c) - 1)
        lo = pos.floor().long().clamp(max=len(c) - 2)
        w = (pos - lo).to(self.a)
        lo = lo.to(self.a.device)
        return (1 - w) * self.a[lo] + w * self.a[lo + 1]

    def energy(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """E_s [B] = E~ / std at the levels s [B] (float64 on the CPU)."""
        t = self.ref.t_of_s(s).float().to(x.device)
        return self.model(x, t) / torch.sqrt(-torch.expm1(-s)).to(x)

    def points(self, x0: torch.Tensor, gen: torch.Generator):
        """(x [2B, 19, 3, 3] float64, s [2B] float64): x0 noised at s, and noised at s' >= s, both scored at s."""
        n = len(x0)
        lo, hi = self.ref.s(torch.tensor([self.ref.loss_t_min, 1.0], dtype=torch.float64)).log()
        s = (lo + (hi - lo) * torch.rand(n, dtype=torch.float64, generator=gen)).exp()
        t = self.ref.t_of_s(s)
        far = (s.log() + torch.rand(n, dtype=torch.float64, generator=gen) * (math.log(S_FAR) - s.log())).exp()
        near_x = draw(self.ref, x0, t, gen)[0]
        far_x = draw(self.ref, x0, self.ref.t_of_s(far), gen)[0]
        return torch.cat([near_x, far_x]), torch.cat([s, s])

    def residual(self, x: torch.Tensor, s: torch.Tensor, gen: torch.Generator = None) -> torch.Tensor:
        """R [B] with the graph of the parameters (x float32 on the model's device, s float64 on the CPU)."""
        n, dev = len(x), x.device
        std = torch.sqrt(-torch.expm1(-s)).to(x)
        t = self.ref.t_of_s(s).float().to(dev)
        grad = -self.model.scaled_score(x, t, create_graph=True) / std[:, None, None]  # grad E [B, 19, 3]
        h = (H_SCALE * s.sqrt()).clamp(H_MIN, H_MAX).to(x)
        v = torch.randn(self.probes, n, x.shape[1], 3, generator=gen, dtype=torch.float64).to(x)
        xs = torch.cat([x[None].expand(self.probes, -1, -1, -1, -1).reshape(-1, *x.shape[1:]),
                        exp_frame(x[None], h[None, :, None, None] * v).reshape(-1, *x.shape[1:]),
                        exp_frame(x[None], -h[None, :, None, None] * v).reshape(-1, *x.shape[1:])])
        e = self.energy(xs, s.repeat(3 * self.probes)).reshape(3, self.probes, n)
        lap = ((e[1] + e[2] - 2 * e[0]) / h ** 2).mean(0)
        up = self.energy(x, s * math.exp(DELTA))
        down = self.energy(x, s * math.exp(-DELTA))
        dlog = (up - down) / (2 * DELTA)
        sx = s.to(x)
        return dlog - sx / 2 * (lap - (grad ** 2).sum((1, 2))) - self.offset(s).to(dev)

    def bin_offsets(self, r: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """a(s) per bin as the mean of residuals r [B] at levels s [B] (bins without points: their neighbours')."""
        k = torch.bucketize(s.log(), self.model.bins.edges[1:-1])
        a = torch.full((len(self.model.bins.centres),), float("nan"), dtype=torch.float64)
        for i in range(len(a)):
            if (k == i).any():
                a[i] = r.detach().double().cpu()[k == i].mean()
        idx = torch.arange(len(a))
        ok = ~a.isnan()
        return torch.tensor([float(a[ok][(idx[ok] - i).abs().argmin()]) for i in range(len(a))], dtype=torch.float64)

    def loss(self, x0: torch.Tensor, gen: torch.Generator, device) -> torch.Tensor:
        x, s = self.points(x0, gen)
        r = self.residual(x.float().to(device), s, gen)
        return torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=self.huber)
