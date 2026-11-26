"""The density of every joint's own rotation as a reference energy for an ACE fit (as the ZBL repulsion next to an
atomic cluster expansion): it carries the limits of the joints, and the expansion is fitted to what remains.

    python explore/ace_energy/joint_density.py --n 5000        # the bandwidth of every joint, chosen on validation

    prior = JointDensity.from_poses(train, n=5000, s0=0.005)   # or a bandwidth per joint
    prior.energy(x, s)          # -std sum_j log p_{s,j}(x_j) [B], up to a function of s, differentiable in x

p_{s,j} is the noised density of joint j, a mixture over n training poses: sum_m k(C_mj^T x_j) with the kernels
k(R) = exp(kappa (tr R - 3) / 2), a matrix Fisher density with the isotropic concentration kappa. kappa(s) has the l = 1
moment of the IGSO(3) heat kernel at s + s0: s is the noise of RSGM (the mixture of the heat kernels is the noised
empirical density), s0 the bandwidth that smooths the finite number of poses (one for all joints, or one per joint: the spine moves little,
the arms a lot). A polynomial in the entries of x inside
the exponential, so smooth everywhere on SO(3).
"""
import argparse
import math
import sys
from pathlib import Path

import torch


def _kappa_table(points: int = 2000):
    """(log s, log kappa): the l = 1 moment (1 + 2 cos omega) / 3 of exp(kappa cos omega) (1 - cos omega) d omega is
    that of the heat kernel, exp(-s)."""
    omega = torch.linspace(0, math.pi, 20001, dtype=torch.float64)[1:]
    kappa = torch.logspace(-3, 4, points, dtype=torch.float64)
    w = (torch.exp(kappa[:, None] * (omega.cos() - 1)) * (1 - omega.cos()))
    moment = (w * (1 + 2 * omega.cos())).sum(1) / (3 * w.sum(1))
    s = -torch.log(moment.clamp(1e-12, 1))
    order = s.argsort()
    return s[order].clamp_min(1e-12).log(), kappa[order].log()


class JointDensity(torch.nn.Module):
    def __init__(self, centres: torch.Tensor, s0):
        super().__init__()
        s0 = torch.as_tensor(s0, dtype=torch.float32).flatten()  # a number, or one per joint
        if len(s0) not in (1, centres.shape[0]):
            raise ValueError(f"s0 has {len(s0)} values; 1 or one per joint ({centres.shape[0]}) are needed.")
        self.register_buffer("s0", s0.expand(centres.shape[0]).clone())
        self.register_buffer("centres", centres.float())  # [J, 9, n]
        log_s, log_kappa = _kappa_table()
        self.register_buffer("log_s", log_s.float(), persistent=False)
        self.register_buffer("log_kappa", log_kappa.float(), persistent=False)

    @classmethod
    def from_poses(cls, poses: torch.Tensor, n: int, s0, seed: int = 0) -> "JointDensity":
        pick = torch.randperm(len(poses), generator=torch.Generator().manual_seed(seed))[:n]
        return cls(poses[pick].flatten(-2).permute(1, 2, 0), s0)

    def kappa(self, s: torch.Tensor) -> torch.Tensor:
        """The concentrations [B, J] of noise levels s [B]."""
        u = torch.log(s.to(self.log_s)[:, None] + self.s0)
        i = torch.searchsorted(self.log_s, u).clamp(1, len(self.log_s) - 1)
        x0, x1, y0, y1 = self.log_s[i - 1], self.log_s[i], self.log_kappa[i - 1], self.log_kappa[i]
        return (y0 + (y1 - y0) * (u - x0) / (x1 - x0)).exp()

    def log_density(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """log p_{s,j}(x_j) [B, J] of poses x [B, J, 3, 3] at s [B], up to a function of s (the norm of the kernel)."""
        tr = torch.bmm(x.flatten(-2).transpose(0, 1), self.centres)  # [J, B, n]: tr(C^T x)
        return torch.logsumexp(0.5 * self.kappa(s).to(x).T[..., None] * (tr - 3), dim=-1).transpose(0, 1)

    def energy(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """E~ref [B] = -std(s) sum_j log p_{s,j}(x_j), std = sqrt(1 - exp(-s))."""
        std = torch.sqrt(-torch.expm1(-s)).to(x)
        return -std * self.log_density(x, s.to(x)).sum(1)

    def scaled_score(self, x: torch.Tensor, s: torch.Tensor, graph: bool = False) -> torch.Tensor:
        """-grad E~ref [B, J, 3] along x_j exp(eps e_a), in closed form; with graph=True differentiable in x (DPS)."""
        with torch.set_grad_enabled(graph and torch.is_grad_enabled()):
            tr = torch.bmm(x.flatten(-2).transpose(0, 1), self.centres)
            k = self.kappa(s).to(x)  # [B, J]
            pi = torch.softmax(0.5 * k.T[..., None] * (tr - 3), dim=-1)  # [J, B, n]
            mean = torch.bmm(pi, self.centres.transpose(1, 2)).transpose(0, 1).reshape(x.shape)  # sum pi C [B, J, 3, 3]
            r = mean.transpose(-1, -2) @ x  # d/d eps of tr(mean^T x exp(eps hat(a))) = tr(r hat(a)) = -(r - r^T)^vee
            v = torch.stack([r[..., 1, 2] - r[..., 2, 1], r[..., 2, 0] - r[..., 0, 2], r[..., 0, 1] - r[..., 1, 0]], -1)
            std = torch.sqrt(-torch.expm1(-s)).to(x)
            return std[:, None, None] * 0.5 * k[..., None] * v


def main() -> None:
    """The bandwidth of every joint with the lowest validation loss of the density alone, as arguments of fit.py."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--n", type=int, default=5000, help="training poses")
    parser.add_argument("--s-max", type=float, default=0.04, help="the noise levels of the choice: the bins below")
    parser.add_argument("--poses", type=int, default=2000, help="validation poses per bin")
    parser.add_argument("--grid", type=float, nargs="+",
                        default=[2.5e-4, 5e-4, 1e-3, 1.5e-3, 2.5e-3, 4e-3, 6e-3, 1e-2, 1.5e-2, 2.5e-2])
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect_energy as inspect  # here: that module imports this one
    from fit import REFERENCE, ROTATIONS, SPLIT, Bins
    from hp.evaluation.data import load_reference
    from hp.models.io import load_model

    train, valid = load_reference("train", ROTATIONS, SPLIT), load_reference("valid", ROTATIONS, SPLIT)
    ref = load_model(Path(REFERENCE).expanduser(), device="cpu", weights="weights_ema").eval()
    bins = Bins(ref, ref.loss_t_min, 16)
    ks = [k for k in range(16) if float(bins.edges[k + 1].exp()) <= args.s_max]
    draws = [inspect.stratified_draws(ref, bins, k, valid, args.poses, 1) for k in ks]
    loss = []
    for s0 in args.grid:
        prior = JointDensity.from_poses(train, args.n, s0)
        per = [((torch.cat([prior.scaled_score(d["xt"][i:i + 250].float(), d["s"][i:i + 250])
                            for i in range(0, len(d["xt"]), 250)]).double() - d["target"]) ** 2).sum(2).mean(0)
               for d in draws]
        loss.append(torch.stack(per).mean(0))  # [J]
    loss = torch.stack(loss)  # [grid, J]
    best = loss.argmin(0)
    print(f"validation loss, sum over the joints, mean over {len(ks)} bins: one s0 {float(loss.sum(1).min()):.3f}, "
          f"one per joint {float(loss.min(0).values.sum()):.3f}")
    print("--joint-density", args.n, " ".join(f"{args.grid[i]:g}" for i in best.tolist()))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    main()
