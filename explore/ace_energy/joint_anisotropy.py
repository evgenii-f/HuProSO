"""How anisotropic the rotations of each joint are, within the joint and between joints, and what that means for the
per-joint noise of RSGM (joint_scale, hp.models.rsgm). Three tables:

1. spread: the RMS angle of joint j around its mean (marginal; the c_j of fit_joint_scale come from it) and around
   its conditional given all other joints (the pairwise matrix Fisher energy trained by maximum likelihood,
   pl/energy_bm.pt: a draw of p(x_j | x_-j) for each of 4000 validation poses, RMS angle to the truth / sqrt 2);
2. anisotropy: the principal stds of joint j (square roots of the eigenvalues of the covariance of log(M_j^T x_j)
   over 200,000 training poses, M_j the Karcher mean of z_statistics), the per-coordinate stds in the parent axes (as
   z_statistics and DPoser's z-normalisation see them), and the conditional principal stds of the Fisher energy (the
   concentration about axis i of a matrix Fisher with proper singular values s is s_j + s_k, the std 1/sqrt of it,
   averaged over the validation poses);
3. drowning times: the Brownian time at which the noise of an axis equals its spread, s* = sigma_axis^2 / c_j, over
   the 57 axes, for isotropic noise (c = 1), c_j by the RMS spread (fit_joint_scale) and c_j by the narrowest axis.

From the root (Mac, a few minutes):

    python explore/ace_energy/joint_anisotropy.py > ~/data/experiments/ace/pl/joint_anisotropy.log
"""
import math
import sys
from pathlib import Path

import roma
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import ROTATIONS, SPLIT  # noqa: E402
from gibbs import load, local, sample_fisher  # noqa: E402
from pseudo_lik import ACE, proper_svd  # noqa: E402
from hp.data.amass_cfg import AmassJoints4Learning  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.rsgm import z_statistics  # noqa: E402

NAMES = [j.name for j in AmassJoints4Learning]


def angle(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    tr = (r1.transpose(-1, -2) @ r2).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((tr - 1) / 2).clamp(-1, 1))


def main() -> None:
    train = load_reference("train", ROTATIONS, SPLIT)
    train = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:200_000]].double()
    valid = load_reference("valid", ROTATIONS, SPLIT).double()[:4000]
    zm, zs = z_statistics(train)
    means = roma.rotvec_to_rotmat(torch.tensor(zm, dtype=torch.float64).reshape(-1, 3))
    coord = torch.tensor(zs, dtype=torch.float64).reshape(-1, 3)
    model = load(str(ACE / "pl" / "energy_bm.pt"))
    w, a = model.coupling(), model.a
    gen = torch.Generator().manual_seed(0)
    marg_rms, cond_rms, principal, cond_principal = [], [], [], []
    for j in range(len(NAMES)):
        v = roma.rotmat_to_rotvec(means[j].T @ train[:, j])
        principal.append(torch.linalg.eigvalsh(torch.cov(v.T)).flip(0).clamp_min(0).sqrt())
        marg_rms.append(float(angle(valid[:, j], means[j].expand_as(valid[:, j])).pow(2).mean().sqrt()))
        f, _ = local(model, valid, None, w, a, j)
        cond_rms.append(float((angle(sample_fisher(f, gen), valid[:, j]).pow(2).mean() / 2).sqrt()))
        _, s, _ = proper_svd(f)
        kappa = torch.stack([s[:, 1] + s[:, 2], s[:, 0] + s[:, 2], s[:, 0] + s[:, 1]], -1).clamp_min(1e-6)
        cond_principal.append((1 / kappa).mean(0).sqrt().sort(descending=True).values)

    print("== 1. spread: RMS angle around the mean (marginal, 4000 validation poses) and around the Fisher conditional")
    print("| joint | marginal RMS (rad) | conditional RMS (rad) | c marginal | c conditional | cond / marg |")
    print("|---|---|---|---|---|---|")
    mm, mc = min(marg_rms), min(cond_rms)
    for n, x, y in zip(NAMES, marg_rms, cond_rms):
        print(f"| {n} | {x:.3f} | {y:.3f} | {(x / mm) ** 2:.1f} | {(y / mc) ** 2:.1f} | {y / x:.2f} |")

    print("\n== 2. anisotropy: principal stds (rad, widest first; 200,000 training poses), per-coordinate stds in the "
          "parent axes, conditional principal stds (Fisher, 4000 validation poses)")
    print("| joint | marginal principal std (rad) | marg. max/min | marg. max/mid | per-coordinate std (parent axes) "
          "| coord. max/min | conditional principal std (rad) | cond. max/min |")
    print("|---|---|---|---|---|---|---|---|")
    for n, ev, pc, cs in zip(NAMES, principal, coord, cond_principal):
        print(f"| {n} | {ev[0]:.3f} / {ev[1]:.3f} / {ev[2]:.3f} | {ev[0] / ev[2]:.1f} | {ev[0] / ev[1]:.1f} | "
              f"{pc.max():.3f} / {pc.min():.3f} | {pc.max() / pc.min():.1f} | {cs[0]:.3f} / {cs[1]:.3f} / {cs[2]:.3f} | "
              f"{cs[0] / cs[2]:.1f} |")

    print("\n== 3. drowning times s* = sigma_axis^2 / c_j over the 57 principal axes")
    rms = [math.sqrt(float((ev ** 2).mean())) for ev in principal]
    narrow = [float(ev[2]) for ev in principal]
    scales = {"isotropic (c = 1)": [1.0] * len(NAMES),
              "c_j by RMS spread (fit_joint_scale)": [(r / min(rms)) ** 2 for r in rms],
              "c_j by the narrowest axis": [(x / min(narrow)) ** 2 for x in narrow]}
    print("| noise | c_j range | range of s* | first 5 axes destroyed (s*) | last axis |")
    print("|---|---|---|---|---|")
    for name, c in scales.items():
        s = sorted((float(ev[i]) ** 2 / c[j], NAMES[j], i) for j, ev in enumerate(principal) for i in range(3))
        first = ", ".join(f"{n}[{i}] {x:.5f}" for x, n, i in s[:5])
        print(f"| {name} | {min(c):.1f}-{max(c):.1f} | {s[-1][0] / s[0][0]:.0f}x | {first} | "
              f"{s[-1][1]}[{s[-1][2]}] {s[-1][0]:.4f} |")
    print("  c_j by the narrowest axis: " + ", ".join(f"{n} {(x / min(narrow)) ** 2:.1f}" for n, x in zip(NAMES, narrow)))


if __name__ == "__main__":
    main()
