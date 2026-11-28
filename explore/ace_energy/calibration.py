"""Calibration of the spread of RSGM inpainting, isotropic against per-joint noise: for every hidden joint, the mean
squared distance of a completion to the truth against the mean squared distance between two completions of the same
pose. If the truth and the K completions were draws of the same conditional, the two would be equal:

    rho = E d(x_k, x)^2 / E d(x_k, x_l)^2  (k != l),  1 if calibrated,

rho > 1 the completions are too close to each other for their distance to the truth (overconfident), rho < 1 too
spread. d is the geodesic angle (rad); per principal axis i of the joint, the coordinate v_i^T log(M_j^T x) (axes and
Karcher means as in axis_errors.py). Completions: the cache of axis_errors.py (abl_norm t118 and abl_norm_aniso t97,
first 500 of the 1000 validation poses, K = 10, left_leg and legs; ~/data/experiments/ace/axis/).

Hypothesis (2026-11-28): per-joint noise decides all joints at the same noise level, while isotropic noise decides the
wide joints (legs) first, when the context is still unreadable; so the per-joint model is sharper on the hidden legs
than the isotropic one at the same overall APD. Rule, fixed before the run: on legs, over the hips and knees (sum of
the squared angles of the four joints), rho_aniso - rho_iso > 0 with its 95% bootstrap interval over poses above 0.
The strong form also needs rho_iso < 1 < rho_aniso. Validation only. From the root (Mac, under a minute):

    python explore/ace_energy/calibration.py > ~/data/experiments/ace/axis/calibration_500.log
"""
import sys
from pathlib import Path

import numpy as np
import roma
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from axis_errors import NAMES21, OUT, ROT, TESTED, axes  # noqa: E402
from hp.evaluation.completion import KEEP, OCCLUSIONS, valid_poses  # noqa: E402

N, K = 500, 10
MODELS = ("abl_norm", "abl_norm_aniso")


def angle2(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    tr = (r1.transpose(-1, -2) @ r2).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((tr - 1) / 2).clamp(-1, 1)) ** 2


def parts(pred: torch.Tensor, gt: torch.Tensor, m, v, j19: int):
    """Per pose [N]: to-truth and pairwise mean squared angle, and the same [N, 3] along the principal axes."""
    p, g = pred.double(), gt.double()
    truth = angle2(p, g[:, None].expand_as(p)).mean(1)
    a, b = p[:, :, None].expand(-1, -1, K, -1, -1), p[:, None].expand(-1, K, -1, -1, -1)
    off = ~torch.eye(K, dtype=torch.bool)
    pair = angle2(a, b)[:, off].mean(1)
    c = roma.rotmat_to_rotvec(m[j19].T @ p) @ v[j19]  # [N, K, 3]
    c0 = roma.rotmat_to_rotvec(m[j19].T @ g) @ v[j19]  # [N, 3]
    truth_ax = ((c - c0[:, None]) ** 2).mean(1)
    pair_ax = ((c[:, :, None] - c[:, None]) ** 2)[:, off].mean(1)
    return truth.numpy(), pair.numpy(), truth_ax.numpy(), pair_ax.numpy()


def ratio_ci(t: np.ndarray, q: np.ndarray, idx: np.ndarray):
    r = t[idx].mean(1) / q[idx].mean(1)
    return t.mean() / q.mean(), *np.quantile(r, [0.025, 0.975])


def main() -> None:
    gt = valid_poses(ROT, 1000)[:N]
    m, v, _ = axes()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, N, (5000, N))
    for occ in ("left_leg", "legs"):
        preds = {name: torch.load(OUT / f"{name}_{occ}_{N}.pt", weights_only=True) for name in MODELS}
        print(f"\n== {occ}: rho = E d(x_k, x)^2 / E d(x_k, x_l)^2 (1 calibrated, > 1 overconfident), 95% bootstrap")
        print("| joint | model | to truth (rad^2) | pairwise (rad^2) | rho | rho axis 0 / 1 / 2 |\n|---|---|---|---|---|---|")
        tot = {name: [0, 0] for name in MODELS}
        for j21 in [j for j in OCCLUSIONS[occ] if j in KEEP]:
            joint, j19 = NAMES21[j21], KEEP.index(j21)
            for name in MODELS:
                t, q, ta, qa = parts(preds[name][:, :, j21], gt[:, j21], m, v, j19)
                r, lo, hi = ratio_ci(t, q, idx)
                ax = " / ".join(f"{ta[:, i].mean() / qa[:, i].mean():.2f}" for i in range(3))
                print(f"| {joint} | {name} | {t.mean():.4f} | {q.mean():.4f} | {r:.2f} [{lo:.2f}, {hi:.2f}] | {ax} |")
                if joint in TESTED:
                    tot[name][0] = tot[name][0] + t
                    tot[name][1] = tot[name][1] + q
        res = {}
        for name in MODELS:
            t, q = tot[name]
            res[name] = (t, q, *ratio_ci(t, q, idx))
            print(f"hips and knees, {name}: rho {res[name][2]:.3f} [{res[name][3]:.3f}, {res[name][4]:.3f}]")
        (ti, qi, ri, *_), (ta, qa, ra, *_) = res["abl_norm"], res["abl_norm_aniso"]
        d = ta[idx].mean(1) / qa[idx].mean(1) - ti[idx].mean(1) / qi[idx].mean(1)
        lo, hi = np.quantile(d, [0.025, 0.975])
        line = f"rho_aniso - rho_iso = {ra - ri:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if occ == "legs":
            holds = lo > 0
            strong = holds and ri < 1 < ra
            line += f"; rule: hypothesis {'holds' if holds else 'does not hold'}" + (", strong form" if strong else "")
        print(line, flush=True)


if __name__ == "__main__":
    main()
