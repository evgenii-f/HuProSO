"""Where per-joint noise loses in inpainting: the error of the hidden leg joints split along the principal axes of
each joint. Hypothesis (2026-11-28): with the Brownian time c_j s(t), c_j by the RMS spread of the joint
(fit_joint_scale), the narrow axes of the wide joints (twist and abduction of the hips and knees) are destroyed first,
and that is where abl_norm_aniso loses against the isotropic abl_norm.

Inpainting of RSGM (hp.evaluation.completion.run, method="inpaint") on the first --poses of the 1000 validation poses
of the Optuna studies, K = 10, occlusions left_leg and legs, at one point of each model's study with close APD:
abl_norm trial 118 (APD 6.57 cm on the 1000 poses) and abl_norm_aniso trial 97 (6.68 cm). The principal axes of
joint j: the eigenvectors of the covariance of log(M_j^T x_j) over 200,000 training poses (M_j the Karcher mean,
hp.models.rsgm.z_statistics). The error of a completion along axis i: v_i^T (log(M_j^T x_hat) - log(M_j^T x)).

Rule, fixed before the run: on legs, over the hips and knees, let D_i be the mean squared error along axis i (0 wide,
2 narrowest) of abl_norm_aniso minus that of abl_norm, and D = D_0 + D_1 + D_2. The hypothesis holds if D > 0,
D_1 + D_2 >= 2/3 D and D_0 <= 1/3 D. Validation only. From the root:

    python explore/ace_energy/axis_errors.py --poses 500 --device mps
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import roma
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hp.evaluation.completion import KEEP, OCCLUSIONS, Body, build, metrics, run, summary, valid_poses  # noqa: E402
from hp.models.rsgm import z_statistics  # noqa: E402

E = "~/data/experiments/"
ROT = "~/data/AMASS_rotations_posedflow_c10"
MODELS = {  # the points of rsgm_inpaint_mean_apd_norm (t118) and rsgm_inpaint_mean_apd_norm_aniso (t97)
    "abl_norm": (E + "rsgm_posedflow_abl_norm_s10/run_0001",
                 dict(steps=200, schedule="edm", replace=False, guidance=0.09515197497012039, resample=5)),
    "abl_norm_aniso": (E + "igso_speed/rsgm_posedflow_abl_norm_aniso_s10/run_0001",
                       dict(steps=400, schedule="edm", replace=False, guidance=0.04805330485651132, resample=5)),
}
OUT = Path("~/data/experiments/ace/axis").expanduser()
NAMES21 = ["LeftUpLeg", "RightUpLeg", "Spine0", "LeftLeg", "RightLeg", "Spine1", "LeftFoot", "RightFoot", "Spine2",
           "LeftToe", "RightToe", "Neck", "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm", "LeftElbow",
           "RightElbow", "LeftHand", "RightHand"]
TESTED = ("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg")


def axes():
    """Karcher means M [19, 3, 3] and principal axes V [19, 3, 3] (columns, widest first) with stds [19, 3]."""
    from fit import ROTATIONS, SPLIT  # noqa: E402
    from hp.evaluation.data import load_reference  # noqa: E402

    train = load_reference("train", ROTATIONS, SPLIT)
    train = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:200_000]].double()
    zm, _ = z_statistics(train)
    m = roma.rotvec_to_rotmat(torch.tensor(zm, dtype=torch.float64).reshape(-1, 3))
    vs, stds = [], []
    for j in range(len(m)):
        lam, v = torch.linalg.eigh(torch.cov(roma.rotmat_to_rotvec(m[j].T @ train[:, j]).T))
        vs.append(v.flip(-1))
        stds.append(lam.flip(0).clamp_min(0).sqrt())
    return m, torch.stack(vs), torch.stack(stds)


def axis_sq(pred, gt, m, v, j19):
    """Squared error [N, K, 3] along the principal axes of joint j19 (index among the 19)."""
    a = roma.rotmat_to_rotvec(m[j19].T @ pred.double())  # [N, K, 3]
    b = roma.rotmat_to_rotvec(m[j19].T @ gt.double())[:, None]  # [N, 1, 3]
    return ((a - b) @ v[j19]) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses", type=int, default=500)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=250)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gt, body = valid_poses(ROT, 1000)[:args.poses], Body()
    m, v, stds = axes()
    sq = {}
    print("| model | occlusion | MPVPE mean | best of 10 | APD | minutes |\n|---|---|---|---|---|---|", flush=True)
    for name, (path, opts) in MODELS.items():
        model = build(dict(run=path), args.device)
        for occ in ("left_leg", "legs"):
            f = OUT / f"{name}_{occ}_{args.poses}.pt"
            t0 = time.time()
            if f.exists():
                pred = torch.load(f, weights_only=True)
            else:
                torch.manual_seed(10)
                pred = run(model, gt, occ, 10, batch=args.batch, together=True, progress=False, method="inpaint",
                           prior=opts).cpu()
                torch.save(pred, f)
            s = summary(metrics(pred, gt, occ, body))
            print(f"| {name} | {occ} | {s['mpvpe_mean']:.2f} | {s['mpvpe_min']:.2f} | {s['apd']:.2f} | "
                  f"{(time.time() - t0) / 60:.1f} |", flush=True)
            for j21 in OCCLUSIONS[occ]:
                if j21 in KEEP:
                    j19 = KEEP.index(j21)
                    sq[name, occ, NAMES21[j21]] = axis_sq(pred[:, :, j21], gt[:, j21], m, v, j19)  # [N, K, 3]
    rng = np.random.default_rng(0)
    rows = []
    for occ in ("left_leg", "legs"):
        print(f"\n== {occ}: RMS error along the principal axes (rad; axis 0 widest), aniso - iso of the mean squared "
              f"error (rad^2, 95% bootstrap over poses)")
        print("| joint | train std (rad) | iso RMS | aniso RMS | D_0 | D_1 | D_2 |\n|---|---|---|---|---|---|---|")
        tot = np.zeros(3)
        per_pose = 0
        for joint in [NAMES21[j] for j in OCCLUSIONS[occ] if j in KEEP]:
            a, b = sq["abl_norm", occ, joint], sq["abl_norm_aniso", occ, joint]
            d = (b - a).mean(1).numpy()  # [N, 3]
            cells = []
            for i in range(3):
                idx = rng.integers(0, len(d), (5000, len(d)))
                lo, hi = np.quantile(d[idx, i].mean(1), [0.025, 0.975])
                cells.append(f"{d[:, i].mean():+.5f} [{lo:+.5f}, {hi:+.5f}]")
            sd = stds[KEEP.index(NAMES21.index(joint))]
            print(f"| {joint} | {sd[0]:.3f} / {sd[1]:.3f} / {sd[2]:.3f} | "
                  + " / ".join(f"{float(x):.3f}" for x in a.mean((0, 1)).sqrt()) + " | "
                  + " / ".join(f"{float(x):.3f}" for x in b.mean((0, 1)).sqrt()) + " | " + " | ".join(cells) + " |")
            if joint in TESTED:
                tot += d.mean(0)
                per_pose = per_pose + d
        if occ == "legs":
            dd = tot.sum()
            idx = rng.integers(0, len(per_pose), (5000, len(per_pose)))
            share = per_pose[idx].mean(1)
            narrow_share = (share[:, 1] + share[:, 2]) / share.sum(1)
            lo, hi = np.quantile(narrow_share, [0.025, 0.975])
            holds = dd > 0 and tot[1] + tot[2] >= 2 / 3 * dd and tot[0] <= 1 / 3 * dd
            print(f"\nrule (legs, hips and knees): D = {dd:+.5f} rad^2, D_0 {tot[0]:+.5f}, D_1 {tot[1]:+.5f}, "
                  f"D_2 {tot[2]:+.5f}; narrow share (D_1 + D_2) / D = "
                  f"{(tot[1] + tot[2]) / dd if dd else float('nan'):.2f} [{lo:.2f}, {hi:.2f}]; "
                  f"hypothesis {'holds' if holds else 'does not hold'}")
            rows.append(dict(D=dd, D0=tot[0], D1=tot[1], D2=tot[2], holds=bool(holds)))
    (OUT / f"rule_{args.poses}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
