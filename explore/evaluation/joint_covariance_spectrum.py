"""The spread of every joint around its Karcher mean on the principal axes: the eigenvalues of the covariance of
log(M_j^T x_j) (z_cov of the z statistics of scripts/train_rsgm_posedflow.py, the joint_cov of per-axis noise), as
the stds along the three axes (deg), their ratio and the share of the first, and the first axis in the frame of the
mean. Section 15 of docs/experiments/2026-11-14_score_network_architectures.md. From the root of the repository:

    PYTHONPATH=src python explore/evaluation/joint_covariance_spectrum.py ~/data/experiments/factorial/rsgm_z_statistics.json
"""
import json
import math
import sys
from pathlib import Path

import torch

from hp.data.amass_cfg import AmassJoints4Learning


def main() -> int:
    stats = json.loads(Path(sys.argv[1]).expanduser().read_text())
    if "z_cov" not in stats:
        print("no z_cov in the file: train or dry-run a config of ANISO_X first (it adds it)")
        return 1
    cov = torch.tensor(stats["z_cov"], dtype=torch.float64).reshape(-1, 3, 3)
    lam, vec = torch.linalg.eigh(cov)
    lam, vec = lam.flip(-1), vec.flip(-1)
    sigma2 = cov.diagonal(dim1=-2, dim2=-1).mean(-1)
    c = sigma2 / sigma2.min()
    deg = 180 / math.pi
    print(f"{stats['poses']:,} training poses of {stats['rotations']}")
    print(f"{'joint':14s} {'c_j':>6s} {'std1':>6s} {'std2':>6s} {'std3':>6s} {'l1/l3':>6s} {'l1 share':>8s}  axis 1 (frame of the mean)")
    for j, name in enumerate(AmassJoints4Learning):
        sd = lam[j].clamp_min(0).sqrt() * deg
        a = vec[j, :, 0] * torch.sign(vec[j, :, 0][vec[j, :, 0].abs().argmax()])
        print(f"{name.name:14s} {c[j]:6.1f} {sd[0]:6.1f} {sd[1]:6.1f} {sd[2]:6.1f} {lam[j, 0] / lam[j, 2]:6.1f} "
              f"{lam[j, 0] / lam[j].sum():8.2f}  [{a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
