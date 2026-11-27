"""Completed poses outside the joint ranges of the training poses: how many, how wrong, and what dropping them would
give (docs/experiments/ace/2026-11-27_compact_fisher_basis.md). On the CPU, from the root of the repository:

    python explore/ace_energy/range_hypotheses.py                       # the network, P+L and the MLP on P+L
    python explore/ace_energy/range_hypotheses.py --rows P+L=diag2/inpaint_products.pt

The rows are completion files of complete.py --out with the poses saved (rows made with --rank), 200 validation poses,
K = 10, four occlusions. A hypothesis is outside if any of its 19 joints has a coordinate (twist, two of swing) outside
the 0.05%..99.95% quantiles of the training poses (joint_ranges.Ranges). Printed per row and occlusion: the share of
hypotheses outside, the mean error (MPVPE, mm) of those inside and outside, the mean over the poses of the mean of
the hypotheses inside (the mean of all K where none is inside: dropping without the ground truth), and the mean of all;
then the joints most often outside (SMPL numbering of LEARNED).
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import ROTATIONS, SPLIT  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.ace_features import LEARNED  # noqa: E402
from joint_ranges import Ranges  # noqa: E402

FOLDER = Path("~/data/experiments/ace").expanduser()
ROWS = ["network=diag2/inpaint_network_rank.pt", "P+L=diag2/inpaint_products.pt", "MLP P+L=diag5/inpaint_mlp_pl_k8.pt"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="+", default=ROWS, help="name=file, the file relative to " + str(FOLDER))
    args = parser.parse_args()
    ranges = Ranges(load_reference("train", ROTATIONS, SPLIT))
    print("| prior | occlusion | outside (% of hyp.) | mean inside | mean outside | mean, outside dropped | mean all |")
    print("|---|---|---|---|---|---|---|")
    for row in args.rows:
        label, f = row.split("=", 1)
        occlusions = torch.load(FOLDER / f, weights_only=True)["occlusions"]
        per_joint, dropped_all, all_all = torch.zeros(19), [], []
        for name, o in occlusions.items():
            x, e = o["poses"].double(), torch.as_tensor(o["mpvpe"]).double()  # [N, K, 19, 3, 3], [N, K]
            c = ranges.coordinates(x.reshape(-1, 19, 3, 3))
            out_j = ((c < ranges.low) | (c > ranges.high)).any(-1)  # [N K, 19]
            per_joint += out_j.double().sum(0)
            out = out_j.any(1).reshape(e.shape)
            keep = (~out).double()
            dropped = torch.where(keep.sum(1) > 0, (e * keep).sum(1) / keep.sum(1).clamp(min=1), e.mean(1))
            outside = f"{float(e[out].mean()):.1f}" if out.any() else "–"
            print(f"| {label} | {name} | {100 * float(out.double().mean()):.1f} | {float(e[~out].mean()):.1f} | "
                  f"{outside} | {float(dropped.mean()):.2f} | {float(e.mean()):.2f} |")
            dropped_all.append(float(dropped.mean()))
            all_all.append(float(e.mean()))
        print(f"| {label} | **all** | | | | {sum(dropped_all) / len(dropped_all):.2f} | "
              f"{sum(all_all) / len(all_all):.2f} |")
        top = per_joint.argsort(descending=True)[:5]
        print(f"| {label} | joints most often outside (SMPL index: hypotheses) | "
              f"{', '.join(f'{LEARNED[int(j)]}: {int(per_joint[j])}' for j in top)} | | | | |")


if __name__ == "__main__":
    main()
