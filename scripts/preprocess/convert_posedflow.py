"""Convert the AMASS of PoseD-Flow (github.com/circle-group/PoseDFlow, the preprocessing of DPoser) to our rotation files.

Their data has one folder per split with pose_body.pt: axis-angle rotations [N, 63] of the 21 body joints of SMPL
(joints 1 ... 21: our AmassJoints 1 ... 21, also the two toes). Ours is one <Dataset>.pt per dataset with quaternions
[N, joints, 4] as [x, y, z, w], of which the loader reads the first 22 joints (hp/data/amass.py). So every split
becomes PoseDFlow_<split>.pt with [N, 22, 4]: the root (joint 0) the identity, joints 1 ... 21 from pose_body.pt.
Select them with amass_split: posedflow (hp/data/amass_cfg.py, POSEDFLOW_AMASS_SPLITS).

    python scripts/preprocess/convert_posedflow.py --src ~/data/posedflow_amass --dst ~/data/AMASS_rotations_posedflow
    python scripts/preprocess/convert_posedflow.py --src ~/data/posedflow_amass --dst ~/data/AMASS_rotations_posedflow_c10 \
        --train-stride 10 --valid-stride 100

`--train-stride N` keeps every N-th training pose (their poses are at 30 Hz already). `--valid-stride N` the same for
the validation poses: their training validates on every 100th (manifm/amass_dataset.py). The test poses are kept whole.
"""
import argparse
from pathlib import Path

import roma
import torch

SPLITS = ("train", "valid", "test")


def convert(axis_angle: torch.Tensor, chunk: int = 1_000_000) -> torch.Tensor:
    """[N, 63] axis-angle of joints 1 ... 21 -> [N, 22, 4] float32 quaternions [x, y, z, w], joint 0 the identity."""
    n = axis_angle.shape[0]
    out = torch.zeros(n, 22, 4, dtype=torch.float32)
    out[:, 0, 3] = 1.0
    for start in range(0, n, chunk):
        part = axis_angle[start:start + chunk].reshape(-1, 21, 3).double()
        out[start:start + chunk, 1:] = roma.rotvec_to_unitquat(part).float()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="folder with train/, valid/, test/ of PoseD-Flow")
    parser.add_argument("--dst", required=True, help="folder of the rotation files to write")
    parser.add_argument("--train-stride", type=int, default=1, help="keep every N-th training pose")
    parser.add_argument("--valid-stride", type=int, default=1, help="keep every N-th validation pose")
    args = parser.parse_args()
    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = dst / f"PoseDFlow_{split}.pt"
        if target.is_file():
            print(f"{target} exists, skipped")
            continue
        poses = torch.load(src / split / "pose_body.pt", weights_only=True, mmap=True)
        stride = {"train": args.train_stride, "valid": args.valid_stride}.get(split, 1)
        if stride > 1:
            poses = poses[::stride]
        if poses.ndim != 2 or poses.shape[1] != 63:
            raise ValueError(f"{src / split / 'pose_body.pt'}: expected axis-angle [N, 63], got {tuple(poses.shape)}")
        quats = convert(poses)
        tmp = target.with_suffix(".pt.tmp")
        torch.save(quats, tmp)
        tmp.rename(target)  # complete or not at all
        print(f"{split}: {len(quats):,} poses -> {target}")


if __name__ == "__main__":
    main()
