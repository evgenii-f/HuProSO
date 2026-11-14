"""Motion sequences of AMASS, frame by frame in time (the preprocessed rotation files keep the frames, not the sequences).

    paths = sequence_files("Transitions")              # the *_stageii.npz files of a dataset in the raw AMASS
    poses, fps = load_sequence(paths[0], fps=30)        # [T, 19, 3, 3] rotation matrices, resampled to ~30 fps

The rotations are those of the preprocessing (scripts/preprocess/preprocess_data.py): the axis-angle `poses` of the
55 SMPL-X joints, of which the 19 of AmassJoints4Learning are kept, the same frames as in AMASS_rotations.
"""
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import roma
import torch

from hp.data.amass_cfg import AmassJoints4Learning
from hp.utils.paths import get_path

DATASET_DIRS = {"EyesJapanDataset": "Eyes_Japan_Dataset"}  # as the preprocessing


def sequence_files(dataset: str, raw_dir=None) -> List[Path]:
    """The motion files of a dataset of the raw AMASS (amass_raw of config/paths.yaml), sorted by path."""
    root = Path(get_path("amass_raw", override=raw_dir)).expanduser() / DATASET_DIRS.get(dataset, dataset)
    files = [p for p in root.glob("*/*.npz") if not p.name.startswith(".") and not p.name.endswith("_stagei.npz")]
    return sorted(files)


def load_sequence(path, fps: float = None) -> Tuple[torch.Tensor, float]:
    """The poses [T, 19, 3, 3] (float32) of a motion file and their frame rate. With `fps`, every n-th frame is kept,
    n = round(mocap rate / fps), so the frame rate is close to `fps` (e.g. 120 Hz -> 30 Hz: every 4th frame)."""
    data = np.load(os.fspath(path), allow_pickle=True)
    rate = float(data["mocap_frame_rate"]) if "mocap_frame_rate" in data.files else float(data["mocap_framerate"])
    poses = data["poses"]
    stride = max(1, int(round(rate / fps))) if fps else 1
    poses = torch.as_tensor(poses[::stride, : 55 * 3], dtype=torch.float64).reshape(-1, 55, 3)
    joints = [j.value for j in AmassJoints4Learning]
    return roma.rotvec_to_rotmat(poses[:, joints]).float(), rate / stride
