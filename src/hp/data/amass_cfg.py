# Partially taken from https://github.com/cvlab-epfl/adv_param_pose_prior/blob/main/lib/datasets/amass.py

from enum import Enum
from typing import List

AmassDatasets = Enum(
    "AmassDatasets",
    [
        "ACCAD",
        "CMU",
        "KIT",
        "EyesJapanDataset",
        "WEIZMANN",
        "BMLrub",
        "BMLmovi",
        "SFU",
        "TotalCapture",
        "Transitions",
        "EKUT",
        "HDM05",
        "HumanEva",
        "MoSh",
        "SSM",
        "TCDHands",
        "PosePrior",
    ],
    start=0,
)
AmassJoints = Enum(
    "AmassJoints",
    [
        "ROOT",
        "LeftUpLeg",
        "RightUpLeg",
        "Spine0",
        "LeftLeg",
        "RightLeg",
        "Spine1",
        "LeftFoot",
        "RightFoot",
        "Spine2",
        "LeftToeBase",
        "RightToeBase",
        "Neck",
        "LeftShoulder",
        "RightShoulder",
        "Head",
        "LeftArm",
        "RightArm",
        "LeftElbow",
        "RightElbow",
        "LeftHand",
        "RightHand",
    ],
    start=0,
)
AmassJoints4Learning: List[AmassJoints] = [
    aj for aj in AmassJoints if aj.value not in [0, 10, 11]
]

ORIGINAL_AMASS_SPLITS = {
    "valid": ["HumanEva", "HDM05", "SFU", "MoSh"],  # 117679 samples
    "test": ["Transitions", "SSM"],  # 1320126 samples
    "train": [
        "CMU",
        "PosePrior",
        "TotalCapture",
        "EyesJapanDataset",
        "KIT",
        "BMLrub",
        "EKUT",
        "TCDHands",
        "ACCAD",
    ],  # 14779964 samples
}

# The AMASS of PoseD-Flow (Nadar et al., CVPR 2026; the preprocessing of DPoser: SMPL bodies, 30 Hz, the first and last
# 10% of each sequence trimmed, the usual dataset split), converted by scripts/preprocess/convert_posedflow.py into one
# file per split. Select with amass_split: posedflow.
POSEDFLOW_AMASS_SPLITS = {
    "train": ["PoseDFlow_train"],  # 18,611,047 poses
    "valid": ["PoseDFlow_valid"],  # 860,080 poses
    "test": ["PoseDFlow_test"],  # 54,200 poses
}

# Small splits (~0.7M frames in total) for quick experiments.
# Select with the environment variable HUPROSO_SUBSET=1.
SUBSET_AMASS_SPLITS = {
    "train": ["ACCAD", "EKUT", "TotalCapture", "TCDHands"],
    "valid": ["MoSh"],
    "test": ["SSM"],
}
