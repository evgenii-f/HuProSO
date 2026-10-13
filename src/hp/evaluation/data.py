"""The real poses a model is evaluated on."""
import os
from typing import Optional
from unittest import mock

import torch

from hp.data.amass import AMASS
from hp.data.amass_cfg import AmassJoints4Learning


def load_reference(split: str = "test", rotations: Optional[str] = None, amass_split: Optional[str] = None) -> torch.Tensor:
    """Poses [N, 19, 3, 3] of a split ("train", "valid" or "test") of AMASS, as float32 rotation matrices.

    `rotations` is the folder of the <Dataset>.pt files (None: amass_rotations of config/paths.yaml), and
    `amass_split` is "original" or "subset" (None: amass_split of config/paths.yaml).
    """
    env = {} if amass_split is None else {"HUPROSO_AMASS_SPLIT": amass_split}
    with mock.patch.dict(os.environ, env):
        data = AMASS(mode=split, path_to_dataset=rotations, rotation_representation="matrix",
                     selected_joints=AmassJoints4Learning).data
    return data.reshape(-1, len(AmassJoints4Learning), 3, 3).float()
