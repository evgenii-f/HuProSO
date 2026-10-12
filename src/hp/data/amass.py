# Partially taken from https://github.com/cvlab-epfl/adv_param_pose_prior/blob/main/lib/datasets/amass.py

from typing import List
import logging

import numpy as np
import os
import torch

from hp.utils.rotation_tools import RotationConverterTorchRoma as RotationConverterTorch
from hp.utils.skeleton_forward_kinematics import perform_forward_kinematics
from hp.data.amass_cfg import AmassJoints, AmassJoints4Learning, ORIGINAL_AMASS_SPLITS, SUBSET_AMASS_SPLITS
from hp.utils.paths import get_path, get_amass_split


class AMASS(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: list = [],
        path_to_dataset: str = None,  # default: "amass_rotations" in config/paths.yaml
        mode: str = None,  # 'train', 'valid', 'test'
        selected_joints: List[AmassJoints] = AmassJoints4Learning,
        rotation_representation: str = "matrix",
        last_dim_flattened: bool = True,
        data: np.ndarray = None,  # instead of the files: quaternions [N, joints, 4] as [x, y, z, w], like <Dataset>.pt
        no_preprocess: bool = False,
        conditioning_modality: str = "",  # "3D", "2D"
    ):

        self.mode = mode
        self.datasets = datasets
        if mode is not None:
            splits = SUBSET_AMASS_SPLITS if get_amass_split() == "subset" else ORIGINAL_AMASS_SPLITS
            self.datasets = splits[mode]

        self.data = data

        logging.info(f"Using the following datasets of AMASS: {self.datasets}.")

        self.path_to_dataset = get_path("amass_rotations", override=path_to_dataset)

        self.selected_joints = selected_joints
        self.selected_joints_ints = [sj.value for sj in selected_joints]

        self.rotation_representation = rotation_representation
        self.last_dim_flattened = last_dim_flattened

        if no_preprocess:
            self.data = self.load_data()
        else:
            self.get_dataset()

        self.feat_for_conditioning = None
        if conditioning_modality in ["2D", "3D"]:
            if self.rotation_representation != "matrix":
                self.data = RotationConverterTorch.convert_from_to(
                    self.data, self.rotation_representation, "matrix"
                )

            self.feat_for_conditioning = perform_forward_kinematics(
                self.data, selected_joints=selected_joints
            )

            if self.rotation_representation != "matrix":
                self.data = RotationConverterTorch.convert_from_to(
                    self.data, "matrix", self.rotation_representation
                )

            self.feat_for_conditioning = self.feat_for_conditioning.to(torch.float32)
            if conditioning_modality == "2D":
                self.feat_for_conditioning = self.feat_for_conditioning[..., :2]
            self.feat_for_conditioning = self.feat_for_conditioning.reshape(
                self.data.shape[0], -1
            )

        elif conditioning_modality == "rotation":
            self.feat_for_conditioning = self.data.reshape(
                -1, len(selected_joints), 3, 3
            )[..., :2]
        elif conditioning_modality == "":
            pass
        else:
            raise NotImplementedError(
                f"Conditioning with {conditioning_modality} is not implemented."
            )

        logging.info(f"Datasets {self.datasets} loaded. Number of samples: {len(self)}")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.tensor:
        sample = self.data[idx]
        if self.last_dim_flattened:
            sample = sample.flatten()

        if self.feat_for_conditioning is not None:
            data = dict()
            data["theta"] = sample
            data["condition"] = self.feat_for_conditioning[idx]
        else:
            data = sample

        return data

    def load_data(self, standard_amass: bool = False) -> np.ndarray:

        if not standard_amass:
            data = []
            for d in self.datasets:
                d_name = os.path.join(self.path_to_dataset, f"{d}.pt")
                if not os.path.isfile(d_name):
                    raise FileNotFoundError(
                        f"Preprocessed dataset {d_name} not found. Create it with "
                        "scripts/preprocess/preprocess_amass_data.sh, point 'amass_rotations' "
                        "in config/paths.yaml to the right folder, or set "
                        "'amass_split: subset' to use only the preprocessed subset."
                    )
                data.append(torch.load(d_name).numpy()[:, :22])
            data = np.concatenate(data, axis=0)
        else:
            ds_dir = f"XXX/VXX_SVXX_TXX/stage_I/{self.mode}/"
            if not os.path.exists(ds_dir):
                raise FileNotFoundError(f"Directory {ds_dir} does not exist.")
            data = torch.load(ds_dir + "pose.pt").reshape(-1, 52, 3)[:, :22]
            data = RotationConverterTorch.convert_from_to(
                data, "axis_angle", "quaternion"
            )
        return data

    def get_dataset(self) -> None:
        if self.data is None:
            data = self.load_data()
        else:
            logging.info("Using provided data.")
            # as an array, like load_data(), so the quaternion convention is swapped below
            data = self.data.numpy() if torch.is_tensor(self.data) else np.asarray(self.data)
        data = data[:, self.selected_joints_ints, :]

        # According to pytorch3d convention [w,x,y,z] instead of [x,y,z,w].
        if np.ndarray == type(data):
            data = torch.tensor(data, dtype=torch.float32)
            data[..., [0, 1, 2, 3]] = data[..., [3, 0, 1, 2]]  # inverse: [1,2,3,0]
            logging.debug(f"Swapping to use quaternion convention [w,x,y,z].")

        self.data = RotationConverterTorch.convert_from_to(
            data, "quaternion", self.rotation_representation
        )

    def sample(self, n_samples: int) -> torch.tensor:
        if len(self) < n_samples:
            n_samples = len(self)
            logging.warning(
                f"Sample size {n_samples} is larger than dataset size {len(self)}. Reduced the number of evaluated samples to {n_samples}."
            )
        idx = np.random.choice(np.arange(len(self)), n_samples, replace=False)
        sample = self.data[idx]

        if self.last_dim_flattened:
            sample = sample.reshape(n_samples, -1)

        if self.feat_for_conditioning is not None:
            data = dict()
            data["theta"] = sample
            data["condition"] = self.feat_for_conditioning[idx]
        else:
            data = sample

        return data
