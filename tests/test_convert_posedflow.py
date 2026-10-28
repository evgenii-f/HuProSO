"""The conversion of the AMASS of PoseD-Flow (scripts/preprocess/convert_posedflow.py) and the posedflow split."""
import importlib.util
import os
from pathlib import Path
from unittest import mock

import roma
import torch

from hp.data.amass import AMASS
from hp.data.amass_cfg import AmassJoints4Learning

spec = importlib.util.spec_from_file_location("convert_posedflow", Path(__file__).parents[1] / "scripts/preprocess/convert_posedflow.py")
convert_posedflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_posedflow)


def fake_split(n, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 63, generator=g) * 0.7).float()


def test_the_rotations_survive_the_conversion():
    aa = fake_split(50, 0)
    q = convert_posedflow.convert(aa, chunk=7)
    assert q.shape == (50, 22, 4) and q.dtype == torch.float32
    assert torch.equal(q[:, 0], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(50, 4))  # the root: the identity
    back = roma.unitquat_to_rotmat(q[:, 1:].double())
    assert torch.allclose(back, roma.rotvec_to_rotmat(aa.reshape(50, 21, 3).double()), atol=1e-6)


def test_the_posedflow_split_loads_the_converted_files(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    sizes = {"train": 40, "valid": 12, "test": 9}
    for i, (split, n) in enumerate(sizes.items()):
        (src / split).mkdir(parents=True)
        torch.save(fake_split(n, i), src / split / "pose_body.pt")
    monkeypatch.setattr("sys.argv", ["convert", "--src", str(src), "--dst", str(dst), "--train-stride", "4", "--valid-stride", "5"])
    convert_posedflow.main()
    assert sorted(p.name for p in dst.iterdir()) == ["PoseDFlow_test.pt", "PoseDFlow_train.pt", "PoseDFlow_valid.pt"]
    with mock.patch.dict(os.environ, {"HUPROSO_AMASS_SPLIT": "posedflow"}):
        for split, n in {"train": 10, "valid": 3, "test": 9}.items():  # every 4th of 40, every 5th of 12
            data = AMASS(mode=split, path_to_dataset=str(dst), rotation_representation="matrix",
                         selected_joints=AmassJoints4Learning).data.reshape(-1, len(AmassJoints4Learning), 3, 3)
            assert data.shape == (n, 19, 3, 3)
    # joint k of ours is joint k of SMPL: our first joint (LeftUpLeg, 1) is their first rotation
    first = roma.rotvec_to_rotmat(fake_split(40, 0)[::4][:, :3].double()).float()
    with mock.patch.dict(os.environ, {"HUPROSO_AMASS_SPLIT": "posedflow"}):
        train = AMASS(mode="train", path_to_dataset=str(dst), rotation_representation="matrix",
                      selected_joints=AmassJoints4Learning).data.reshape(-1, 19, 3, 3)
    assert torch.allclose(torch.as_tensor(train[:, 0]).float(), first, atol=1e-5)
