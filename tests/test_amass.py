"""AMASS dataset: rotations passed as `data` give the same dataset as when they are loaded from the files."""
import numpy as np
import pytest
import torch

from hp.data.amass import AMASS
from hp.data.amass_cfg import ORIGINAL_AMASS_SPLITS


@pytest.fixture
def quaternions(tmp_path, monkeypatch):
    """Random rotation files <Dataset>.pt of the test split, as the preprocessing writes them, and their content."""
    rng = np.random.default_rng(0)
    parts = []
    for name in ORIGINAL_AMASS_SPLITS["test"]:
        q = rng.normal(size=(20, 55, 4))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        torch.save(torch.from_numpy(q), tmp_path / f"{name}.pt")
        parts.append(q)
    monkeypatch.setenv("HUPROSO_AMASS_ROTATIONS", str(tmp_path))
    monkeypatch.setenv("HUPROSO_AMASS_SPLIT", "original")
    monkeypatch.delenv("HUPROSO_SUBSET", raising=False)
    return np.concatenate(parts)


def test_data_from_files(quaternions):
    ds = AMASS(mode="test")
    assert ds.datasets == ORIGINAL_AMASS_SPLITS["test"]
    assert ds.data.shape == (len(quaternions), 19, 3, 3)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_provided_data_equals_files(quaternions, as_tensor):
    data = torch.from_numpy(quaternions) if as_tensor else quaternions
    provided = AMASS(data=data)
    assert provided.datasets == []
    torch.testing.assert_close(provided.data, AMASS(mode="test").data)


def test_provided_subset(quaternions):
    idx = [3, 0, 17, 25]
    torch.testing.assert_close(AMASS(data=quaternions[idx]).data, AMASS(mode="test").data[idx])


def test_loading_as_float32_gives_the_same_poses(quaternions):
    default = AMASS(mode="test", rotation_representation="matrix")
    assert default.load_data().dtype == np.float64  # the files' type, unless asked otherwise
    small = AMASS(mode="test", rotation_representation="matrix", load_dtype="float32")
    assert small.load_data().dtype == np.float32
    assert torch.equal(small.data, default.data)
