from pathlib import Path

import numpy as np
import pytest

from hp.data.compare import dataset_report, overview, sequences, speed_retained, verify
from hp.data.compression import run
from test_compression import config, raw  # noqa: F401  (fixtures and helpers)


@pytest.fixture
def pair(raw, tmp_path):  # noqa: F811
    run(config(raw, tmp_path, stride=5, crop_start_s=0.0, crop_end_s=0.0))
    return str(raw), str(tmp_path / "out")


def test_sequences_lists_missing_and_extra(pair):
    raw_dir, comp_dir = pair
    assert len(sequences(raw_dir, comp_dir, "A")["common"]) == 2
    (Path(comp_dir) / "A" / "s1" / "b_stageii.npz").unlink()
    seq = sequences(raw_dir, comp_dir, "A")
    assert len(seq["common"]) == 1 and seq["common"][0].endswith("a_stageii.npz")
    assert len(seq["missing"]) == 1 and seq["missing"][0].endswith("b_stageii.npz")


def test_overview_counts(pair):
    raw_dir, comp_dir = pair
    row = overview(raw_dir, comp_dir, ["A"])[0]
    assert row["sequences"] == 2 and row["frames_orig"] == 1241 and row["frames_comp"] == 200 + 49


def test_verify_detects_a_modified_file(pair):
    raw_dir, comp_dir = pair
    rels = sequences(raw_dir, comp_dir, "A")["common"]
    assert verify(raw_dir, comp_dir, rels)["bad"] == []
    path = f"{comp_dir}/{rels[0]}"
    data = dict(np.load(path, allow_pickle=True))
    data["poses"] = data["poses"] + 1e-3
    np.savez(path, **data)
    assert verify(raw_dir, comp_dir, rels)["bad"] == [rels[0]]


def test_dataset_report(pair):
    raw_dir, comp_dir = pair
    rep = dataset_report(raw_dir, comp_dir, "A", n_ref=300, n_w=100, n_seq=2, n_verify=2)
    assert rep["n_orig"] == 1241 and rep["n_comp"] == 249 and rep["verify"]["bad"] == []
    assert len(rep["recall"]) == 300 and len(rep["w1"]) == 19
    assert rep["recall"].min() > 0  # references are dropped frames, never kept ones
    assert 0 < speed_retained(rep["joint_speed"]) and rep["root_speed"][0].shape[0] == 1241
