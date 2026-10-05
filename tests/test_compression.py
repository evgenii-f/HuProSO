import json
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from hp.data.compression import compress, kept_frames, run


def write_sequence(path, n, fps=120.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(n)
    np.savez(
        path,
        poses=rng.normal(size=(n, 165)),
        trans=rng.normal(size=(n, 3)),
        root_orient=rng.normal(size=(n, 3)),
        markers=rng.normal(size=(n, 80, 3)),
        betas=np.arange(16.0),
        gender=np.array("male"),
        surface_model_type=np.array("smplx"),
        mocap_frame_rate=np.array(fps),
    )


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    write_sequence(str(root / "A" / "s1" / "a_stageii.npz"), 1000)
    write_sequence(str(root / "A" / "s1" / "b_stageii.npz"), 241)
    write_sequence(str(root / "B" / "s2" / "c_stageii.npz"), 600, fps=60.0)
    (root / "A" / "s1").joinpath("male_stagei.npz").write_bytes(b"not a sequence")
    return root


def config(raw, tmp_path, **kwargs):
    cfg = dict(stride=4, crop_start_s=0.5, crop_end_s=0.25, output_dir=str(tmp_path / "out"),
               datasets=None, raw_dir=str(raw), num_workers=1, overwrite=False)
    cfg.update(kwargs)
    return OmegaConf.create(cfg)


def test_kept_frames():
    idx, (a, b) = kept_frames(1000, 120.0, 4, 0.5, 0.25)
    assert (a, b) == (60, 970) and idx[0] == 60 and idx[-1] <= 969 and set(np.diff(idx)) == {4}


def test_kept_frames_keeps_one_frame_when_crops_are_too_long():
    idx, _ = kept_frames(100, 120.0, 4, 10.0, 10.0)
    assert len(idx) == 1


def test_compress_in_memory():
    seq = dict(poses=np.arange(100)[:, None] * np.ones((1, 165)), trans=np.zeros((100, 3)), fps=100.0, gender="male")
    out = compress(seq, stride=5)
    assert out["fps"] == 20.0 and len(out["poses"]) == 20 and np.array_equal(out["index"], np.arange(0, 100, 5))


def test_run_writes_compressed_sequences(raw, tmp_path):
    manifest = run(config(raw, tmp_path))
    out = tmp_path / "out"
    src = np.load(raw / "A" / "s1" / "a_stageii.npz")
    dst = np.load(out / "A" / "s1" / "a_stageii.npz")
    idx = dst["frame_index"]
    assert idx[0] == 60 and set(np.diff(idx)) == {4}
    assert np.array_equal(dst["poses"], src["poses"][idx]) and np.array_equal(dst["trans"], src["trans"][idx])
    assert float(dst["mocap_frame_rate"]) == 30.0 and np.array_equal(dst["betas"], src["betas"])
    assert "markers" not in dst.files and not (out / "A" / "s1" / "male_stagei.npz").exists()
    assert float(np.load(out / "B" / "s2" / "c_stageii.npz")["mocap_frame_rate"]) == 15.0
    assert manifest["total"]["sequences"] == 3 and manifest["datasets"]["A"]["frames_before"] == 1241
    assert json.load(open(out / "compression.json"))["params"]["stride"] == 4


def test_run_only_selected_datasets(raw, tmp_path):
    manifest = run(config(raw, tmp_path, datasets=["B"]))
    assert list(manifest["datasets"]) == ["B"] and not (tmp_path / "out" / "A").exists()


def test_run_resumes_and_keeps_counts(raw, tmp_path):
    first = run(config(raw, tmp_path))
    path = tmp_path / "out" / "A" / "s1" / "a_stageii.npz"
    mtime = path.stat().st_mtime_ns
    second = run(config(raw, tmp_path))
    assert path.stat().st_mtime_ns == mtime and second == first


def test_run_refuses_other_parameters(raw, tmp_path):
    run(config(raw, tmp_path))
    with pytest.raises(ValueError, match="compressed with"):
        run(config(raw, tmp_path, stride=10))
    run(config(raw, tmp_path, stride=10, overwrite=True))
    assert json.load(open(tmp_path / "out" / "compression.json"))["params"]["stride"] == 10


def test_run_rejects_invalid_settings(raw, tmp_path):
    with pytest.raises(ValueError):
        run(config(raw, tmp_path, stride=0))
    with pytest.raises(ValueError):
        run(config(raw, tmp_path, output_dir=str(raw)))
