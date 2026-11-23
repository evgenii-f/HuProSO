import json

import numpy as np
import pytest
import torch

from hp.evaluation import denoising_results as R
from hp.evaluation.denoising import METRICS_VERSION


def write_row(cache, label, split, sigma, spec, settings, mpjpe, names):
    folder = cache / R.folder_of(label)
    folder.mkdir(parents=True, exist_ok=True)
    key = {"model": {"spec": spec}, "split": split, "sigma": sigma, "names": names, "settings": settings,
           "version": METRICS_VERSION}
    m = torch.tensor(mpjpe, dtype=torch.float64)
    torch.save({"key": json.dumps(key), "pred": torch.zeros(1),
                "metrics": {"mpjpe": m, "mpvpe": 2 * m, "input_mpjpe": 3 * m}},
               folder / f"{split}_{round(sigma * 1000)}mm.pt")


def test_cache_rows_carry_their_label_settings_budget_and_per_sequence_errors(tmp_path):
    names = ["HDM05/a/x", "SFU/b/y"]
    write_row(tmp_path, "RSGM res_norm: their c10, DPoser method", "valid", 0.04, {"run": "r", "method": "dposer"},
              {"steps_per_iter": 67}, [[10.0, 20.0], [30.0, 40.0]], names)
    write_row(tmp_path, "PoseD-Flow: their checkpoint", "test", 0.1, {"posedflow": "c.ckpt"}, {},
              [[1.0, 1.0], [1.0, 1.0]], names)
    rows = R.load_cache(tmp_path, models={"RSGM res_norm: their c10, DPoser method": {"run": "r", "method": "dposer"}})
    assert len(rows) == 2 and len(R.load_cache(tmp_path, split="valid")) == 1
    r = R.find(rows, "RSGM res_norm: their c10, DPoser method", 0.04, "valid")
    assert r["mpjpe"] == 25.0 and r["mpvpe"] == 50.0 and r["method"] == "dposer" and r["model"] == "r"
    assert np.allclose(r["per_sequence"]["mpjpe"], [15.0, 35.0])
    assert r["steps"] == 3 * 67 and r["nfe"] == 3 * 67 and r["settings"]["t_max"] == 0.25  # DPoser's default
    flow = [x for x in rows if x["split"] == "test"][0]
    assert flow["label"] == "PoseD-Flow: their checkpoint" and flow["steps"] == 400 and flow["nfe"] == 400 * 100
    assert R.by_group(r) == {"HDM05": 15.0, "SFU": 35.0}


def test_old_metrics_versions_are_left_out(tmp_path):
    write_row(tmp_path, "w/o prior", "valid", 0.04, {"method": "dposer"}, {}, [[1.0]], ["a/b"])
    path = next(tmp_path.glob("*/*.pt"))
    saved = torch.load(path, weights_only=True)
    key = json.loads(saved["key"])
    key["version"] = METRICS_VERSION - 1
    saved["key"] = json.dumps(key)
    torch.save(saved, path)
    assert R.load_cache(tmp_path) == []


def test_without_a_prior_there_are_no_network_calls():
    assert R.budget({"method": "dposer"}, 0.1, {}) == {"steps": 240, "nfe": 0}


def test_paired_difference_and_its_interval():
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0, 5.0])
    p = R.paired(a, b)
    assert p["mean"] == -1.0 and p["lo"] == p["hi"] == -1.0 and p["a_better"] == 1.0
    q = R.paired(np.r_[a, 10.0], np.r_[b, 0.0], n_boot=2000)
    assert q["lo"] < q["mean"] < q["hi"]


def test_humaneva_groups_by_subject():
    assert R.dataset_of("HumanEva/S2/Walking_1_stageii_batch000") == "S2"
    assert R.dataset_of("MoSh/50002/x_batch001") == "MoSh"


@pytest.mark.skipif(not __import__("pathlib").Path("~/data/experiments/denoising/optuna/rsgm_sds_40mm.db")
                    .expanduser().is_file(), reason="needs the study of rsgm_sds_40mm")
def test_the_study_of_40mm_has_its_best_trial():
    rows = R.load_study("~/data/experiments/denoising/optuna/rsgm_sds_40mm.db")
    best = min(rows, key=lambda r: r["mpjpe"])
    assert len(rows) == 40 and abs(best["mpjpe"] - 18.687) < 0.01 and best["settings"]["steps_per_iter"] == 67
    assert abs(best["settings"]["t_min"] / best["settings"]["t_max"] - 0.387) < 0.001 and best["nfe"] == 201
    assert best["study"] == "rsgm_sds_40mm" and best["trial"] == 36 and "t_min_fraction" in best["params"]


def test_a_row_made_with_other_settings_than_its_spec_is_marked(tmp_path):
    spec = {"run": "r", "method": "dposer", "settings": {0.04: {"t_max": 0.1}}}
    write_row(tmp_path, "RSGM: c10, DPoser method", "test", 0.04, {**spec, "settings": {0.04: {"t_max": 0.05}}},
              {"t_max": 0.05}, [[1.0]], ["a/b"])
    assert R.load_cache(tmp_path, models={"RSGM: c10, DPoser method": spec})[0]["label"] == \
        "RSGM: c10, DPoser method [other settings]"
    write_row(tmp_path, "RSGM: c10, DPoser method", "test", 0.04, spec, {"t_max": 0.1}, [[1.0]], ["a/b"])
    assert R.load_cache(tmp_path, models={"RSGM: c10, DPoser method": spec})[0]["label"] == "RSGM: c10, DPoser method"


def test_a_row_in_two_caches_is_taken_from_the_first(tmp_path):
    for name, value in (("a", 1.0), ("b", 2.0)):
        write_row(tmp_path / name, "w/o prior", "valid", 0.04, {"method": "dposer"}, {}, [[value]], ["x/y"])
    rows = R.load([tmp_path / "a", tmp_path / "b"])
    assert len(rows) == 1 and rows[0]["mpjpe"] == 1.0
