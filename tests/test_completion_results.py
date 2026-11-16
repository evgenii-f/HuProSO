"""Results of pose completion on the validation poses in one form (hp/evaluation/completion_results.py)."""
import json

import optuna
import pytest

from hp.evaluation.completion import OCCLUSIONS
from hp.evaluation.completion_results import at_occlusion, load, load_jsonl, load_study


def cells(base):
    """Per occlusion values base + i (mean), base / 2 + i (best of K), 1 + i (APD), as the scripts write them."""
    return {o: {"mpvpe_mean": base + i, "mpvpe_min": base / 2 + i, "apd": 1.0 + i} for i, o in enumerate(OCCLUSIONS)}


def test_jsonl_of_both_scripts(tmp_path):
    path = tmp_path / "valid.jsonl"
    rows = [dict(model="small", spec={"run": "~/r"}, variant="dps0.1", options={"guidance": 0.1}, poses=100,
                 occlusions=cells(100), mean=0, best=0, seconds=6.0, device="mps"),
            dict(model="small", spec={"run": "~/r"}, poses=100, ode="heun-edm50", occlusions=cells(90), mean=0, best=0,
                 seconds=96.0, device="mps"),
            dict(model="DPoser full", method="dposer", options=None, poses=100, occlusions=cells(80), mean=0, best=0,
                 seconds=4.0)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    inpaint, dflow, baseline = load_jsonl(path)
    assert (baseline["method"], baseline["setting"], baseline["options"]) == ("dposer", "default", {})
    assert baseline["device"] is None and baseline["mean"] == pytest.approx(81.5)
    path.write_text(json.dumps(dict(model="x", method="vae", poses=1, occlusions=cells(1))) + "\n")
    with pytest.raises(ValueError):
        load_jsonl(path)
    assert (inpaint["method"], inpaint["setting"], inpaint["options"]) == ("inpaint", "dps0.1", {"guidance": 0.1})
    assert (dflow["method"], dflow["setting"]) == ("dflow", "heun-edm50")
    assert inpaint["mean"] == pytest.approx(101.5) and inpaint["best_of_k"] == pytest.approx(51.5)
    assert inpaint["apd"] == pytest.approx(2.5)
    assert inpaint["run"] == "~/r" and inpaint["poses"] == 100 and inpaint["seconds"] == 6.0
    arms = at_occlusion(inpaint, "arms")
    assert arms["mean"] == 102 and arms["apd"] == 3 and at_occlusion(inpaint, None) is inpaint


def make_study(path, name, config=None, params=({"lr": 0.1}, {"lr": 0.2}), attrs=None):
    study = optuna.create_study(study_name=name, storage=f"sqlite:///{path}", directions=["minimize", "maximize"])
    if config:
        study.set_user_attr("config", config)
    for i, p in enumerate(params):
        trial = study.ask()
        for key, value in p.items():
            trial.suggest_float(key, 0.01, 1.0)
        values = cells(100 + 10 * i)
        for o in OCCLUSIONS:
            for key, value in values[o].items():
                trial.set_user_attr(f"{o}_{key}", value)
        trial.set_user_attr("minutes", 2.0)
        for key, value in (attrs or {}).items():
            trial.set_user_attr(key, value)
        study.tell(trial, [1.0, 1.0])
    study.ask()  # a running trial: left out


def test_a_study_with_its_configuration(tmp_path):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    path = tmp_path / "rsgm_inpaint_mean_apd_deep.db"
    make_study(path, "rsgm_inpaint_mean_apd_deep",
               dict(model="~/data/experiments/deep/run_0001", method="inpaint", poses=1000, fixed={"schedule": "edm"}),
               params=({"guidance": 0.1}, {"guidance": 0.2}), attrs={"host": "workstation2", "device": "cuda:0"})
    records = load_study(path)
    assert [r["setting"] for r in records] == ["trial 0", "trial 1"]
    first = records[0]
    assert first["method"] == "inpaint" and first["poses"] == 1000 and first["run"] == "~/data/experiments/deep/run_0001"
    assert first["options"]["schedule"] == "edm" and "guidance" in first["options"]
    assert first["seconds"] == 120 and first["device"] == "workstation2 cuda:0"
    assert records[1]["mean"] == pytest.approx(111.5)


def test_an_old_study_by_its_name_and_load_of_several_files(tmp_path):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    path = tmp_path / "old.db"
    make_study(path, "completion_dposer_rsgm_posedflow_small_s10_v1000")
    make_study(tmp_path / "bad.db", "something_else")
    records = load_study(path, "completion_dposer_rsgm_posedflow_small_s10_v1000")
    assert records[0]["method"] == "dposer" and records[0]["poses"] == 1000
    assert records[0]["run"] == "~/data/experiments/rsgm_posedflow_small_s10/run_0001" and records[0]["device"] is None
    with pytest.raises(ValueError):
        load_study(tmp_path / "bad.db", "something_else")
    assert len(load([path, tmp_path / "missing.jsonl"])) == 2
