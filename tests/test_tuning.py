"""The choice of the settings of the completion in `scripts/evaluate.py all`: ready, a grid or a study."""
import json
import sys
from pathlib import Path

import optuna
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"), str(Path(__file__).resolve().parent)]

import evaluate  # noqa: E402
import tune_completion  # noqa: E402
from hp.evaluation import protocol_cli as cli  # noqa: E402
from hp.evaluation import references as R  # noqa: E402
from hp.evaluation import tuning as T  # noqa: E402
from test_references import fake  # noqa: E402, F401

METHODS = ["inpaint", "dflow"]


def plan(method="inpaint", mode="grid", arch="MLP 512x4", **kwargs):
    config = R.load()
    row, own = R.set_for(config, method, arch)
    return T.make_plan(config, method, mode, "~/data/experiments/x_s10/run_0001", row, own, **kwargs)


def test_modes():
    assert T.modes(["study"], METHODS) == {"inpaint": "study", "dflow": "study"}
    assert T.modes(["inpaint=grid", "dflow=ready"], METHODS) == {"inpaint": "grid", "dflow": "ready"}
    assert T.modes(["ready", "inpaint=grid"], METHODS) == {"inpaint": "grid", "dflow": "ready"}
    for wrong in (["grid"], ["inpaint=grid"], ["fast"], ["sds=grid"]):
        with pytest.raises(cli.UsageError):
            T.modes(wrong, METHODS)
    assert T.modes(["grid"], ["inpaint"]) == {"inpaint": "grid"}


def test_plans():
    grid = plan()
    assert len(grid.grid) == grid.config["n_trials"] == 10 and grid.config["params"]["steps"] == [400]
    assert plan(arch="timeres 256x8").config["params"]["steps"] == [200]
    unknown = plan(arch="MLP 9x9")
    assert unknown.config["params"]["steps"] == [400, 200] and len(unknown.grid) == 20 and unknown.minutes is None
    study = plan(mode="study")
    assert study.config["n_trials"] == 120 and study.config["first"][0]["guidance"] == pytest.approx(0.0577, abs=1e-4)
    assert plan(mode="study", n_trials=30).name == study.name  # the number of trials is not part of the study
    assert plan(method="dflow", mode="study", arch="timeres 1024x2").config["n_trials"] == 60
    assert len({grid.name, study.name, unknown.name, plan(batch=100).name}) == 4


@pytest.mark.parametrize("method, mode", [("inpaint", "grid"), ("inpaint", "study"), ("dflow", "study")])
def test_configurations_are_read_by_the_tuning_script(tmp_path, method, mode):
    import yaml

    file = tmp_path / "x.yaml"
    file.write_text(yaml.safe_dump(plan(method, mode).config))
    config = tune_completion.load_config(file)
    study = optuna.create_study()
    for first in config["first"]:
        study.enqueue_trial(first)
    trials = [study.ask() for _ in config["first"]]
    assert [tune_completion.values_of(config, t) for t in trials] == [
        {**config["fixed"], **first} for first in config["first"]]


def test_announce_says_that_it_is_not_one_setting():
    assert "GRID of completion inpaint: 10 settings" in T.announce(plan(), 3, "cuda:1")
    assert "3 done, 7 to run" in T.announce(plan(), 3, "cuda:1")
    assert "OPTUNA STUDY of completion dflow: 60 settings" in T.announce(plan("dflow", "study"), 0, "cuda:1")


def test_choose_and_ends():
    points = [dict(mean=m, apd=a, params={"steps": 400, "guidance": g}) for m, a, g in
              ((86.0, 6.2, 0.06), (88.0, 6.5, 0.04), (90.0, 6.9, 0.02))]
    best, note = T.choose(points, 6.4)
    assert best["mean"] == 88.0 and not note
    best, note = T.choose(points, 7.5)
    assert best["apd"] == 6.9 and "no point" in note
    grid = plan().grid
    assert T.at_end({"params": grid[0]}, grid) and not T.at_end({"params": grid[4]}, grid)


def test_missing_runs_stopped_trials_again():
    grid = plan().grid
    trials = [dict(state="COMPLETE", params=grid[0]), dict(state="FAIL", params=grid[1]),
              dict(state="RUNNING", params=grid[2]), dict(state="WAITING", params=grid[3])]
    assert T.missing(grid, trials) == [grid[1], grid[2], *grid[4:]]


def argv_of(fake, *more):  # noqa: F811
    return [a for a in fake[1] if a != "--like"] + list(more)


def test_the_completion_has_no_default(fake, capsys):  # noqa: F811
    for more in ([], ["--like", "denoising=rsgm budget"], ["--like", "rsgm", "--tune", "study"], ["--tune", "grid"],
                 ["--tune", "study", "--quick", "--yes"], ["--tune", "study", "--split", "test", "--yes"],
                 ["--tune", "study", "--protocols", "denoising"]):
        with pytest.raises(SystemExit):
            evaluate.main(argv_of(fake, *more))
    with pytest.raises(SystemExit):
        evaluate.main(argv_of(fake, "--tune", "study"))  # not a terminal, no --yes
    assert not fake[0]
    assert evaluate.main(argv_of(fake, "--protocols", "denoising")) == 0  # no completion, nothing to choose
    assert "no default" in capsys.readouterr().err


def finished_study(monkeypatch, tmp_path, stop_after=None):
    """tune() on a fake objective: mean and APD fall with guidance."""
    monkeypatch.setattr(tune_completion, "STUDY_DIR", str(tmp_path / "studies"))
    monkeypatch.setattr(tune_completion, "valid_poses", lambda *a: [None] * 1000)
    monkeypatch.setattr(tune_completion, "Body", lambda: None)
    monkeypatch.setattr(tune_completion, "build", lambda *a: None)
    ran = []

    def objective_of(config, *_):
        def objective(trial):
            values = tune_completion.values_of(config, trial)
            ran.append(values["guidance"])
            if stop_after is not None and len(ran) == stop_after:
                raise KeyboardInterrupt
            for o in tune_completion.OCCLUSIONS:
                trial.set_user_attr(f"{o}_mpvpe_mean", 100 - 100 * values["guidance"])
                trial.set_user_attr(f"{o}_mpvpe_min", 50.0)
                trial.set_user_attr(f"{o}_apd", 7 - 10 * values["guidance"])
            return 100 - 100 * values["guidance"], 7 - 10 * values["guidance"]
        return objective
    monkeypatch.setattr(tune_completion, "objective_of", objective_of)
    return ran


def test_grid_end_to_end(fake, monkeypatch, tmp_path):  # noqa: F811
    ran = finished_study(monkeypatch, tmp_path)
    argv = argv_of(fake, "--tune", "grid", "--methods", "inpaint", "--yes")
    assert evaluate.main(argv) == 0
    assert len(ran) == 10
    completion = [a for n, a in fake[0] if n == "completion"]
    settings = json.loads(completion[0].settings)
    assert len(completion) == 1 and settings["guidance"] == 0.0585 and settings["steps"] == 400  # APD 6.415
    report = (fake[2] / "mine_valid.md").read_text()
    assert "grid of 10 settings" in report and "chosen on these same validation poses" in report
    assert "ready settings" not in report
    assert evaluate.main(argv) == 0 and len(ran) == 10  # finished trials are not run again


def test_grid_is_continued(fake, monkeypatch, tmp_path):  # noqa: F811
    ran = finished_study(monkeypatch, tmp_path, stop_after=4)
    argv = argv_of(fake, "--tune", "grid", "--methods", "inpaint", "--yes", "--protocols", "completion")
    with pytest.raises(KeyboardInterrupt):  # Ctrl+C twice
        evaluate.main(argv)
    assert not fake[0]
    ran = finished_study(monkeypatch, tmp_path)
    assert evaluate.main(argv) == 0
    assert len(ran) == 7 and sorted(ran)[0] == 0.0342 and len(set(ran)) == 7


def test_dry_run_only_announces(fake, capsys, monkeypatch, tmp_path):  # noqa: F811
    ran = finished_study(monkeypatch, tmp_path)
    assert evaluate.main(argv_of(fake, "--tune", "inpaint=grid", "dflow=ready", "--dry-run")) == 0
    assert "GRID of completion inpaint" in capsys.readouterr().out and not ran
    assert not Path(tmp_path / "studies").exists()
