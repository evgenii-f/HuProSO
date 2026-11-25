"""The sampling-weight study (scripts/tune_weighting.py): weights, metrics, objectives, known runs, fronts."""
import importlib.util
import math
from pathlib import Path

import optuna
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tune_weighting", ROOT / "scripts" / "tune_weighting.py")
tw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tw)

CONFIG = ROOT / "config" / "weighting_tuning" / "rdw_inpaint.yaml"


def test_weights():
    r = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])  # median 4
    assert tw.weights_of(r, 0.0, 0.1, 10.0).tolist() == [1.0] * 5
    assert tw.weights_of(r, 1.0, 0.5, 2.0).tolist() == [0.5, 0.5, 1.0, 2.0, 2.0]
    w = tw.weights_of(r, 3.3, 0.1, 10.0)
    assert w.min() >= 0.1 and w.max() <= 10.0
    assert tw.ess(torch.ones(7)) == pytest.approx(1.0)


def test_objectives():
    assert tw.parse_objective("generation.wasserstein") == ("generation", "wasserstein", "minimize")
    assert tw.parse_objective("generation.log_likelihood")[2] == "maximize"
    assert tw.parse_objective("inpaint.hv_best")[2] == "maximize"
    assert tw.parse_objective("dflow.band_mean")[2] == "minimize"
    for bad in ("generation.hv_best", "inpaint.wasserstein", "wasserstein", "other.mean"):
        with pytest.raises(ValueError):
            tw.parse_objective(bad)


def test_completion_metrics():
    # one point that covers the whole band: area 4.5 x (reference - value)
    m = tw.completion_metrics([dict(mean=95.0, best_of_k=50.0, apd=9.0)])
    assert m["hv_mean"] == pytest.approx(4.5 * 20, abs=0.1)
    assert m["hv_best"] == pytest.approx(4.5 * 15, abs=0.1)
    assert m["band_mean"] == pytest.approx(95.0, abs=0.05)
    assert (m["best"], m["mean"]) == (50.0, 95.0)


def test_known_run_and_config(tmp_path):
    config = tw.load_config(CONFIG)
    assert config["objectives"] == ["generation.wasserstein", "inpaint.hv_best"]
    entry = config["known"][1]
    params = {k: entry[k] for k in ("beta", "w_min", "w_max")}
    assert tw.known_run(config, params).name == "run_0001"
    assert tw.known_run(config, {**params, "w_min": 0.5}) is None
    # beta 0: the limits do not matter
    assert "fact_RDW_s10" in str(tw.known_run(config, {"beta": 0.0, "w_min": 0.7, "w_max": 3.0}))
    for name, change in (("unknown.yaml", {"extra": 1}), ("objective.yaml", {"objectives": ["dflow.hv_best"]}),
                         ("metric.yaml", {"objectives": ["inpaint.nothing"]}), ("sampler.yaml", {"sampler": "x"})):
        raw = {**yaml.safe_load(CONFIG.read_text()), **change}
        path = tmp_path / name
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError):
            tw.load_config(path)


def test_pareto_in_any_metrics():
    def attrs(w, hv, fid):
        return {"generation.wasserstein": w, "inpaint.hv_best": hv, "generation.fid_val": fid}
    rows = [(0, attrs(14.5, 70, 0.036)), (1, attrs(14.9, 80, 0.030)), (2, attrs(15.5, 78, 0.034)),
            (3, attrs(17.3, 83, 0.074))]
    assert [n for n, _ in tw.pareto(rows, ["generation.wasserstein", "inpaint.hv_best"])] == [0, 1, 3]
    assert [n for n, _ in tw.pareto(rows, ["generation.fid_val", "inpaint.hv_best"])] == [1, 3]


def test_study_continues_only_with_the_same_config(tmp_path):
    config = tw.load_config(CONFIG)
    study = tw.open_study("s", tmp_path, config["objectives"], create=True)
    assert [d.name.lower() for d in study.directions] == ["minimize", "maximize"]
    tw.check_or_record(study, config)
    tw.check_or_record(study, {**config, "n_trials": 99})
    with pytest.raises(SystemExit):
        tw.check_or_record(study, {**config, "seed": 11})
    optuna.delete_study(study_name="s", storage=study._storage)


def test_report_lists_the_front(tmp_path, capsys):
    config = tw.load_config(CONFIG)
    study = tw.open_study("r", tmp_path, config["objectives"], create=True)
    for k, (beta, w, hv) in enumerate(((0.0, 14.5, 76.0), (1.0, 15.0, 80.0), (2.0, 16.0, 79.0))):
        study.add_trial(optuna.trial.create_trial(
            params={"beta": beta, "w_min": 0.1, "w_max": 10.0},
            distributions={"beta": optuna.distributions.FloatDistribution(0, 3.5),
                           "w_min": optuna.distributions.FloatDistribution(0.1, 1.0, log=True),
                           "w_max": optuna.distributions.FloatDistribution(2, 10, log=True)},
            values=[w, hv], user_attrs={"generation.wasserstein": w, "inpaint.hv_best": hv}))
    tw.report(study, config["objectives"])
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith(("*", "  |")) or l.startswith(" ")]
    starred = [l for l in out.splitlines() if l.startswith("* ")]
    assert len(starred) == 2 and lines
    assert not math.isnan(float(starred[0].split("|")[5]))
