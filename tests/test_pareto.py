import json
import subprocess
import sys

import pytest
import torch

from hp.evaluation import pareto as P
from hp.utils.paths import REPO_ROOT


class Timed(torch.nn.Module):
    """A model that only counts its calls of sample(n)."""

    sample_steps = 7

    def __init__(self, calls):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3))
        self.calls = calls

    def num_parameters(self):
        return 3

    def sample(self, n, cond=None):
        self.calls.append(n)
        return torch.zeros(n, 19, 3, 3)


@pytest.fixture
def calls():
    return []


@pytest.fixture
def timed(calls, tmp_path):
    """cached_runtime of the counting model on cpu, in a cache under tmp_path."""
    settings = {"model": {"kind": "run", "run": "/home/someone/runs/a", "mtime": 1.5}, **P.runtime_settings("cpu")}

    def run(batches=(8, 1), repeats=2, force=False, settings=settings, label="model"):
        return P.cached_runtime(label, "runtime_test", settings, lambda: Timed(calls), batches, repeats, force,
                                device="cpu", cache=tmp_path, log=lambda message: None)
    return run


def stored(tmp_path, label="model", stage="runtime_test"):
    return json.loads((tmp_path / P.slug(label) / f"{stage}.json").read_text())


# ---- the choice of models -----------------------------------------------------------------------------------------
MODELS = {"RSGM 512x4: their c10": dict(short="RSGM 512x4", run="a"),
          "RSGM 512x4: their full": dict(short="RSGM 512x4 full", run="b"),
          "DPoser: their full, 100 steps": dict(short="DPoser full, 100", run="c")}


def test_select_by_name_short_label_or_part_in_the_order_of_models():
    assert list(P.select(None, MODELS)) == list(MODELS)
    assert list(P.select(["RSGM 512x4"], MODELS)) == ["RSGM 512x4: their c10"]  # a short label: only that model
    assert list(P.select(["dposer", "512x4"], MODELS)) == list(MODELS)[:2] + ["DPoser: their full, 100 steps"]
    with pytest.raises(KeyError):
        P.select(["NRDF"], MODELS)


def test_a_run_that_is_not_here_is_skipped(tmp_path):
    zoo = P.Zoo({"x": dict(short="x", run=str(tmp_path / "missing"))}, "cpu", log=lambda message: None)
    assert zoo.builders == {} and zoo.groups == {}


# ---- the cache ----------------------------------------------------------------------------------------------------
def test_settings_compare_across_home_folders_and_in_whole_seconds():
    here = {"run": "/Users/me/data/x", "mtime": 1700000000.25}
    assert P.portable(here) == P.portable({"run": "/home/evgenii/data/x", "mtime": 1700000000.9})
    assert P.portable(here) != P.portable({"run": "/Users/me/data/y", "mtime": 1700000000.25})


def test_cached_computes_once_and_again_for_other_settings_or_when_forced(tmp_path):
    computed = []

    def get(settings, force=False, label="a"):
        return P.cached(label, "paper", settings, lambda: computed.append(1) or len(computed), force, tmp_path,
                        log=lambda message: None)
    assert get({"run": "/Users/me/x"}) == 1
    assert get({"run": "/home/me/x"}) == 1  # the same settings on another machine
    assert get({"run": "/Users/me/y"}) == 2
    assert get({"run": "/Users/me/y"}, force=True) == 3
    assert get({"run": "/Users/me/y"}, label="b") == 3  # the same evaluation under another name: reused
    assert len(computed) == 3


# ---- the times ----------------------------------------------------------------------------------------------------
def test_runtime_times_each_n_with_a_warm_up_and_keeps_it(timed, calls, tmp_path):
    result = timed()
    assert calls == [8, 8, 8, 1, 1, 1]  # warm-up and 2 repeats per n
    assert list(result["batches"]) == ["8", "1"] and result["parameters"] == 3 and result["sample_steps"] == 7
    entry = result["batches"]["8"]
    assert entry["repeats"] == 2 and len(entry["all_seconds"]) == 2 and "date" in entry["conditions"]
    assert entry["us_per_pose"] == pytest.approx(entry["seconds"] / 8 * 1e6)
    assert "batches" not in stored(tmp_path)["settings"] and "repeats" not in stored(tmp_path)["settings"]
    calls.clear()
    assert timed() == result and calls == []


def test_runtime_adds_new_n_and_times_again_only_what_is_asked(timed, calls):
    first = timed(batches=(8, 1))
    calls.clear()
    result = timed(batches=(64,))
    assert calls == [64, 64, 64]
    assert list(result["batches"]) == ["64", "8", "1"] and result["batches"]["8"] == first["batches"]["8"]
    calls.clear()
    timed(batches=(8,), repeats=3)  # other repeats: that n again
    assert calls == [8] * 4
    calls.clear()
    result = timed(batches=(1,), force=True)
    assert calls == [1] * 3 and set(result["batches"]) == {"64", "8", "1"}


def test_runtime_of_another_torch_is_set_aside(timed, calls, tmp_path):
    timed(batches=(8,))
    path = tmp_path / "model" / "runtime_test.json"
    saved = json.loads(path.read_text())
    saved["settings"]["torch"] = "2.0.0"
    path.write_text(json.dumps(saved))
    calls.clear()
    timed(batches=(1,))
    aside = tmp_path / "model" / "runtime_test.torch_2_0_0.json"
    assert json.loads(aside.read_text()) == saved  # kept as it was
    assert list(stored(tmp_path)["result"]["batches"]) == ["1"] and calls == [1, 1, 1]
    timed(batches=(1,), settings={**stored(tmp_path)["settings"], "torch": "2.0.0"})  # back: this one is set aside too
    assert (tmp_path / "model" / f"runtime_test.{P.slug('torch ' + torch.__version__)}.json").is_file()


def test_runtime_of_the_first_format_is_read_without_timing(timed, calls, tmp_path):
    settings = {"model": {"kind": "run", "run": "/home/someone/runs/a", "mtime": 1.5}, **P.runtime_settings("cpu")}
    old = {"name": "model", "stage": "runtime_test", "settings": {**settings, "batches": [8, 1], "repeats": 2},
           "result": {"parameters": 3, "sample_steps": 7, "batches": {
               "8": {"seconds": 0.8, "us_per_pose": 1e5, "all_seconds": [0.8, 0.8]},
               "1": {"seconds": 0.1, "us_per_pose": 1e5, "all_seconds": [0.1, 0.1]}}}}
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "runtime_test.json").write_text(json.dumps(old))
    result = timed(batches=(8, 1), repeats=2)
    assert calls == [] and result["batches"]["8"]["seconds"] == 0.8 and result["batches"]["1"]["repeats"] == 2
    assert "batches" not in stored(tmp_path)["settings"]  # rewritten in the current format


# ---- the script ---------------------------------------------------------------------------------------------------
def test_the_script_refuses_an_unknown_model():
    out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "evaluate_pareto.py"), "--models", "no such model",
                          "--list"], capture_output=True, text=True)
    assert out.returncode != 0 and "no model matches 'no such model'" in out.stderr
