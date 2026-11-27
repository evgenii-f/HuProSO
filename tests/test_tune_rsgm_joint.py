"""The fixed parts of scripts/tune_rsgm_joint.py: the budget, the settings of a trial, the choice."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tune_rsgm_joint.py"


@pytest.fixture(scope="module")
def tune():
    spec = importlib.util.spec_from_file_location("tune_rsgm_joint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_poses_for_every_batch(tune):
    assert tune.steps_of(4096) == 30_000
    assert tune.steps_of(2048) == 60_000
    assert tune.steps_of(8192) == 15_000
    assert {b * tune.steps_of(b) for b in tune.SPACE["batch_size"]} == {tune.POSES}


def test_overrides_of_the_reference(tune):
    o = tune.overrides(tune.REFERENCE)
    assert "model.args.net_width=1024" in o and "model.args.net_blocks=2" in o and "model.args.dropout=0.1" in o
    assert "train.max_iterations=30000" in o and "train.keep_best=1" in o
    assert not any(x.startswith("train.max_iterations=200") for x in o)
    assert "train.max_iterations=200" in tune.overrides(tune.REFERENCE, smoke=True)


def test_reference_is_in_the_space(tune):
    s, r = tune.SPACE, tune.REFERENCE
    for key in ("net_width", "dropout", "batch_size"):
        assert r[key] in s[key]
    for key in ("net_blocks", "lr", "weight_decay"):
        assert s[key][0] <= r[key] <= s[key][1]


def test_choice(tune):
    trials = [
        dict(number=0, mean=90.0, w=15.2, apd=6.5),  # the reference
        dict(number=1, mean=86.0, w=15.1, apd=6.0),  # on the front, W below the reference: chosen first
        dict(number=2, mean=80.0, w=15.0, apd=4.5),  # best on both, but infeasible (APD <= 5)
        dict(number=3, mean=85.0, w=15.4, apd=6.2),  # on the front, but W above the reference
        dict(number=4, mean=87.0, w=15.15, apd=6.1),  # dominated by trial 1
        dict(number=5, mean=88.0, w=15.05, apd=5.5),  # on the front, W below the reference
    ]
    chosen = [t["number"] for t in tune.choose(trials, reference_w=15.2)]
    assert chosen == [1, 5]
