"""scripts/evaluate.py denoising and the points of the test (scripts/denoising_test_points.py), with fake sequences, a
fake body and a fake method: no SMPL-X and no models are needed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import roma
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A run folder, 6 test and 4 validation sequences of 3 frames, and a method that returns the ground truth; the
    MPJPE of a run is the number of its sequences."""
    import evaluate
    import hp.evaluation.denoising as D

    run = tmp_path / "run"
    run.mkdir()
    (run / "weights_ema.pt").write_bytes(b"x")
    torch.manual_seed(0)
    data = {"test": roma.random_rotmat(6 * 3 * 21).view(6, 3, 21, 3, 3),
            "valid": roma.random_rotmat(4 * 3 * 21).view(4, 3, 21, 3, 3)}
    calls = []
    monkeypatch.setattr(D, "sequences", lambda split, file=None, n_valid=100: (
        data[split], [f"{split}/{i}" for i in range(len(data[split]))]))
    monkeypatch.setattr(D, "Body", lambda device=None: None)
    monkeypatch.setattr(D, "gt_joints", lambda gt, body: torch.zeros(*gt.shape[:2], 22, 3))
    monkeypatch.setattr(D, "noisy_joints", lambda joints, sigma: joints)
    monkeypatch.setattr(D, "build", lambda spec, device: None)

    def run_fake(model, gt, noisy, sigma, body, method, batch, prior=None, **kw):
        calls.append((len(gt), sigma, method, prior, {k: v for k, v in kw.items() if k not in ("partial", "key",
                                                                                              "stop")}))
        return gt
    monkeypatch.setattr(D, "run", run_fake)
    monkeypatch.setattr(D, "metrics", lambda pred, gt, noisy, body: {
        k: np.full(gt.shape[:2], float(len(gt))) for k in ("mpjpe", "mpvpe", "input_mpjpe")})
    return evaluate, run, tmp_path / "out", calls


def test_denoising_runs_both_splits_and_the_noise_levels(fake, capsys):
    evaluate, run, out, calls = fake
    argv = ["denoising", "--run", f"mine={run}", "--set", "t_max=0.05", "steps_per_iter=2", "--split", "all",
            "--out", str(out), "--device", "cpu"]
    assert evaluate.main(argv) == 0
    assert [(n, s, m) for n, s, m, _, _ in calls] == [(4, 0.04, "dposer"), (4, 0.1, "dposer"), (6, 0.04, "dposer"),
                                                      (6, 0.1, "dposer")]
    assert calls[0][4] == {"t_max": 0.05, "steps_per_iter": 2}
    assert (out / "mine" / "valid_40mm.pt").is_file() and (out / "mine" / "test_100mm.pt").is_file()
    assert "| mine | 6.00 / 6.00 | 6.00 / 6.00 |" in capsys.readouterr().out
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [(r["label"], r["split"], r["sigma"]) for r in rows] == [("mine", "valid", 0.04), ("mine", "valid", 0.1),
                                                                     ("mine", "test", 0.04), ("mine", "test", 0.1)]
    evaluate.main(argv)  # cached: nothing runs again
    assert len(calls) == 4


def test_dflow_settings_and_prior_options_go_apart(fake):
    evaluate, run, out, calls = fake
    evaluate.main(["denoising", "--run", f"mine={run}", "--method", "dflow", "--set", "lr=0.2", "ode_steps=50",
                   "--sigmas", "0.04", "--out", str(out), "--device", "cpu"])
    _, _, method, prior, settings = calls[0]
    assert method == "dflow" and prior == {"ode_steps": 50} and settings == {"lr": 0.2}


def test_other_settings_are_kept_unless_recompute(fake, capsys):
    evaluate, run, out, calls = fake
    base = ["denoising", "--run", f"mine={run}", "--sigmas", "0.1", "--out", str(out), "--device", "cpu"]
    evaluate.main(base + ["--set", "lr=0.03"])
    label = ["denoising", "--run", f"mine={run}", "--set", "lr=0.05", "--sigmas", "0.1", "--out", str(out),
             "--device", "cpu"]
    evaluate.main(label)  # the same label, other settings: kept
    assert len(calls) == 1 and "kept, made with other settings" in capsys.readouterr().out
    evaluate.main(label + ["--dry-run"])
    assert "100 mm other" in capsys.readouterr().out
    evaluate.main(label + ["--recompute"])
    assert len(calls) == 2 and calls[1][4] == {"lr": 0.05}
    evaluate.main(label + ["--force"])
    assert len(calls) == 3


def test_quick_runs_the_first_validation_sequences_apart(fake, capsys):
    evaluate, run, out, calls = fake
    evaluate.main(["denoising", "--run", f"mine={run}", "--quick", "--poses", "3", "--sigmas", "0.04",
                   "--out", str(out), "--device", "cpu"])
    assert calls[0][0] == 3 and (out / "valid_first3" / "mine" / "valid_40mm.pt").is_file()
    assert not (out / "mine").exists() and "valid: 3 sequences" in capsys.readouterr().out


def test_a_posedflow_checkpoint_gets_dflow_and_dposer_rejects_nothing():
    import evaluate

    assert evaluate.denoising_spec("x.ckpt") == {"posedflow": "x.ckpt"}
    assert evaluate.denoising_spec("run", settings={"t_max": 0.1}) == {"run": "run", "method": "dposer",
                                                                     "settings": {"t_max": 0.1}}
    with pytest.raises(Exception):
        evaluate.denoising_spec("run", method="inpaint")


def test_the_points_are_the_rows_of_the_test_with_their_specs():
    """Every point keeps the label and the spec of its row of evaluate_denoising.MODELS (so its test is cached), and
    every row with a study names a trial and validation numbers at both noise levels."""
    from evaluate_denoising import MODELS

    from hp.evaluation.denoising import SIGMAS

    groups = yaml.safe_load((ROOT / "config" / "denoising_test_points.yaml").read_text())
    labels = [p["label"] for points in groups.values() for p in points]
    assert len(labels) == len(set(labels)) == 13 and "RSGM 1536x6: their c10, DPoser method" in labels
    for points in groups.values():
        for p in points:
            assert json.dumps(p["spec"], sort_keys=True) == json.dumps(MODELS[p["label"]], sort_keys=True)
            if isinstance(p["source"], dict):
                assert set(p["source"]) == set(SIGMAS) and set(p["validation"]) == set(SIGMAS)
    assert not any("first guess" in label or " lik" in label or " ln " in label for label in labels)


@pytest.mark.skipif(not Path("~/data/experiments/denoising/optuna/rsgm_sds_budget_100mm.db").expanduser().is_file(),
                    reason="needs the studies")
def test_the_points_file_is_what_the_script_writes():
    import denoising_test_points

    assert denoising_test_points.main(["--check"]) == 0


def test_like_takes_the_settings_per_noise_level_and_set_applies_on_top(fake):
    from evaluate_denoising import MODELS, SDS

    evaluate, run, out, calls = fake
    evaluate.main(["denoising", "--run", f"mine={run}", "--like", "ours", "--set", "lr=0.01", "--out", str(out),
                   "--device", "cpu"])
    assert [c[4] for c in calls] == [{**SDS[0.04], "lr": 0.01}, {**SDS[0.1], "lr": 0.01}]
    assert MODELS[evaluate.DENOISING_LIKE["ours"]]["settings"] == SDS


def test_like_a_dflow_row_and_its_ready_names():
    import argparse

    import evaluate
    from evaluate_denoising import MODELS

    from hp.evaluation import protocol_cli as cli

    assert all(label in MODELS for label in evaluate.DENOISING_LIKE.values())
    args = argparse.Namespace(run=["x=r"], weights=None, models=[], points=None, fronts=None, method=None,
                              assignments=["ode_steps=50"], settings=None, like=evaluate.DENOISING_LIKE["posedflow"])
    spec = cli.models(args, MODELS, default_method="dposer", spec_of=evaluate.denoising_spec,
                      settings_of=evaluate.denoising_settings_of)["x"]
    assert spec == {"run": "r", "prior": {"ode_steps": 50}}  # D-Flow (no method), the ODE option to the prior
    args.like, args.method, args.assignments = evaluate.DENOISING_LIKE["ours"], "dflow", []
    with pytest.raises(cli.UsageError):  # ours is dposer
        cli.models(args, MODELS, default_method="dposer", spec_of=evaluate.denoising_spec,
                   settings_of=evaluate.denoising_settings_of)
