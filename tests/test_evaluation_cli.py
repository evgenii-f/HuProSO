import dataclasses

import pytest
from omegaconf import OmegaConf
from test_training_smoke import rotations  # noqa: F401  (a fixture: random <Dataset>.pt files, and env settings)

from hp.evaluation.cli import format_report, main
from hp.evaluation.evaluate import EvalConfig, load_report
from hp.training.cli import main as train
from hp.utils.paths import REPO_ROOT


def test_evaluate_yaml_matches_the_eval_config_defaults():
    eval_section = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "config" / "evaluate.yaml").eval)
    assert eval_section == dataclasses.asdict(EvalConfig())


def small_eval(rotations, *extra):  # noqa: F811
    return [f"data.rotations={rotations}", "data.split=original", "device=cpu", "eval.n_wasserstein=40", "eval.n_queries=40",
            "eval.n_generated=80", "eval.n_likelihood=40", "eval.repeats=2", "eval.sample_batch=40", *extra]


@pytest.fixture
def run(rotations, tmp_path):  # noqa: F811
    return train(["model.args.flow.layers=2", "train.max_iterations=2", "train.batch_size=64", "train.valid_interval=2",
                  "train.checkpoint_every=0", "train.device=cpu", "train.progress=false", "tensorboard=false",
                  f"run_root={tmp_path / 'runs'}", f"data.rotations={rotations}", "data.split=original"])


def test_the_script_evaluates_a_trained_run(run, rotations, capsys):  # noqa: F811
    report = main([f"run={run}", "split=test", *small_eval(rotations)])
    assert (run / "metrics_test.json").is_file() and load_report(run / "metrics_test.json") == report
    assert report["model"]["name"] == "so3_flow" and report["reference"]["poses"] == 300  # the test split: 2 x 150 poses
    assert report["run"]["weights"] == "weights" and report["likelihood"] is not None
    out = capsys.readouterr().out
    assert "wasserstein" in out and "recall" in out and "log-likelihood" in out and "metrics_test.json" in out


def test_the_contrast_is_the_train_split_unless_switched_off(run, rotations, capsys):  # noqa: F811
    report = main([f"run={run}", *small_eval(rotations)])
    assert "contrast" in report["wasserstein"] and report["sizes"]["contrast_candidates"] == 80
    out = capsys.readouterr().out
    assert "contrast" in out and "real poses of other subjects" in out and "no goal" in out
    plain = main([f"run={run}", "contrast=null", *small_eval(rotations)])
    assert "contrast" not in plain["wasserstein"] and "contrast_recall" not in plain


def test_reports_of_two_splits_do_not_overwrite_each_other(run, rotations):  # noqa: F811
    main([f"run={run}", "split=test", *small_eval(rotations)])
    main([f"run={run}", "split=valid", *small_eval(rotations)])
    assert (run / "metrics_test.json").is_file() and (run / "metrics_valid.json").is_file()
    assert load_report(run / "metrics_valid.json")["reference"]["poses"] == 600  # 4 validation sets of 150 poses


def test_an_output_path_and_the_run_are_required(run, rotations, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="Which run"):
        main(["split=test"])
    main([f"run={run}", f"output={tmp_path / 'custom.json'}", *small_eval(rotations)])
    assert (tmp_path / "custom.json").is_file()


def test_the_summary_names_the_numbers_and_the_missing_density(run, rotations):  # noqa: F811
    report = main([f"run={run}", *small_eval(rotations)])
    text = format_report(report)
    assert "so3_flow" in text and "same-set floor" in text and "SO(3)^J" in text
    report["likelihood"] = None
    assert "no exact density" in format_report(report)
