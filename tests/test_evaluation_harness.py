import json

import numpy as np
import pytest
import roma
import torch
from test_training_smoke import rotations  # noqa: F401  (a fixture: random <Dataset>.pt files, and env settings)

from hp.data.amass import AMASS
from hp.evaluation.data import load_reference
from hp.evaluation.evaluate import EvalConfig, evaluate, evaluate_run, load_report, save_report
from hp.models import PoseModel, build_model, register_model, registry
from hp.models.io import save_model

JOINTS = 3
CONFIG = EvalConfig(n_wasserstein=200, n_queries=150, n_generated=300, n_likelihood=100, repeats=3, sample_batch=100)


def cloud(n, sigma_deg=5.0, shift_deg=0.0, seed=0):
    """Poses scattered by about sigma_deg around the rotation of shift_deg about z, the same in every joint."""
    g = torch.Generator().manual_seed(seed)
    noise = roma.rotvec_to_rotmat(torch.randn(n, JOINTS, 3, generator=g) * np.deg2rad(sigma_deg))
    return noise @ roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, np.deg2rad(shift_deg)], dtype=torch.float32))


class Around(PoseModel):
    """A generator with a known quality: poses scattered by sigma_deg around a rotation of shift_deg."""

    def __init__(self, num_joints=JOINTS, sigma_deg=5.0, shift_deg=0.0):
        super().__init__(num_joints)
        self.sigma_deg, self.shift_deg = sigma_deg, shift_deg
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def loss(self, pose, cond=None):
        return self.weight.sum()

    def sample(self, n, cond=None):
        center = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, np.deg2rad(self.shift_deg)], dtype=torch.float32))
        return roma.rotvec_to_rotmat(torch.randn(n, self.num_joints, 3) * np.deg2rad(self.sigma_deg)) @ center


REFERENCE = cloud(600, seed=0)


def mean(report, metric):
    return report[metric]["mean_deg"]["mean"]


def test_a_generator_like_the_data_is_as_good_as_the_data():
    r = evaluate(Around(), REFERENCE, CONFIG)
    assert r["wasserstein"]["ratio_to_floor"]["mean"] < 1.5
    assert mean(r, "recall") < 1.3 * mean(r, "baseline") and mean(r, "precision") < 1.3 * mean(r, "baseline")


def test_a_shifted_generator_is_far_from_the_data():
    r = evaluate(Around(shift_deg=30), REFERENCE, CONFIG)
    assert r["wasserstein"]["ratio_to_floor"]["mean"] > 3
    assert mean(r, "recall") > 3 * mean(r, "baseline") and mean(r, "precision") > 3 * mean(r, "baseline")


def test_a_collapsed_generator_keeps_its_precision_but_loses_recall():
    r = evaluate(Around(sigma_deg=0.0), REFERENCE, CONFIG)  # every sample is the same pose
    assert mean(r, "precision") < mean(r, "baseline")  # its poses are realistic
    assert mean(r, "recall") > 1.3 * mean(r, "baseline")  # but most real poses are not covered


def test_a_too_wide_generator_loses_precision():
    r = evaluate(Around(sigma_deg=15), REFERENCE, CONFIG)
    assert mean(r, "precision") > 2 * mean(r, "baseline") and r["wasserstein"]["ratio_to_floor"]["mean"] > 2


def test_the_report_is_reproducible_and_follows_the_seed():
    first, second = evaluate(Around(), REFERENCE, CONFIG), evaluate(Around(), REFERENCE, CONFIG)
    for r in (first, second):
        r["sampling"]["seconds_per_1000"] = 0  # the only number that depends on the machine
    assert first == second
    other = evaluate(Around(), REFERENCE, EvalConfig(**{**CONFIG.__dict__, "seed": 1}))
    assert mean(other, "recall") != mean(first, "recall")


def test_the_random_state_of_the_caller_is_left_alone():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    evaluate(Around(), REFERENCE, CONFIG)
    assert torch.equal(torch.rand(3), expected)


def test_the_model_is_in_eval_mode_while_sampling_and_its_mode_is_restored():
    seen = []

    class Spy(Around):
        def sample(self, n, cond=None):
            seen.append(self.training)
            return super().sample(n, cond)

    model = Spy()
    model.train()
    evaluate(model, REFERENCE, CONFIG)
    assert seen and not any(seen) and model.training
    model.eval()
    evaluate(model, REFERENCE, CONFIG)
    assert not model.training


def test_the_likelihood_exists_only_for_models_with_a_log_prob():
    assert evaluate(Around(), REFERENCE, CONFIG)["likelihood"] is None
    flow = build_model("so3_flow", num_joints=JOINTS, flow={"layers": 2})
    likelihood = evaluate(flow, REFERENCE, CONFIG)["likelihood"]
    assert np.isfinite(likelihood["mean"]) and likelihood["space"] == "SO(3)^J" and likelihood["nan_fraction"] == 0
    assert likelihood["n"] == 100


def test_sample_sizes_shrink_for_few_real_poses_and_bad_input_is_refused():
    small = evaluate(Around(), cloud(40), CONFIG)
    assert small["wasserstein"]["n"] == 20 and small["sizes"]["queries"] == 20
    with pytest.raises(ValueError, match="shape"):
        evaluate(Around(), torch.eye(3).expand(50, 4, 3, 3), CONFIG)
    with pytest.raises(ValueError, match="at least 4"):
        evaluate(Around(), cloud(3), CONFIG)

    class Broken(Around):
        def sample(self, n, cond=None):
            return super().sample(n, cond) * float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        evaluate(Broken(), REFERENCE, CONFIG)


def test_the_report_is_json_and_survives_saving(tmp_path):
    report = evaluate(Around(), REFERENCE, CONFIG)
    json.dumps(report)
    assert report["reference"] == {"poses": 600, "joints": JOINTS} and report["sampling"]["generated"] == 300
    assert report["config"]["seed"] == 0 and len(report["wasserstein"]["generated"]["by_joint_deg"]) == JOINTS
    path = save_report(report, tmp_path / "out" / "metrics.json")
    assert load_report(path) == report


def test_a_saved_run_is_evaluated_next_to_its_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    register_model("around")(Around)
    save_model(build_model("around", num_joints=JOINTS, sigma_deg=5.0), tmp_path / "run")
    report = evaluate_run(tmp_path / "run", REFERENCE, CONFIG)
    assert (tmp_path / "run" / "metrics.json").is_file() and report["model"]["name"] == "around"
    assert report["model"]["build_args"]["sigma_deg"] == 5.0 and report["run"]["weights"] == "weights"
    evaluate_run(tmp_path / "run", REFERENCE, CONFIG, output=tmp_path / "other.json")
    assert (tmp_path / "other.json").is_file()


def test_the_real_poses_of_a_split_come_from_the_rotation_files(rotations):  # noqa: F811
    poses = load_reference("test", rotations=str(rotations), amass_split="original")
    assert poses.shape == (300, 19, 3, 3) and poses.dtype == torch.float32  # 150 frames in each of the two test sets
    assert torch.equal(poses, AMASS(mode="test").data.reshape(-1, 19, 3, 3).float())


def test_real_data_of_other_subjects_is_the_yardstick():
    """Other subjects are not as close to the data as the data itself, so a model is judged against them."""
    config = CONFIG
    far = evaluate(Around(shift_deg=30), REFERENCE, config, contrast=cloud(600, shift_deg=30, seed=5))
    plain = evaluate(Around(shift_deg=30), REFERENCE, config)
    assert "contrast" not in plain["wasserstein"] and "contrast_recall" not in plain
    w = far["wasserstein"]
    assert w["contrast"]["joint_mean_deg"]["mean"] > 3 * w["floor"]["joint_mean_deg"]["mean"]  # unseen subjects are far
    # a model that matches the other subjects is as close to the data as those subjects are
    assert w["generated"]["joint_mean_deg"]["mean"] == pytest.approx(w["contrast"]["joint_mean_deg"]["mean"], rel=0.25)
    assert mean(far, "recall") == pytest.approx(mean(far, "contrast_recall"), rel=0.25)
    assert mean(far, "precision") == pytest.approx(mean(far, "contrast_precision"), rel=0.25)
    assert far["sizes"]["contrast_candidates"] == 300


def test_the_report_with_a_contrast_set_is_reproducible_and_checks_its_input():
    contrast = cloud(300, shift_deg=10, seed=5)
    first, second = evaluate(Around(), REFERENCE, CONFIG, contrast), evaluate(Around(), REFERENCE, CONFIG, contrast)
    for r in (first, second):
        r["sampling"]["seconds_per_1000"] = 0
    assert first == second
    with pytest.raises(ValueError, match="contrast"):
        evaluate(Around(), REFERENCE, CONFIG, contrast=torch.eye(3).expand(50, 4, 3, 3))


def test_the_report_explains_itself_and_the_contrast_only_when_there_is_one():
    plain = evaluate(Around(), REFERENCE, CONFIG)
    assert set(plain["notes"]) == {"units", "wasserstein", "floor_and_baseline"}
    with_contrast = evaluate(Around(), REFERENCE, CONFIG, contrast=cloud(300, seed=5))
    assert "no goals" in with_contrast["notes"]["floor_and_baseline"] and "Compare the model with this" in with_contrast["notes"]["contrast"]
