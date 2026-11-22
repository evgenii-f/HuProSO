"""The shared command line of the test protocols (hp/evaluation/protocol_cli.py) and scripts/evaluate.py completion,
with a fake model, poses and metrics."""
import argparse
import json
import sys

import pytest
import roma
import torch
import yaml

from hp.evaluation import protocol_cli as cli
from hp.utils.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def parse(*argv):
    parser = argparse.ArgumentParser()
    cli.add_model_arguments(parser)
    cli.add_common_arguments(parser, out="/tmp/out")
    return parser.parse_args(list(argv))


def test_settings_from_json_and_set_with_yaml_values():
    values = cli.method_settings('{"lr": 0.1, "schedule": "edm"}',
                                 ["lr=0.05", "replace=false", "steps=200", "x=[1, 2]"])
    assert values == {"lr": 0.05, "schedule": "edm", "replace": False, "steps": 200, "x": [1, 2]}
    with pytest.raises(cli.UsageError):
        cli.method_settings("[1]")
    with pytest.raises(cli.UsageError):
        cli.method_settings(None, ["lr"])


def test_the_settings_go_where_the_registry_keeps_them():
    dflow = cli.run_spec("~/r/run_0001", settings={"lr": 0.05, "steps": 200, "ode_steps": 25, "sampler": "heun"})
    assert dflow == {"run": "~/r/run_0001", "dflow": {"lr": 0.05, "steps": 200},
                     "prior": {"ode_steps": 25, "sampler": "heun"}}
    assert cli.run_spec("~/r") == {"run": "~/r"}  # their protocol: D-Flow with its defaults
    inpaint = cli.run_spec("~/r", "weights", "inpaint", {"steps": 200, "guidance": 0.06})
    assert inpaint == {"run": "~/r", "weights": "weights", "method": "inpaint",
                       "prior": {"steps": 200, "guidance": 0.06}}
    assert cli.run_spec("~/c/epoch-009.ckpt", "weights") == {"posedflow": "~/c/epoch-009.ckpt"}


def test_other_settings_never_share_a_default_label():
    path = "~/data/experiments/full/rsgm_s10/run_0001"
    plain = cli.default_label(path, cli.run_spec(path))
    assert plain == "full/rsgm_s10/run_0001"
    a = cli.default_label(path, cli.run_spec(path, method="inpaint", settings={"guidance": 0.1}))
    b = cli.default_label(path, cli.run_spec(path, method="inpaint", settings={"guidance": 0.2}))
    assert a.startswith("full/rsgm_s10/run_0001, inpaint, ") and a != b


def test_the_folders_of_the_splits(tmp_path):
    assert cli.split_dir(tmp_path, "test") == tmp_path
    assert cli.split_dir(tmp_path, "valid") == tmp_path / "valid"
    assert cli.split_dir(tmp_path, "test", 20) == tmp_path / "test_first20"
    assert cli.splits("all") == ["valid", "test"] and cli.splits("test") == ["test"]


def test_quick_is_a_prefix_of_the_validation():
    def resolved(*argv, quick=200):
        parser = argparse.ArgumentParser()
        cli.add_common_arguments(parser, out="/tmp/out", quick=quick)
        args = parser.parse_args(list(argv))
        cli.resolve_poses(args)
        return args.split, args.poses

    assert resolved() == ("test", None) and resolved("--split", "all") == ("all", None)
    assert resolved("--quick") == ("valid", 200) and resolved("--quick", "--split", "valid") == ("valid", 200)
    assert resolved("--quick", "--poses", "100") == ("valid", 100)
    with pytest.raises(cli.UsageError):
        resolved("--quick", "--split", "test")
    with pytest.raises(SystemExit):  # a protocol without a quick check has no --quick
        resolved("--quick", quick=None)


def test_models_of_the_run_registry_and_points(tmp_path):
    registry = {"A": {"run": "~/a"}, "B": {"run": "~/b", "method": "dposer"}}
    points = tmp_path / "points.yaml"
    points.write_text(yaml.safe_dump({"F": [{"label": "p1", "spec": {"run": "~/p"}, "validation": {"mean": 1}},
                                            {"label": "p2", "spec": {"run": "~/q"}}],
                                      "G": [{"label": "A", "spec": {"run": "~/a"}}]}))
    out = cli.models(parse("--run", "mine=~/m", "--set", "lr=0.05", "--models", "B"), registry)
    assert out == {"mine": {"run": "~/m", "dflow": {"lr": 0.05}}, "B": registry["B"]}
    assert list(cli.models(parse("--models", "all"), registry)) == ["A", "B"]
    assert list(cli.models(parse("--points", str(points)), registry)) == ["p1", "p2", "A"]
    assert list(cli.models(parse("--points", str(points), "--fronts", "F"), registry)) == ["p1", "p2"]
    assert list(cli.models(parse("--points", str(points), "--models", "p2"), registry)) == ["p2"]
    assert list(cli.models(parse("--points", str(points), "--models", "all"), registry)) == ["A", "B", "p1", "p2"]
    assert cli.recorded(str(points)) == {"p1": {"mean": 1}}
    for wrong in (parse("--models", "C"), parse("--set", "lr=1"), parse(), parse("--fronts", "F"),
                  parse("--run", "A=~/other")):
        with pytest.raises(cli.UsageError):
            cli.models(wrong, registry)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A run folder, fake poses of both splits, a model that returns the ground truth plus the split's offset, and
    metrics without SMPL-X."""
    import evaluate
    import hp.evaluation.completion as completion

    run = tmp_path / "run"
    run.mkdir()
    (run / "weights_ema.pt").write_bytes(b"x")
    torch.manual_seed(0)
    poses = {"test": roma.random_rotmat(6 * 21).view(6, 21, 3, 3),
             "valid": roma.random_rotmat(4 * 21).view(4, 21, 3, 3)}
    calls = []
    monkeypatch.setattr(evaluate, "poses_of", lambda split, args: poses[split][: args.poses])
    monkeypatch.setattr(completion, "build", lambda spec, device: None)

    def run_fake(model, gt, occlusion, k, **kw):
        calls.append((len(gt), occlusion, kw))
        return gt[:, None].repeat(1, k, 1, 1, 1)
    monkeypatch.setattr(completion, "run", run_fake)
    monkeypatch.setattr(completion, "Body", lambda: type("FakeBody", (), {"identity": {"model": 1}})())
    monkeypatch.setattr(completion, "metrics", lambda pred, gt, occlusion, body: {
        "mpvpe": torch.full(pred.shape[:2], float(len(gt))).numpy(), "mpjpe": torch.zeros(pred.shape[:2]).numpy(),
        "mgeo": torch.zeros(pred.shape[:2]).numpy(), "apd": torch.ones(len(gt)).numpy()})
    return evaluate, run, tmp_path / "out", calls


def test_evaluate_completion_runs_both_splits_into_their_folders(fake, capsys):
    evaluate, run, out, calls = fake
    argv = ["completion", "--run", f"mine={run}", "--method", "inpaint", "--set", "steps=3", "--split", "all",
            "--out", str(out), "--occlusions", "legs", "trunk", "--k", "2", "--device", "cpu"]
    assert evaluate.main(argv) == 0
    assert [(n, o) for n, o, _ in calls] == [(4, "legs"), (4, "trunk"), (6, "legs"), (6, "trunk")]
    assert calls[0][2]["prior"] == {"steps": 3} and calls[0][2]["method"] == "inpaint"
    assert (out / "valid" / "mine" / "legs.pt").is_file() and (out / "mine" / "trunk.pt").is_file()
    rows = [json.loads(line) for line in (out / "valid" / "results.jsonl").read_text().splitlines()]
    assert [(r["label"], r["occlusion"], r["split"], r["mpvpe_mean"]) for r in rows] == [
        ("mine", "legs", "valid", 4.0), ("mine", "trunk", "valid", 4.0)]
    assert "| mine | 6.00 | 6.00 | 6.00 | 6.00 | 1.00 |" in capsys.readouterr().out  # test: mean of 6 poses

    evaluate.main(argv)  # everything cached: nothing runs again, results.jsonl keeps one row per (model, occlusion)
    assert len(calls) == 4 and len((out / "valid" / "results.jsonl").read_text().splitlines()) == 2


def test_other_settings_are_kept_unless_recompute(fake, capsys):
    evaluate, run, out, calls = fake
    base = ["completion", "--run", f"mine={run}", "--split", "test", "--out", str(out), "--occlusions", "legs",
            "--k", "2", "--device", "cpu"]
    evaluate.main(base)
    evaluate.main(base + ["--set", "lr=0.05"])  # the same label, other settings: kept
    assert len(calls) == 1 and "kept, made with other settings" in capsys.readouterr().out
    evaluate.main(base + ["--dry-run", "--set", "lr=0.05"])
    assert "legs other" in capsys.readouterr().out and len(calls) == 1
    evaluate.main(base + ["--set", "lr=0.05", "--recompute"])
    assert len(calls) == 2 and calls[1][2]["lr"] == 0.05


def test_a_quick_check_goes_to_a_folder_of_its_own(fake):
    evaluate, run, out, calls = fake
    evaluate.main(["completion", "--run", f"mine={run}", "--poses", "2", "--out", str(out), "--occlusions", "arms",
                   "--k", "1", "--device", "cpu", "--no-metrics"])
    assert (out / "test_first2" / "mine" / "arms.pt").is_file() and not (out / "mine").exists()


def test_quick_runs_the_first_validation_poses(fake, capsys):
    evaluate, run, out, calls = fake
    evaluate.main(["completion", "--run", f"mine={run}", "--quick", "--poses", "3", "--out", str(out),
                   "--occlusions", "arms", "--k", "1", "--device", "cpu"])
    assert calls[0][0] == 3 and (out / "valid_first3" / "mine" / "arms.pt").is_file()
    assert "valid: 3 poses" in capsys.readouterr().out


def test_the_old_command_runs_the_registry_on_the_test(fake, monkeypatch):
    import evaluate_completion

    evaluate, run, out, calls = fake
    monkeypatch.setattr(evaluate_completion, "MODELS", {"R": {"run": str(run)}})
    evaluate_completion.main(["--cache", str(out), "--occlusions", "trunk", "--device", "cpu", "--together"])
    assert (out / "R" / "trunk.pt").is_file() and calls[0][2]["together"] is True
