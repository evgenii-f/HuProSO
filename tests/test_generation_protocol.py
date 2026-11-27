"""scripts/evaluate.py generation (hp.evaluation.generation), with a fake model, small random poses and a fake skeleton:
no data and no SMPL are needed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import roma
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hp.evaluation import generation as G  # noqa: E402
from hp.evaluation import pareto as P  # noqa: E402
from hp.evaluation import protocol_cli as cli  # noqa: E402
from hp.evaluation.evaluate import EvalConfig  # noqa: E402
from hp.models.base import PoseModel  # noqa: E402


class Skeleton:
    def __init__(self, path=None):
        pass

    def joints(self, poses):
        return poses.double().cpu()[:, :, :, 0].repeat(1, 2, 1)[:, :22]


class Uniform(PoseModel):
    """Uniform rotations with a constant log-density, and one option of the sampler."""
    name = "uniform"
    has_log_prob = True
    density_space = "SO(3)^19"
    sample_steps = 100

    def __init__(self):
        super().__init__(num_joints=19)
        self.build_args = {}
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def sample(self, n, cond=None):
        return roma.random_rotmat(n * 19).view(n, 19, 3, 3)

    def log_prob(self, pose, cond=None):
        return torch.full((len(pose),), 2.5)

    def loss(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A run folder, 60 / 50 / 400 test / valid / train poses, small sizes, and the number of models built."""
    run = tmp_path / "run"
    run.mkdir()
    (run / P.MANIFEST).write_text("{}")
    (run / "weights_ema.pt").write_bytes(b"x")
    torch.manual_seed(0)
    data = {name: roma.random_rotmat(n * 19).view(n, 19, 3, 3) for name, n in
            (("test", 60), ("valid", 50), ("train", 400))}
    built = []

    def load_model(path, device, weights):
        built.append(Uniform())
        return built[-1]
    small = dict(n_wasserstein=20, n_queries=10, n_generated=40, repeats=2, sample_batch=20)
    for name, value in dict(load_model=load_model, SmplSkeleton=Skeleton, SETS=3, SET_SIZE=20, N_CONTRAST=50,
                            HARNESS=EvalConfig(n_likelihood=10, **small)).items():
        monkeypatch.setattr(P, name, value)
    monkeypatch.setattr(P.Protocol.__init__, "__defaults__", (3, 20, P.SKELETON, P.GT_STATS))
    monkeypatch.setattr(P.contrast_poses, "__defaults__", (50,))
    monkeypatch.setattr(P.harness_settings, "__defaults__", (P.HARNESS, 50))
    monkeypatch.setattr(P.paper_settings, "__defaults__", (3, 20))
    monkeypatch.setattr(G, "VALID_HARNESS", EvalConfig(n_likelihood=0, **small))
    monkeypatch.setattr(G, "VALID_LIKELIHOOD", EvalConfig(n_likelihood=10, **small))
    monkeypatch.setattr(G.Arena.__init__, "__defaults__",
                        (P.SKELETON, P.GT_STATS, lambda split, rotations, amass: data[split]))
    stats = tmp_path / "stats.npz"
    np.savez(stats, muj=np.zeros(66), covj=np.eye(66))
    import evaluate
    return evaluate, run, tmp_path / "out", built, stats, data


def argv(run, out, *more):
    return ["generation", "--run", f"mine={run}", "--out", str(out), "--device", "cpu", *more]


def test_quick_is_the_validation_without_the_likelihood(fake, capsys):
    evaluate, run, out, built, _, _ = fake
    assert evaluate.main(argv(run, out, "--quick")) == 0
    stages = {p.stem for p in (out / "valid" / "mine").iterdir()}
    assert stages == {"generation", "harness"} and not (out / "mine").exists()
    quick = json.loads((out / "valid" / "results.jsonl").read_text())
    assert "FID_val" in quick and "FID" not in quick and "log_likelihood" not in quick and quick["poses"] == 50
    n = len(built)
    evaluate.main(argv(run, out, "--split", "valid"))  # only the likelihood is missing
    full = json.loads((out / "valid" / "results.jsonl").read_text())
    assert len(built) == n + 1 and full["log_likelihood"] == 2.5
    assert {k: full[k] for k in quick} == quick
    printed = capsys.readouterr().out
    assert "| mine |" in printed and "| real train poses |" in printed and "| real valid poses |" in printed


def test_the_test_has_the_stages_settings_and_files_of_the_pareto_comparison(fake):
    evaluate, run, out, built, stats, data = fake
    evaluate.main(argv(run, out, "--split", "test", "--gt-stats", str(stats)))
    assert {p.name for p in (out / "mine").iterdir()} == {"harness.json", "paper.json"}
    row = json.loads((out / "results.jsonl").read_text())
    assert row["split"] == "test" and "FID" in row and "FID_val" not in row and row["log_likelihood"] == 2.5
    n = len(built)
    zoo = P.Zoo({"other name": dict(short="x", run=str(run))}, "cpu")
    protocol = P.Protocol(data["train"], "cpu", gt_stats=str(stats))
    P.evaluate_quality(zoo, data["test"], data["train"], harness=P.HARNESS, n_contrast=50, protocol=protocol, cache=out)
    assert len(built) == n  # the comparison finds both stages and the two rows of real poses


def test_fid_val_is_the_fid_against_the_validation_poses(fake):
    _, _, _, _, _, data = fake
    protocol = G.Arena("valid", "cpu").protocol
    joints = Skeleton().joints(data["valid"]).reshape(50, -1).numpy()
    mu, cov = protocol.reference(Skeleton())
    assert np.allclose(mu, joints.mean(0)) and np.allclose(cov, np.cov(joints, rowvar=False))
    out = protocol(lambda k: data["valid"])  # the validation poses themselves
    assert set(out) == {"FID_val", "FID own", "APD", "dNN"} and abs(out["FID_val"]["mean"]) < 1e-3
    assert out["FID own"]["mean"] > 10 * abs(out["FID_val"]["mean"])


def test_other_settings_are_kept_unless_recompute(fake, capsys):
    evaluate, run, out, built, _, _ = fake
    evaluate.main(argv(run, out, "--quick"))
    n = len(built)
    other = argv(run, out, "--quick", "--set", "sample_steps=7")
    evaluate.main(other)
    assert len(built) == n and "kept, made with other settings" in capsys.readouterr().out
    evaluate.main(other + ["--dry-run"])
    assert "generation other, harness other" in capsys.readouterr().out
    evaluate.main(other + ["--recompute"])
    assert len(built) == n + 2 and built[-1].sample_steps == 7
    evaluate.main(other)
    assert len(built) == n + 2
    evaluate.main(other + ["--force"])
    assert len(built) == n + 4


def test_the_settings_are_part_of_the_default_label_and_of_the_model(fake):
    evaluate, run, out, built, _, _ = fake
    base = ["generation", "--run", str(run), "--out", str(out), "--device", "cpu", "--quick"]
    evaluate.main(base)
    evaluate.main(base + ["--set", "sample_steps=7"])
    labels = [json.loads(line)["label"] for line in (out / "valid" / "results.jsonl").read_text().splitlines()]
    assert len(set(labels)) == 2


def test_wrong_arguments_are_usage_errors(fake):
    evaluate, run, out, _, _, _ = fake
    for more in (["--set", "guidance=1"], ["--method", "dposer"], ["--points", "x.yaml"], ["--rotations", "x"],
                 ["--quick", "--split", "test"], ["--poses", "5"], ["--like", "no such row"]):
        with pytest.raises(SystemExit):
            evaluate.main(argv(run, out, *more))
    with pytest.raises(SystemExit):
        evaluate.main(["generation", "--run", "x.ckpt", "--out", str(out), "--device", "cpu"])


def test_an_option_the_model_does_not_have_is_an_error(fake):
    evaluate, run, out, _, _, _ = fake
    with pytest.raises(ValueError, match="no option 'schedule'"):  # it would change nothing
        evaluate.main(argv(run, out, "--quick", "--set", "schedule=edm"))


def test_like_takes_the_sampler_of_a_row_and_set_applies_on_top(fake):
    evaluate, run, _, _, _, _ = fake
    parse = lambda like, *sets: cli.models(  # noqa: E731
        __import__("argparse").Namespace(run=[f"x={run}"], weights=None, models=[], points=None, fronts=None,
                                         method=None, assignments=list(sets), settings=None, like=like),
        P.MODELS, default_method="sample", spec_of=evaluate.generation_spec,
        settings_of=evaluate.generation_settings_of)["x"]
    heun = evaluate.GENERATION_LIKE["heun"]
    assert parse(heun)["settings"] == dict(sampler="heun", schedule="edm", sample_steps=25)
    assert parse(heun, "sample_steps=50")["settings"]["sample_steps"] == 50
    assert parse(None) == {"run": str(run)}
    with pytest.raises(cli.UsageError):
        parse("PoseD-Flow: their checkpoint, midpoint 100 (paper)")
    assert all(v is None or v in P.MODELS for v in evaluate.GENERATION_LIKE.values())


def test_a_soft_stop_finishes_the_stage_and_then_stops():
    soft = cli.SoftStop("stage")
    assert not soft.stop()
    soft()
    assert soft.stop()
    with pytest.raises(KeyboardInterrupt):
        soft()
