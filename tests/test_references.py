"""scripts/evaluate.py all and references (hp.evaluation.references): the sets, the comparison and the files of the
references, and the runs of all with fake protocols. No data and no models are needed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate  # noqa: E402
from hp.evaluation import protocol_cli as cli  # noqa: E402
from hp.evaluation import references as R  # noqa: E402

RNG = np.random.default_rng(0)
OCCLUSIONS = ("left_leg", "legs", "arms", "trunk")


def completion(network="x", architecture="MLP 512x4", shift=0.0, n=1000, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(99).normal(90, 20, (n, 10))
    parts = {o: {"mpvpe": base + shift + rng.normal(0, noise, (n, 10)), "apd": np.full(n, 6.5)} for o in OCCLUSIONS}
    return R.completion_entry(parts, group="inpaint", network=network, architecture=architecture, label=network, spec={})


def run_folder(tmp_path, name="run_0001", **build_args):
    folder = tmp_path / name
    folder.mkdir(parents=True)
    (folder / "model.json").write_text(json.dumps({"build_args": {"name": "rsgm", **build_args}}))
    return folder


def test_sets_follow_the_rule():
    config = R.load()
    assert len(config["sets"]) == 7
    for row in config["sets"]:
        assert row["method"] in R.METHODS
        assert row["validation"]["apd"] >= config["apd_floor"] - 0.005  # D-Flow of small: 6.398, the floor itself
        assert R.front_mean(config["front"], row["validation"]["apd"]) <= row["validation"]["mean"] + 0.01
    assert R.set_for(config, "inpaint", "MLP 512x4")[1]
    row, own = R.set_for(config, "dflow", "timeres 256x8")
    assert not own and row["architecture"] == "timeres 1024x2"


def test_architecture():
    assert R.architecture({"name": "rsgm", "net": "timeres", "net_width": 256, "net_blocks": 8}) == "timeres 256x8"
    assert R.architecture({"name": "rsgm", "hidden": [512] * 4}) == "MLP 512x4"
    assert R.architecture({"name": "rsgm", "hidden": [256, 512]}) == "MLP 256-512"
    with pytest.raises(ValueError):
        R.architecture({"name": "dposer"})


def test_resolve_run(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    run = run_folder(tmp_path / "a")
    assert R.resolve_run(str(run)) == R.resolve_run(str(tmp_path / "a")) == "~/a/run_0001"
    run_folder(tmp_path / "a", "run_0002")
    with pytest.raises(ValueError, match="name one"):
        R.resolve_run(str(tmp_path / "a"))


def test_front_mean():
    front = [{"apd": 6.0, "mean": 80.0}, {"apd": 7.0, "mean": 90.0}]
    assert R.front_mean(front, 6.5) == 85.0
    assert R.front_mean(front, 5.0) == 80.0  # below: the lowest mean
    assert R.front_mean(front, 8.0) is None


def test_difference_marks_noise():
    model, close, far = completion(seed=1), completion(seed=2), completion(seed=3, shift=5.0)
    assert R.difference(model, close, "mean", 2).endswith("~")
    assert R.difference(model, far, "mean", 2) not in ("", "?") and not R.difference(model, far, "mean", 2).endswith("~")
    assert R.difference(model, {**far, "arrays": {}}, "mean", 2).endswith("?")


def test_generation_noise_is_at_least_the_spread_between_runs():
    entry = lambda **v: R.generation_entry(v, 8601, network="x", architecture="", label="x", spec={})  # noqa: E731
    a = entry(precision=12.7, precision_std=0.01, FID_val=0.03, FID_val_ci=0.001)
    b = entry(precision=12.9, precision_std=0.01, FID_val=0.04, FID_val_ci=0.001)
    assert R.difference(a, b, "precision", 2) == "-0.20~"
    assert R.difference(a, b, "FID_val", 4) == "-0.0100"
    assert R.difference(a, b, "dNN", 4) == ""


def test_save_and_read(tmp_path):
    numbers, arrays = tmp_path / "r.jsonl", tmp_path / "r.pt"
    R.save([completion("a"), completion("b", shift=1)], numbers, arrays)
    R.save([completion("a", shift=2)], numbers, arrays)  # replaces a
    entries = {e["network"]: e for e in R.read(numbers, arrays)}
    assert set(entries) == {"a", "b"} and entries["a"]["numbers"]["mean"] > entries["b"]["numbers"]["mean"]
    assert all(len(e["arrays"]["mean"]) == 1000 for e in entries.values())
    quick = R.first(entries["b"], 200)
    assert quick["n"] == 200 and quick["numbers"]["mean"] == pytest.approx(entries["b"]["arrays"]["mean"][:200].mean())
    other = [json.loads(line) for line in numbers.read_text().splitlines()]  # numbers of another run of the references
    other[0]["numbers"]["mean"] += 1
    numbers.write_text("".join(json.dumps(r) + "\n" for r in other))
    assert [bool(e["arrays"]) for e in R.read(numbers, arrays)] == [False, True]
    arrays.unlink()
    assert R.first(R.read(numbers, arrays)[0], 200) is None  # no arrays on this machine


def test_compare_ranks_and_targets():
    config = R.load()
    table = R.compare(completion("mine", shift=-50), R.recorded_entries(config), config)
    assert "| rank of the model | 1 of 6 | 1 of 6 |  | 1 of 6 |" in table
    assert "MLP 512x4: small c10 (same architecture)" in table and "target: DPoser full, inpaint" in table
    assert "no references" in R.compare(completion("mine", n=200), R.recorded_entries(config), config)
    assert "target" not in R.compare(completion("mine", n=200), R.recorded_entries(config), config)


def test_profile():
    config = R.load()
    runs = evaluate.profile(None, config, "MLP 512x4", R.METHODS)
    assert [(p, g) for p, g, _, _ in runs] == [("generation", ""), ("completion", "inpaint"), ("completion", "dflow"),
                                               ("denoising", "")]
    assert "no set" not in runs[1][3]
    assert "no set of MLP 9x9" in evaluate.profile(None, config, "MLP 9x9", ["inpaint"])[1][3]
    runs = evaluate.profile(["rsgm", "denoising=rsgm budget", "generation=own"], config, "MLP 512x4", ["inpaint"])
    assert runs[0][2] == ["--like", "own"] and runs[-1][2] == ["--like", "rsgm budget"]
    runs = evaluate.profile(["RSGM 512x4: their c10, inpaint t52"], config, "MLP 9x9", R.METHODS)
    assert [(g, a) for p, g, a, _ in runs if p == "completion"] == [
        ("inpaint", ["--like", "RSGM 512x4: their c10, inpaint t52"])]
    with pytest.raises(cli.UsageError):
        evaluate.profile(["nothing"], config, "MLP 9x9", R.METHODS)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """all with fake protocols: what they were called with, and fake results."""
    calls = []

    def protocol(name):
        def run(args):
            calls.append((name, args))
            label = args.run[0].rpartition("=")[0]
            if name == "completion":
                result = {o: {"mpvpe": RNG.normal(90, 5, (args.poses or 1000, 10)),
                              "apd": np.full(args.poses or 1000, 6.5)} for o in OCCLUSIONS}
            elif name == "denoising":
                result = {s: {k: RNG.normal(20, 1, (args.poses or 100, 60)) for k in ("mpjpe", "mpvpe")}
                          for s in (0.04, 0.1)}
            else:
                result = dict(label=label, split=args.split, poses=8601, wasserstein=15.0, wasserstein_std=0.1)
            args.collect[label] = ({}, result)
            return 0
        return run
    for name in R.PROTOCOLS:
        monkeypatch.setattr(evaluate, name, protocol(name))
    run = run_folder(tmp_path / "run", hidden=[512] * 4)
    return calls, ["all", "--run", f"mine={run}", "--like", "--reports", str(tmp_path / "reports")], tmp_path / "reports"


def test_all_is_validation_by_default(fake):
    calls, argv, reports = fake
    assert evaluate.main(argv) == 0
    assert [n for n, _ in calls] == ["generation", "completion", "completion", "denoising"]
    assert all(a.split == "valid" for _, a in calls)
    labels = [a.run[0].rpartition("=")[0] for _, a in calls]
    assert len(set(labels)) == 4
    text = (reports / "mine_valid.md").read_text()
    assert "## completion inpaint" in text and "## denoising" in text and "target" in text


def test_all_quick(fake):
    calls, argv, reports = fake
    assert evaluate.main([*argv, "--quick", "--methods", "inpaint"]) == 0
    assert [(n, getattr(a, "poses", None)) for n, a in calls] == [("generation", None), ("completion", 200),
                                                                 ("denoising", 30)]
    assert (reports / "mine_valid_quick.md").is_file()
    with pytest.raises(SystemExit):
        evaluate.main([*argv, "--quick", "--split", "test"])


def test_all_poses_and_test_split(fake):
    calls, argv, reports = fake
    assert evaluate.main([*argv, "--poses", "50", "--sequences", "7", "--protocols", "completion", "denoising"]) == 0
    assert [a.poses for _, a in calls] == [50, 50, 7]
    calls.clear()
    assert evaluate.main([*argv, "--split", "test", "--protocols", "denoising"]) == 0
    assert calls[0][1].split == "test"
    text = (reports / "mine_test.md").read_text()  # the references and the targets are numbers of the validation
    assert "small c10" not in text and "target" not in text and "no references" not in text


def test_all_stops_with_the_protocol(fake, monkeypatch):
    calls, argv, reports = fake
    monkeypatch.setattr(evaluate, "generation", lambda args: 1)  # stopped by Ctrl+C
    assert evaluate.main(argv) == 1 and not calls and not reports.exists()


def test_references_write_their_files(fake, monkeypatch, tmp_path):
    calls, _, _ = fake
    written = []
    monkeypatch.setattr(R, "resolve_run", lambda path: path)
    monkeypatch.setattr(R, "save", lambda entries: written.extend(entries))
    assert evaluate.main(["references", "--networks", "small c10"]) == 0
    assert [(e["protocol"], e["group"]) for e in written] == [("generation", ""), ("completion", "inpaint"),
                                                              ("completion", "dflow"), ("denoising", "")]
    assert all(e["commit"] and a.split == "valid" for e, (_, a) in zip(written, calls))
