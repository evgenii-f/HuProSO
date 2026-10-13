import dataclasses
import os

import pytest
from omegaconf import OmegaConf
from test_training_smoke import rotations  # noqa: F401  (a fixture: random <Dataset>.pt files, and env settings)

from hp.models.io import load_model
from hp.models.so3_flow import SO3Flow
from hp.training import TrainConfig
from hp.training.cli import main
from hp.training.config import CONFIG_DIR, load_config, make_run_dir


def test_train_yaml_matches_the_train_config_defaults():
    train = OmegaConf.to_container(load_config().train)
    defaults = dataclasses.asdict(TrainConfig())
    assert train.keys() == defaults.keys()
    assert {k: v for k, v in train.items() if k != "device"} == {k: v for k, v in defaults.items() if k != "device"}


def test_the_default_is_the_paper_flow():
    cfg = load_config()
    assert cfg.model.name == "so3_flow" and cfg.model.args == {"flow": {}}


def test_command_line_beats_the_model_file_beats_the_base_config(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "train.yaml").write_text("model: tiny\nseed: 1\ntrain: {lr: 0.1, batch_size: 8}\n")
    (tmp_path / "models" / "tiny.yaml").write_text("name: tiny\nargs: {size: 3}\ntrain: {lr: 0.2}\n")
    cfg = load_config(["train.batch_size=4", "model.args.size=5"], config_dir=tmp_path)
    assert (cfg.train.lr, cfg.train.batch_size, cfg.model.args.size, cfg.seed) == (0.2, 4, 5, 1)
    assert load_config(["train.lr=0.3"], config_dir=tmp_path).train.lr == 0.3


def test_an_unknown_model_lists_the_available_ones():
    with pytest.raises(ValueError, match="so3_flow"):
        load_config(["model=nope"])
    assert (CONFIG_DIR / "models" / "so3_flow.yaml").is_file()


def test_run_folders_are_numbered_and_never_reused(tmp_path):
    first, second = make_run_dir(tmp_path, "m"), make_run_dir(tmp_path, "m")
    assert (first.name, second.name) == ("run_0001", "run_0002") and first.parent == tmp_path / "m"
    first.rmdir()
    assert make_run_dir(tmp_path, "m").name == "run_0003"  # the number of the deleted run is not given again


def args(tmp_path, rotations, *extra):  # noqa: F811
    return ["model.args.flow.layers=2", "train.max_iterations=3", "train.batch_size=64", "train.valid_interval=2",
            "train.checkpoint_every=3", "train.device=cpu", "train.progress=false", "tensorboard=false",
            f"run_root={tmp_path / 'runs'}", f"data.rotations={rotations}", "data.split=original", *extra]


def test_the_script_trains_and_saves_a_loadable_run(rotations, tmp_path):  # noqa: F811
    before = dict(os.environ)
    run = main(args(tmp_path, rotations))
    assert run == tmp_path / "runs" / "so3_flow" / "run_0001"
    assert sorted(p.name for p in run.iterdir()) == [
        "checkpoint_000003", "config.yaml", "history.json", "model.json", "weights.pt"]
    saved = OmegaConf.load(run / "config.yaml")
    assert saved.train.max_iterations == 3 and saved.model.args.flow.layers == 2
    model = load_model(run)
    assert isinstance(model, SO3Flow) and model.num_joints == 19
    assert main(args(tmp_path, rotations)).name == "run_0002"  # the second run does not touch the first
    assert dict(os.environ) == before  # the settings of the run did not leak into the environment
