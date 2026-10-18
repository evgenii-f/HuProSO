import dataclasses
import json
import os
import signal

import pytest
from omegaconf import OmegaConf
from test_training_smoke import rotations  # noqa: F401  (a fixture: random <Dataset>.pt files, and env settings)

from hp.models.io import load_model
from hp.models.so3_flow import SO3Flow
from hp.training import TrainConfig, Trainer, find_checkpoint, read_state
from hp.training.cli import EXIT_STOPPED, exit_code, main
from hp.training.config import CONFIG_DIR, load_config, make_run_dir


def test_train_yaml_matches_the_train_config_defaults():
    train = OmegaConf.to_container(load_config().train)
    defaults = {f.name: f.default for f in dataclasses.fields(TrainConfig)}  # as declared (weight_decay: None)
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


def test_the_model_can_be_chosen_and_its_arguments_changed_together(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "train.yaml").write_text("model: tiny\nseed: 1\ntrain: {lr: 0.1}\n")
    (tmp_path / "models" / "tiny.yaml").write_text("name: tiny\nargs: {size: 3}\n")
    (tmp_path / "models" / "big.yaml").write_text("name: big\nargs: {size: 30, depth: 2}\n")
    for overrides in (["model=big", "model.args.size=40"], ["model.args.size=40", "model=big"]):
        cfg = load_config(overrides, config_dir=tmp_path)
        assert cfg.model.name == "big" and OmegaConf.to_container(cfg.model.args) == {"size": 40, "depth": 2}


def test_a_config_file_that_is_no_mapping_is_refused(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "train.yaml").write_text("- model\n- seed\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(config_dir=tmp_path)


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
        "checkpoint_000003", "config.yaml", "history.json", "model.json", "status.json", "train_state.pt", "weights.pt"]
    saved = OmegaConf.load(run / "config.yaml")
    assert saved.train.max_iterations == 3 and saved.model.args.flow.layers == 2
    model = load_model(run)
    assert isinstance(model, SO3Flow) and model.num_joints == 19
    assert main(args(tmp_path, rotations)).name == "run_0002"  # the second run does not touch the first
    assert dict(os.environ) == before  # the settings of the run did not leak into the environment


def test_resume_goes_on_in_the_same_run_folder(rotations, tmp_path):  # noqa: F811
    base = args(tmp_path, rotations, "train.max_iterations=4", "train.checkpoint_every=2", "train.keep_last=0", "train.keep_best=0", "tensorboard=false")
    run = main(base)
    full = load_model(run)
    (run / "weights.pt").unlink(), (run / "history.json").unlink(), (run / "train_state.pt").unlink()  # a run that was killed
    assert main(["resume=" + str(run), "train.device=cpu"]) == run  # no run_0002: the newest checkpoint of run_0001
    assert [p.name for p in (tmp_path / "runs" / "so3_flow").iterdir()] == ["run_0001"]
    # the newest checkpoint is the last step: nothing to train, no new session, only the final files are rewritten
    assert not list(run.glob("config_resume_*")) and len(json.loads((run / "history.json").read_text())["loss"]) == 4
    assert exit_code(run) == 0
    assert isinstance(load_model(run), SO3Flow) and full.num_joints == 19


def test_resume_from_a_checkpoint_with_more_steps_to_do(rotations, tmp_path):  # noqa: F811
    run = main(args(tmp_path, rotations, "train.max_iterations=4", "train.checkpoint_every=2", "train.keep_last=0", "train.keep_best=0"))
    assert main([f"resume={run / 'checkpoint_000002'}", "train.device=cpu", "train.progress=false"]) == run
    assert len(json.loads((run / "history.json").read_text())["loss"]) == 4  # steps 3 and 4 were made again


def test_the_script_trains_with_adamw_and_resumes_it(rotations, tmp_path):  # noqa: F811
    run = main(args(tmp_path, rotations, "train.max_iterations=4", "train.checkpoint_every=2", "train.keep_last=0",
                    "train.keep_best=0", "train.optimizer=adamw", "train.weight_decay=0.01"))
    saved = OmegaConf.load(run / "config.yaml")
    assert saved.train.optimizer == "adamw" and saved.train.weight_decay == 0.01
    assert read_state(run)["config"]["optimizer"] == "adamw"
    assert main([f"resume={run / 'checkpoint_000002'}", "train.device=cpu", "train.progress=false"]) == run
    assert len(json.loads((run / "history.json").read_text())["loss"]) == 4


def test_the_saved_config_holds_the_weight_decay_that_was_used(rotations, tmp_path):  # noqa: F811
    for extra, optimizer, weight_decay in (((), "adam", 0.0), (("train.optimizer=adamw",), "adamw", 0.01),
                                          (("train.optimizer=adamw", "train.weight_decay=0.0"), "adamw", 0.0)):
        run = main(args(tmp_path, rotations, *extra))
        saved = OmegaConf.load(run / "config.yaml").train
        assert (saved.optimizer, saved.weight_decay) == (optimizer, weight_decay)
        assert read_state(run)["config"]["weight_decay"] == weight_decay


def test_resume_refuses_to_change_the_model_or_the_optimization(rotations, tmp_path):  # noqa: F811
    run = main(args(tmp_path, rotations, "train.max_iterations=4", "train.checkpoint_every=2"))
    for bad in ("train.lr=0.1", "train.max_iterations=8", "train.optimizer=adamw", "train.weight_decay=0.01",
                "model.args.flow.layers=3", "seed=1"):
        with pytest.raises(ValueError, match="Cannot change"):
            main([f"resume={run}", bad])
    with pytest.raises(FileNotFoundError, match="Nothing to resume"):
        main([f"resume={tmp_path}"])


def test_find_checkpoint_names_the_run_and_the_state_to_use(rotations, tmp_path):  # noqa: F811
    run = main(args(tmp_path, rotations, "train.max_iterations=4", "train.checkpoint_every=2", "train.keep_last=0", "train.keep_best=0"))
    assert find_checkpoint(run) == (run, run)  # a finished run holds its final state
    assert find_checkpoint(run / "checkpoint_000002") == (run, run / "checkpoint_000002")
    (run / "train_state.pt").unlink()
    assert find_checkpoint(run) == (run, run / "checkpoint_000004")  # a run that stopped: the newest checkpoint
    (run / "checkpoint_000004" / "train_state.pt").unlink()  # one of the old kind, weights only, is skipped
    assert find_checkpoint(run) == (run, run / "checkpoint_000002")


def test_a_signal_stops_the_run_cleanly_and_resume_finishes_it(rotations, tmp_path, monkeypatch):  # noqa: F811
    step = Trainer.training_step
    calls = []

    def step_then_signal(self, batch):  # the scheduler's warning arrives during the second step
        calls.append(1)
        if len(calls) == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return step(self, batch)

    base = args(tmp_path, rotations, "train.max_iterations=4", "train.checkpoint_every=0")
    handler_before = signal.getsignal(signal.SIGTERM)
    with monkeypatch.context() as patch:
        patch.setattr(Trainer, "training_step", step_then_signal)
        run = main(base)
    assert exit_code(run) == EXIT_STOPPED and not (run / "weights.pt").exists()
    assert [p.name for p in run.glob("checkpoint_*")] == ["checkpoint_000002"]
    assert signal.getsignal(signal.SIGTERM) is handler_before  # the handler of the run is taken off again
    assert main([f"resume={run}", "train.device=cpu"]) == run
    assert exit_code(run) == 0 and len(json.loads((run / "history.json").read_text())["loss"]) == 4


def test_the_time_limit_stops_the_script_with_a_status_to_resubmit(rotations, tmp_path):  # noqa: F811
    run = main(args(tmp_path, rotations, "train.max_iterations=4", "train.time_limit_minutes=1e-9"))
    assert exit_code(run) == EXIT_STOPPED == 75
    assert main([f"resume={run}", "train.device=cpu", "train.time_limit_minutes=null"]) == run
    assert exit_code(run) == 0


def test_resume_may_move_the_data_but_not_change_the_split(rotations, tmp_path):  # noqa: F811
    run = main(args(tmp_path, rotations, "train.max_iterations=4", "train.checkpoint_every=2"))
    with pytest.raises(ValueError, match="Cannot change"):
        main([f"resume={run / 'checkpoint_000002'}", "data.split=subset"])
    assert main([f"resume={run / 'checkpoint_000002'}", f"data.rotations={rotations}", "train.device=cpu"]) == run
