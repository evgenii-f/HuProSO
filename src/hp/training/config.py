"""Configuration of the training script: config/train.yaml, then config/models/<model>.yaml, then the command line."""
from pathlib import Path
from typing import Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from hp.utils.paths import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "config"


def _load_mapping(path: Path) -> DictConfig:
    """A yaml file that must hold a mapping (OmegaConf.load may also give a list)."""
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"{path} must hold a mapping of settings, not a list.")
    return config


def load_config(overrides: Sequence[str] = (), config_dir: Optional[Path] = None) -> DictConfig:
    """Merge the base config, the config of the chosen model and `overrides` (dotted "key=value" strings).

    `model=<name>` chooses the model file; other `model.*` overrides change that model's arguments.
    """
    config_dir = Path(config_dir or CONFIG_DIR)
    base = _load_mapping(config_dir / "train.yaml")
    # "model=<name>" is taken out first: in one dotlist with "model.args.*" it would be replaced by a dict
    names = [o.split("=", 1)[1] for o in overrides if o.startswith("model=")]
    cli = OmegaConf.from_dotlist([o for o in overrides if not o.startswith("model=")])
    name = base.pop("model")
    if names:
        name = names[-1]
    path = config_dir / "models" / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (config_dir / "models").glob("*.yaml"))
        raise ValueError(f"No config for model '{name}'. Available: {available}.")
    model_file = _load_mapping(path)
    chosen = {"model": {"name": model_file.name, "args": model_file.get("args", {})}, "train": model_file.get("train", {})}
    config = OmegaConf.merge(base, chosen, cli)
    assert isinstance(config, DictConfig)  # a merge into a mapping is a mapping; OmegaConf types it as either
    return config


def make_run_dir(root, name: str) -> Path:
    """<root>/<name>/run_0001, then run_0002, ...: one more than the highest number there is, so that an
    existing run is never overwritten and the number of a deleted run is not given to a new one."""
    parent = Path(root).expanduser() / name
    numbers = [int(p.name[4:]) for p in parent.glob("run_[0-9][0-9][0-9][0-9]")]
    run_dir = parent / f"run_{max(numbers, default=0) + 1:04d}"
    run_dir.mkdir(parents=True)
    return run_dir


# What may be changed when a run is resumed: where the data is (the same data: the trainer checks its size), how it
# is loaded, the device, how often things are written. Everything else (the data split, the model, the optimization)
# must stay as the run had it.
RESUMABLE = ("data.rotations", "data.load_dtype", "tensorboard", "train.device", "train.num_workers", "train.progress",
             "train.checkpoint_every", "train.keep_last", "train.keep_best", "train.time_limit_minutes")


def _leaves(node, prefix=""):
    for key, value in node.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _leaves(value, name + ".")
        else:
            yield name


def load_resume_config(run_dir, overrides: Sequence[str] = ()) -> DictConfig:
    """The config.yaml of the run, with the `overrides` that are allowed when resuming (RESUMABLE)."""
    saved = _load_mapping(Path(run_dir) / "config.yaml")
    cli = OmegaConf.from_dotlist(list(overrides))
    refused = [k for k in _leaves(OmegaConf.to_container(cli)) if not any(k == r or k.startswith(r + ".") for r in RESUMABLE)]
    if refused:
        raise ValueError(f"Cannot change {refused} when resuming a run; only {list(RESUMABLE)} can be changed.")
    config = OmegaConf.merge(saved, cli)
    assert isinstance(config, DictConfig)
    return config
