"""Locations of data on disk and the compute device, configurable without reinstalling.

Each value is looked up, in this order, in:
  1. an explicit argument (``override``),
  2. an environment variable (see ``ENV_VARS``),
  3. ``config/paths.yaml`` in the repository (copy of ``config/paths.example.yaml``),
  4. ``~/.config/huproso/paths.yaml``.
The files are read on every call, so editing them takes effect immediately.

Shell usage: ``python -m hp.utils.paths amass_rotations``.
"""
import os
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILES = [
    REPO_ROOT / "config" / "paths.yaml",
    Path("~/.config/huproso/paths.yaml").expanduser(),
]
ENV_VARS = {
    "amass_raw": "HUPROSO_AMASS_RAW",
    "amass_rotations": "HUPROSO_AMASS_ROTATIONS",
    "body_models": "HUPROSO_BODY_MODELS",
    "amass_split": "HUPROSO_AMASS_SPLIT",
    "device": "HUPROSO_DEVICE",
}
DEFAULTS = {"amass_split": "original", "device": "auto"}
HINTS = {
    "amass_raw": "directory with the extracted AMASS datasets (ACCAD/, CMU/, ...)",
    "amass_rotations": "directory with the preprocessed <Dataset>.pt files",
    "body_models": "directory with smplx/SMPLX_NEUTRAL.npz",
    "amass_split": "'original', 'subset' or 'posedflow'",
    "device": "'auto', 'cpu', 'mps' or 'cuda:N'",
}


class PathNotConfigured(RuntimeError):
    pass


def get_path(key: str, override: Optional[str] = None) -> str:
    if key not in ENV_VARS:
        raise KeyError(f"Unknown setting '{key}'. Known: {sorted(ENV_VARS)}")
    value = override or os.environ.get(ENV_VARS[key])
    if not value:
        for file in CONFIG_FILES:
            if file.is_file():
                value = OmegaConf.load(file).get(key)
                if value:
                    break
    if not value:
        value = DEFAULTS.get(key)
    if not value:
        raise PathNotConfigured(
            f"'{key}' ({HINTS[key]}) is not configured. Set it in "
            f"{CONFIG_FILES[0]} (see config/paths.example.yaml), in "
            f"{CONFIG_FILES[1]}, or via the environment variable {ENV_VARS[key]}."
        )
    value = str(value)
    return value if key in ("amass_split", "device") else os.path.expanduser(value)


def get_amass_split() -> str:
    # HUPROSO_SUBSET=1 is kept as a shortcut for amass_split=subset.
    if os.environ.get("HUPROSO_SUBSET"):
        return "subset"
    split = get_path("amass_split")
    if split not in ("original", "subset", "posedflow"):
        raise PathNotConfigured(f"amass_split must be 'original', 'subset' or 'posedflow', got '{split}'.")
    return split


if __name__ == "__main__":
    print(get_path(sys.argv[1]))
