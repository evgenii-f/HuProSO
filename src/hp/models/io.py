"""Self-describing checkpoints: a folder with model.json (which model, how it was built) and the weights.

    save_model(model, "run/last")      # writes model.json and weights.pt
    model = load_model("run/last")     # rebuilds the right class from model.json and loads the weights

The model must have been built with build_model(), which records how.
"""
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from hp.models.base import PoseModel
from hp.models.registry import build_model

MANIFEST = "model.json"
FORMAT_VERSION = 1


def _plain(value):
    """OmegaConf containers become plain python, so that json can write them."""
    return OmegaConf.to_container(value) if OmegaConf.is_config(value) else value


def save_model(model: PoseModel, directory, weights: str = "weights") -> Path:
    """Write model.json and <weights>.pt into `directory`. Another `weights` name (e.g. "weights_ema")
    stores a second set of weights of the same model next to the first."""
    if not model.build_args:
        raise ValueError("The model was not made with build_model(), so a checkpoint could not rebuild it.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"format": FORMAT_VERSION, "build_args": {k: _plain(v) for k, v in model.build_args.items()}}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2))
    torch.save(model.state_dict(), directory / f"{weights}.pt")
    return directory


def load_model(directory, device="cpu", weights: str = "weights") -> PoseModel:
    """Rebuild the model of a checkpoint folder and load its weights. It is returned in eval mode."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found. Is {directory} a checkpoint written by save_model()?")
    manifest = json.loads(manifest_path.read_text())
    model = build_model(**manifest["build_args"])
    model.load_state_dict(torch.load(directory / f"{weights}.pt", map_location=device, weights_only=True))
    return model.to(device).eval()
