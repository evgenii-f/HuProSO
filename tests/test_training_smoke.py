"""Smoke test of the prior training: a few iterations on random rotations, with the config of the pretrained prior."""
import hashlib
import importlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")

from hp.core.prior_training import load_train_script, make_config  # noqa: E402
from hp.data.amass_cfg import ORIGINAL_AMASS_SPLITS  # noqa: E402
from hp.utils.paths import REPO_ROOT  # noqa: E402

BASE_CONFIG = REPO_ROOT / "experiments" / "pretrained_models" / "id_0900" / "config.yaml"
SMALL = {"batch_size": 64, "num_eval_samples": 100, "num_emd_samples": 50}
# 3 iterations: optimizer steps, a validation pass (interval 2) and an intermediate checkpoint (every 3)
STEPS = dict(max_iterations=3, valid_interval=2, save_ckpt_every=3, device="cpu")


@pytest.fixture
def rotations(tmp_path, monkeypatch):
    """Random rotation files <Dataset>.pt for every dataset of the official splits, as the preprocessing writes them."""
    folder = tmp_path / "rotations"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for names in ORIGINAL_AMASS_SPLITS.values():
        for name in names:
            q = rng.normal(size=(150, 55, 4))
            torch.save(torch.from_numpy(q / np.linalg.norm(q, axis=-1, keepdims=True)), folder / f"{name}.pt")
    monkeypatch.setenv("HUPROSO_AMASS_ROTATIONS", str(folder))
    monkeypatch.setenv("HUPROSO_AMASS_SPLIT", "original")
    monkeypatch.setenv("HUPROSO_DEVICE", "cpu")
    monkeypatch.delenv("HUPROSO_SUBSET", raising=False)
    return folder


def train(tmp_path, **kwargs):
    cfg = make_config(BASE_CONFIG, exp_id=1, save_root=str(tmp_path / "runs"), overrides=SMALL, **{**STEPS, **kwargs})
    load_train_script().main(cfg)
    return Path(cfg.save.path_dir)


def digest(folder):
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(Path(folder).glob("*")) if p.is_file()}


def test_training_runs_and_saves_a_loadable_model(rotations, tmp_path):
    run = train(tmp_path)
    assert run == tmp_path / "runs" / "id_0001"
    for name in ("config.yaml", "eval.json", "loss.png", "model_ckpt_pdf_estimator.pt", "permute_dims.pt"):
        assert (run / name).is_file(), name
    assert len(list(run.glob("model_ckpt_pdf_estimator_*.pt"))) == 1  # the intermediate checkpoint
    result = json.load(open(run / "eval.json"))
    assert len(result["loss"]) == 3 and np.isfinite(result["loss"]).all()
    assert result["valid_metrics"]["iter"] == [2]  # one validation pass, after the second iteration

    cfg = OmegaConf.load(run / "config.yaml")  # the saved model loads and evaluates
    cfg.only_eval, cfg.standard_mode, cfg.condition = True, True, False
    trainer = getattr(importlib.import_module("hp.core.trainer"), cfg.trainer)(cfg)
    trainer.load_model(save_path=str(run))
    assert np.isfinite(trainer.eval()["eval_avg_ll"])


def test_training_does_not_overwrite_an_existing_run(rotations, tmp_path):
    first = train(tmp_path, max_iterations=2, valid_interval=2, save_ckpt_every=10)
    before = digest(first)
    second = train(tmp_path, max_iterations=2, valid_interval=2, save_ckpt_every=10)
    assert second != first and second.name == "id_0001_0"
    assert digest(first) == before
