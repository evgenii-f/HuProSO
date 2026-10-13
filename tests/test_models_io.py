import json

import pytest
import torch
from conftest import fixed_rotations
from omegaconf import OmegaConf
from test_models_interface import Gaussian

from hp.models import build_model, register_model, registry
from hp.models.io import MANIFEST, load_model, save_model


@pytest.fixture
def gaussian(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})
    register_model("gaussian")(Gaussian)
    model = build_model("gaussian", num_joints=19, representation="6d")
    with torch.no_grad():
        model.mean.add_(torch.randn_like(model.mean))  # not the initial weights
    return model


def test_round_trip_of_a_simple_model(gaussian, tmp_path):
    save_model(gaussian, tmp_path / "run")
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == [MANIFEST, "weights.pt"]
    loaded = load_model(tmp_path / "run")
    assert isinstance(loaded, Gaussian) and loaded.build_args == gaussian.build_args
    assert not loaded.training  # returned in eval mode
    pose = fixed_rotations()
    assert torch.equal(loaded.loss(pose), gaussian.loss(pose))


def test_the_manifest_says_which_model_and_how_it_was_built(gaussian, tmp_path):
    save_model(gaussian, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST).read_text())
    assert manifest["build_args"] == {"name": "gaussian", "num_joints": 19, "representation": "6d"}


def test_the_flow_round_trip_keeps_the_dimension_order(tmp_path):
    model = build_model("so3_flow", num_joints=4, flow={"layers": 2}).eval()
    save_model(model, tmp_path)
    loaded = load_model(tmp_path)  # starts with another random order of the dimensions
    pose = fixed_rotations(joints=4)
    with torch.no_grad():
        assert torch.equal(loaded.log_prob(pose), model.log_prob(pose))


def test_config_objects_in_the_build_arguments_are_written_as_plain_json(tmp_path):
    model = build_model("so3_flow", num_joints=4, flow=OmegaConf.create({"layers": 2}))
    save_model(model, tmp_path)
    assert json.loads((tmp_path / MANIFEST).read_text())["build_args"]["flow"] == {"layers": 2}
    assert load_model(tmp_path).num_parameters() == model.num_parameters()


def test_a_second_set_of_weights_next_to_the_first(gaussian, tmp_path):
    other = build_model("gaussian", num_joints=19, representation="6d")
    save_model(gaussian, tmp_path)
    save_model(other, tmp_path, weights="weights_ema")
    pose = fixed_rotations()
    assert torch.equal(load_model(tmp_path).loss(pose), gaussian.loss(pose))
    assert torch.equal(load_model(tmp_path, weights="weights_ema").loss(pose), other.loss(pose))


def test_a_model_not_made_by_build_model_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="build_model"):
        save_model(Gaussian(num_joints=19), tmp_path)


def test_a_folder_without_a_manifest_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="save_model"):
        load_model(tmp_path)


def test_an_unknown_model_in_the_manifest_lists_the_available_ones(gaussian, tmp_path):
    save_model(gaussian, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST).read_text())
    manifest["build_args"]["name"] = "vanished"
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unknown model 'vanished'"):
        load_model(tmp_path)
