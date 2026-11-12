"""VPoser with the network and loss of their V02_05, trained here (hp/models/vposer.py)."""
from pathlib import Path

import numpy as np
import pytest
import roma
import torch
from conftest import fixed_rotations, tiny_body_model

from hp.models import build_model
from hp.models.vposer import OUR_19, BodyMesh, VPoser, continuous_rotation, geodesic

torch.set_default_dtype(torch.float32)
SMPLX = Path("~/data/body_models/smplx/SMPLX_NEUTRAL.npz").expanduser()


@pytest.fixture
def body(tmp_path):
    return tiny_body_model(tmp_path / "body.npz")


def model(body, joints=(1, 2, 3, 4), **kwargs):
    torch.manual_seed(0)
    args = {"num_neurons": 32, "latent_dim": 4, "body_model": body, "joints": list(joints), **kwargs}
    return build_model("vposer", num_joints=len(joints), **args)


def their_decoder(module_input):
    """ContinousRotReprDecoder.forward of human_body_prior/models/vposer_model.py, as they wrote it."""
    reshaped_input = module_input.view(-1, 3, 2)
    b1 = torch.nn.functional.normalize(reshaped_input[:, :, 0], dim=1)
    dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    b2 = torch.nn.functional.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def test_the_rotation_decoder_is_theirs_and_gives_back_a_rotation_from_its_first_two_columns():
    torch.manual_seed(0)
    x = torch.randn(10, 6)
    assert torch.equal(continuous_rotation(x), their_decoder(x))
    r = roma.random_rotmat(10)
    assert torch.allclose(continuous_rotation(r[:, :, :2].reshape(10, 6)), r, atol=1e-6)


def test_the_geodesic_is_the_angle_between_the_rotations():
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    r1, r2 = roma.rotvec_to_rotmat(0.3 * axis), roma.rotvec_to_rotmat(-0.4 * axis)
    assert torch.allclose(geodesic(r1, r2), torch.tensor(0.7), atol=1e-5)


def test_the_architecture_is_theirs():
    m = build_model("vposer", num_joints=21, body_model="unused.npz")
    # their log: "Total Trainable Parameters Count in vp_model is 0.94 M."
    assert round(m.num_parameters() / 1e6, 2) == 0.94
    assert m.smplx_joints == list(range(1, 22))
    assert build_model("vposer", num_joints=19).smplx_joints == OUR_19 and 10 not in OUR_19 and 11 not in OUR_19
    with pytest.raises(ValueError, match="joints"):
        build_model("vposer", num_joints=4)


def test_the_body_at_rest_and_one_joint_turned(body):
    mesh = BodyMesh(body, [1, 2])
    data = np.load(body)
    v, j = mesh(torch.eye(3).repeat(1, 2, 1, 1))
    assert torch.allclose(v[0], torch.as_tensor(data["v_template"]), atol=1e-6)
    assert torch.allclose(j[0], torch.as_tensor(data["J_regressor"] @ data["v_template"]), atol=1e-6)
    # turning joint 1 moves its subtree (joints 3 and 4) rigidly about it, and not joint 2
    r = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.5, 0.0]))
    v, j_turned = mesh(torch.stack([r, torch.eye(3)])[None])
    rest = j[0]
    for child in (3, 4):
        assert torch.allclose(j_turned[0, child], rest[1] + r @ (rest[child] - rest[1]), atol=1e-5)
    assert torch.allclose(j_turned[0, 2], rest[2]) and torch.allclose(j_turned[0, 1], rest[1])


@pytest.mark.skipif(not SMPLX.is_file(), reason="needs the SMPL-X body model")
def test_the_smplx_body_at_rest_is_its_template():
    mesh = BodyMesh(SMPLX, OUR_19)
    v, _ = mesh(torch.eye(3).repeat(2, 19, 1, 1))
    assert v.shape == (2, 10475, 3)
    assert torch.allclose(v[1], torch.as_tensor(np.load(SMPLX)["v_template"]), atol=1e-6)


def test_the_kl_term_is_that_of_two_normals(body):
    m = model(body).train()
    pose = fixed_rotations(batch=6, joints=4)
    mean, scale = m.encode_distribution(pose)
    q, p = torch.distributions.Normal(mean, scale), torch.distributions.Normal(0.0, 1.0)
    expected = torch.distributions.kl_divergence(q, p).sum(-1).mean()
    ours = (0.5 * (mean ** 2 + scale ** 2) - torch.log(scale) - 0.5).sum(-1).mean()
    assert torch.allclose(ours, expected, atol=1e-6)


def test_the_extra_terms_are_on_for_the_first_steps_and_the_steps_count_in_training_only(body):
    m = model(body, extra_terms_steps=1).train()
    pose = fixed_rotations(batch=6, joints=4)
    torch.manual_seed(1)
    with_extra = m.loss(pose)
    assert m.steps == 1
    torch.manual_seed(1)
    without = m.loss(pose)  # the same noise and dropout, step 1: no extra terms
    assert m.steps == 2 and with_extra > without
    m.eval()
    m.loss(pose)
    assert m.steps == 2


def test_the_validation_loss_is_their_v2v_with_fixed_noise(body):
    m = model(body).eval()
    pose = fixed_rotations(batch=6, joints=4)
    first = m.loss(pose)
    assert torch.equal(first, m.loss(pose))
    mean, scale = m.encode_distribution(pose)
    z = mean + scale * torch.randn(mean.shape, generator=torch.Generator().manual_seed(0))
    v_rec, _ = m.mesh()(m.decode_latent(z))
    v, _ = m.mesh()(pose)
    assert torch.allclose(first, (v_rec - v).norm(dim=-1).mean())


def test_samples_are_decoded_without_dropout_and_need_no_body_model():
    m = build_model("vposer", num_joints=4, num_neurons=32, latent_dim=4, body_model="missing.npz", joints=[1, 2, 3, 4])
    m.train()
    torch.manual_seed(3)
    in_training = m.sample(5)
    m.eval()
    torch.manual_seed(3)
    assert torch.equal(in_training, m.sample(5)) and m.training is False
    assert m.decoder_net.training is False
    m.train()
    m.sample(1)
    assert m.decoder_net.training  # the mode is restored
    with pytest.raises(FileNotFoundError):
        m.loss(fixed_rotations(batch=2, joints=4))


def test_the_body_model_is_not_saved_with_the_weights(body):
    m = model(body).eval()
    m.loss(fixed_rotations(batch=2, joints=4))
    assert isinstance(m._mesh, BodyMesh) and "_mesh" not in dict(m.named_children())
    assert set(m.state_dict()) == set(VPoser(4, num_neurons=32, latent_dim=4, joints=[1, 2, 3, 4]).state_dict())
    assert "steps" in m.state_dict()
