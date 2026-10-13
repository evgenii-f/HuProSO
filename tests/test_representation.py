import pytest
import roma
import torch
from conftest import fixed_rotations

from hp.models.representation import REPRESENTATIONS, get_representation

NAMES = sorted(REPRESENTATIONS)


def is_rotation(r, atol=1e-5):
    eye = torch.eye(3).expand_as(r)
    return torch.allclose(r @ r.transpose(-1, -2), eye, atol=atol) and torch.allclose(r.det(), torch.ones(r.shape[:-2]), atol=atol)


@pytest.mark.parametrize("name", NAMES)
def test_round_trip_of_rotations(name):
    rep, pose = get_representation(name), fixed_rotations()
    vector = rep.to_vector(pose)
    assert vector.shape == (8, 19, rep.dim)
    assert torch.allclose(rep.from_vector(vector), pose, atol=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_pose_vector_has_the_joints_one_after_the_other(name):
    rep, pose = get_representation(name), fixed_rotations()
    vector = rep.pose_to_vector(pose)
    assert vector.shape == (8, 19 * rep.dim)
    assert torch.equal(vector[:, : rep.dim], rep.to_vector(pose[:, 0]))  # the first joint comes first
    assert torch.allclose(rep.pose_from_vector(vector, 19), pose, atol=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_any_vector_becomes_a_valid_rotation(name):
    rep = get_representation(name)
    torch.manual_seed(0)
    noise = torch.randn(64, 19, rep.dim) * 3
    assert is_rotation(rep.from_vector(noise))


@pytest.mark.parametrize("name", NAMES)
def test_gradient_flows_back_through_the_conversion(name):
    rep = get_representation(name)
    torch.manual_seed(0)
    z = torch.randn(4, 19, rep.dim, requires_grad=True)
    rep.from_vector(z).sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


@pytest.mark.parametrize("name", NAMES)
def test_identity_rotation(name):
    rep = get_representation(name)
    eye = torch.eye(3).expand(2, 19, 3, 3)
    assert torch.allclose(rep.from_vector(rep.to_vector(eye)), eye, atol=1e-6)


def test_conversion_keeps_the_rotation_not_only_the_matrix():
    # the angle between a rotation and its round trip is zero (geodesic distance, not entry-wise)
    pose = fixed_rotations()
    for name in NAMES:
        rep = get_representation(name)
        assert roma.rotmat_geodesic_distance(rep.from_vector(rep.to_vector(pose)), pose).max() < 1e-3


def test_unknown_representation_names_the_options():
    with pytest.raises(ValueError, match="6d"):
        get_representation("euler")
