"""The wrapper of the pretrained NRDF model (hp/models/nrdf.py)."""
from pathlib import Path

import pytest
import roma
import torch
from torch.nn import functional as F

from hp.models.nrdf import NRDFField, NRDFPretrained, geodesic, hamilton, nearest_distances, perturb, rdf_step

CHECKPOINT = Path("~/data/nrdf_ckpt/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1/checkpoints/"
                  "checkpoint_epoch_best.tar").expanduser()


def unit_quaternions(*shape):
    q = F.normalize(torch.randn(*shape, 4, dtype=torch.float64), dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def test_hamilton_is_the_product_of_the_rotations():
    torch.manual_seed(0)
    a, b = unit_quaternions(50), unit_quaternions(50)
    product = roma.unitquat_to_rotmat(hamilton(a, b)[..., [1, 2, 3, 0]])
    expected = roma.unitquat_to_rotmat(a[..., [1, 2, 3, 0]]) @ roma.unitquat_to_rotmat(b[..., [1, 2, 3, 0]])
    assert torch.allclose(product, expected, atol=1e-12)


def test_a_step_at_distance_zero_keeps_the_pose():
    torch.manual_seed(1)
    q, egrad = unit_quaternions(20, 21), torch.randn(20, 21, 4, dtype=torch.float64)
    assert torch.allclose(rdf_step(q, egrad, torch.zeros(20, dtype=torch.float64), 0.01), q, atol=1e-12)


def test_a_step_gives_unit_quaternions_with_a_positive_real_part():
    torch.manual_seed(2)
    q, egrad = unit_quaternions(20, 21), torch.randn(20, 21, 4, dtype=torch.float64)
    out = rdf_step(q, egrad, torch.ones(20, dtype=torch.float64), 0.1)
    assert torch.allclose(out.norm(dim=-1), torch.ones(20, 21, dtype=torch.float64), atol=1e-12)
    assert (out[..., 0] >= 0).all()


def test_the_distance_is_the_exact_minimum_when_all_poses_are_candidates():
    torch.manual_seed(5)
    clean, noisy = unit_quaternions(60, 21).float(), unit_quaternions(8, 21).float()
    brute = geodesic(noisy[:, None], clean).amin(1)
    assert torch.allclose(nearest_distances(noisy, clean[:8], clean, k=60, block=7), brute)


def test_the_blocks_do_not_change_the_candidates():
    torch.manual_seed(6)
    clean, noisy = unit_quaternions(200, 21).float(), unit_quaternions(8, 21).float()
    assert torch.allclose(nearest_distances(noisy, clean[:8], clean, k=20, block=33), nearest_distances(noisy, clean[:8], clean, k=20))


def test_the_pose_it_came_from_bounds_the_distance():
    torch.manual_seed(7)
    origin = unit_quaternions(8, 21).float()
    noisy = perturb(origin.double(), 0.785).float()
    assert (nearest_distances(noisy, origin, unit_quaternions(50, 21).float(), k=5) <= geodesic(noisy, origin) + 1e-6).all()


def test_the_loss_of_a_field_is_finite_and_has_gradients():
    torch.manual_seed(8)
    net = NRDFField()
    loss = net.loss(unit_quaternions(4, 21).float(), torch.ones(4), unit_quaternions(4, 21).float())
    loss.backward()
    assert loss.isfinite() and all(p.grad is not None for p in net.parameters())


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="the NRDF checkpoint is not downloaded")
def test_the_checkpoint_loads_and_the_projection_lowers_the_distance():
    torch.manual_seed(3)
    model = NRDFPretrained(CHECKPOINT, iterations=20).eval()
    start = model.start(50)
    assert torch.allclose(start.norm(dim=-1), torch.ones(50, 21), atol=1e-5)
    assert model.distance(model.project(start)).mean() < model.distance(start).mean()


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="the NRDF checkpoint is not downloaded")
def test_samples_are_rotations_of_our_joints_and_of_no_density():
    torch.manual_seed(4)
    model = NRDFPretrained(CHECKPOINT, iterations=5).eval()
    with torch.no_grad():  # as the evaluation
        samples = model.sample(10).double()
    assert samples.shape == (10, 19, 3, 3) and not model.has_log_prob
    assert (samples.transpose(-1, -2) @ samples - torch.eye(3, dtype=torch.float64)).abs().max() < 1e-5
