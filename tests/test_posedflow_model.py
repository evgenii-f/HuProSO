"""The wrapper of the pretrained PoseD-Flow model (hp/models/posedflow.py)."""
import math
from pathlib import Path

import pytest
import roma
import torch

from hp.models.posedflow import NUM_JOINTS, PoseDFlowPretrained, projx, vee, wrapped_normal_log_density

CHECKPOINT = Path("~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt").expanduser()


def test_the_wrapped_normal_is_a_density_relative_to_the_uniform():
    torch.manual_seed(0)
    uniform = roma.random_rotmat(400_000, dtype=torch.float64)
    assert wrapped_normal_log_density(uniform).exp().mean().item() == pytest.approx(1.0, abs=0.01)


def test_the_wrapped_normal_matches_its_samples():
    """The mean of a function of the angle under the density equals its mean over exp(N(0, I))."""
    torch.manual_seed(1)
    uniform = roma.random_rotmat(400_000, dtype=torch.float64)
    weights = wrapped_normal_log_density(uniform).exp()
    angle = roma.rotmat_to_rotvec(uniform).norm(dim=-1)
    samples = roma.rotvec_to_rotmat(torch.randn(400_000, 3, dtype=torch.float64))
    assert (weights * angle).mean().item() == pytest.approx(roma.rotmat_to_rotvec(samples).norm(dim=-1).mean().item(), abs=0.01)


def test_projx_keeps_rotations_and_projects_perturbed_ones():
    r = roma.random_rotmat(100, dtype=torch.float64)
    assert torch.allclose(projx(r), r, atol=1e-12)
    p = projx(r + 0.01 * torch.randn_like(r))
    assert torch.allclose(p @ p.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(100, 3, 3), atol=1e-10)
    assert torch.allclose(torch.linalg.det(p), torch.ones(100, dtype=torch.float64))


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="the PoseD-Flow checkpoint is not downloaded")
def test_the_checkpoint_loads_and_samples():
    model = PoseDFlowPretrained(CHECKPOINT, sample_steps=10).double()
    x = roma.random_rotmat(8 * NUM_JOINTS, dtype=torch.float64).reshape(8, NUM_JOINTS, 3, 3)
    t = torch.rand(8, dtype=torch.float64)
    ambient = model.field_ambient(t, x.reshape(8, -1)).view(8, NUM_JOINTS, 3, 3)
    assert torch.allclose(vee(x.transpose(-1, -2) @ ambient), model.field_frame(x, t), atol=1e-10)
    s = model.sample(16)
    assert s.shape == (16, 19, 3, 3)
    assert torch.allclose(torch.linalg.det(s), torch.ones(16, 19, dtype=torch.float64))
    raw = PoseDFlowPretrained(CHECKPOINT, weights="raw")
    assert not torch.equal(next(raw.parameters()), next(model.parameters()).float())


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="the PoseD-Flow checkpoint is not downloaded")
def test_the_log_prob_bases_differ_by_the_noise_density():
    model = PoseDFlowPretrained(CHECKPOINT, likelihood_steps=5).double()
    uniform = PoseDFlowPretrained(CHECKPOINT, likelihood_steps=5, base="uniform").double()
    x = model.sample_21(4)
    lp, lu = model.log_prob_21(x), uniform.log_prob_21(x)
    assert torch.isfinite(lp).all() and not torch.allclose(lp, lu)
