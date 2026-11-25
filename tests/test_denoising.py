"""Motion denoising (hp/models/motion_denoising.py, hp/evaluation/denoising.py)."""
from pathlib import Path

import numpy as np
import pytest
import roma
import torch

from hp.evaluation.completion import KEEP
from hp.models.dflow import prior_of
from hp.models.motion_denoising import Skeleton, dflow_denoise, dposer_denoise, to_body
from hp.models.rsgm import RSGM

SMPLX = Path("~/data/body_models/smplx/SMPLX_NEUTRAL.npz").expanduser()
AMASS = Path("~/data/amass").expanduser()


def toy_skeleton():
    torch.manual_seed(0)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    return Skeleton(torch.randn(22, 3) * 0.2, parents)


def small_rsgm():
    torch.manual_seed(0)
    return RSGM(19, hidden=(32, 32)).eval()


@pytest.mark.skipif(not SMPLX.is_file(), reason="needs SMPL-X")
def test_the_skeleton_gives_the_joints_of_the_body_model():
    from hp.evaluation.denoising import Body

    body = Body(str(SMPLX))
    torch.manual_seed(1)
    poses = roma.rotvec_to_rotmat(0.5 * torch.randn(4, 21, 3))
    _, joints = body(poses)
    assert torch.allclose(body.skeleton(poses).double(), joints, atol=1e-5)


def test_to_body_puts_the_toes_at_rest():
    x = roma.random_rotmat(2 * 19).view(2, 19, 3, 3)
    body = to_body(x, KEEP)
    assert torch.allclose(body[:, KEEP], x)
    assert torch.allclose(body[:, [9, 10]], torch.eye(3).expand(2, 2, 3, 3))


def test_the_joints_do_not_depend_on_the_toes():
    sk = toy_skeleton()
    x = roma.random_rotmat(3 * 21).view(3, 21, 3, 3)
    y = x.clone()
    y[:, [9, 10]] = roma.random_rotmat(6).view(3, 2, 3, 3)
    assert torch.allclose(sk(x), sk(y), atol=1e-6)


def test_dflow_denoising_sequences_together_is_the_same_as_alone():
    """The loss is a sum over the sequences, the temporal term inside a sequence: together = alone."""
    model, sk = small_rsgm(), toy_skeleton()
    prior = prior_of(model, ode_steps=3)
    torch.manual_seed(2)
    noisy = sk(roma.random_rotmat(2 * 5 * 21).view(10, 21, 3, 3)).view(2, 5, 22, 3)
    torch.manual_seed(3)
    both = dflow_denoise(prior, sk, noisy, KEEP, steps=3, lr=0.05)
    alone = []
    for i in range(2):
        torch.manual_seed(3)
        prior.guess_weight = 0.75
        start = prior.start(torch.eye(3).repeat(10, 19, 1, 1))[5 * i:5 * (i + 1)]
        # the same start as in the batch: replace prior.start for this call
        prior.start = lambda y0, s=start: s
        alone.append(dflow_denoise(prior, sk, noisy[i:i + 1], KEEP, steps=3, lr=0.05))
        del prior.start
    assert torch.allclose(both, torch.cat(alone), atol=1e-4)


def test_dflow_denoising_lowers_the_joint_error():
    model, sk = small_rsgm(), toy_skeleton()
    prior = prior_of(model, ode_steps=3)
    torch.manual_seed(4)
    target = sk(roma.rotvec_to_rotmat(0.3 * torch.randn(8, 21, 3))).view(2, 4, 22, 3)
    torch.manual_seed(5)
    zero = dflow_denoise(prior, sk, target, KEEP, steps=1, lr=0.0)
    torch.manual_seed(5)
    fit = dflow_denoise(prior, sk, target, KEEP, steps=60, lr=0.05)
    error = lambda x: (sk(x.flatten(0, 1)).view(2, 4, 22, 3) - target).norm(dim=-1).mean()
    assert error(fit) < error(zero)


@pytest.mark.skipif(not SMPLX.is_file(), reason="needs SMPL-X")
def test_dposer_denoising_without_a_prior_fits_the_joints():
    from hp.evaluation.denoising import Body

    body = Body(str(SMPLX))
    torch.manual_seed(6)
    gt = roma.rotvec_to_rotmat(0.3 * torch.randn(2, 6, 21, 3))
    _, joints = body(gt.flatten(0, 1))
    noisy = joints.float().view(2, 6, 22, 3)
    init = torch.zeros(2, 6, 63)
    out = dposer_denoise(body.model, noisy, init, None, iterations=2, steps_per_iter=40)
    _, fit = body(out.flatten(0, 1))
    _, start = body(roma.rotvec_to_rotmat(init.view(12, 21, 3)))
    assert (fit - joints).norm(dim=-1).mean() < 0.5 * (start - joints).norm(dim=-1).mean()


def test_noise_is_fixed_per_sequence():
    from hp.evaluation.denoising import noisy_joints

    joints = torch.zeros(3, 4, 22, 3)
    a, b = noisy_joints(joints, 0.04), noisy_joints(joints[1:], 0.04)
    assert torch.equal(a[0], noisy_joints(joints[:1], 0.04)[0])
    assert not torch.equal(a[1], b[0])  # the seed is the index of the sequence
    assert abs(a.std().item() - 0.04) < 0.01


@pytest.mark.skipif(not (AMASS / "HumanEva").is_dir(), reason="needs the raw AMASS")
def test_humaneva_gives_their_190_sequences():
    from hp.evaluation.denoising import chunks_of

    files = sorted((AMASS / "HumanEva").glob("*/*_stageii.npz"))
    assert sum(len(chunks_of(f)) for f in files) == 190
    chunk = chunks_of(files[0])[0]
    assert chunk.shape == (60, 63) and np.isfinite(chunk).all()


def test_their_riemannian_adam_turns_every_joint_by_lr_over_sqrt2_in_the_first_step():
    """geoopt's RiemannianAdam on SO3Power: one second moment per joint, |hat(s)|_F^2 = 2 |s|^2."""
    from hp.models.dflow import riemannian_adam_dflow

    class Identity:  # x_1 = x_0, no running cost
        def generate(self, x):
            return x, x.new_zeros(())

    torch.manual_seed(7)
    start = roma.random_rotmat(2 * 4).view(2, 4, 3, 3)
    target = roma.random_rotmat(2 * 4).view(2, 4, 3, 3)
    seen = []
    loss = lambda x: (seen.append(x.detach()), ((x - target) ** 2).sum())[1]
    riemannian_adam_dflow(Identity(), loss, start, steps=2, lr=0.1, running_weight=0.0)
    angle = roma.rotmat_geodesic_distance(seen[0], seen[1])
    assert torch.allclose(angle, torch.full_like(angle, 0.1 / 2 ** 0.5), atol=1e-5)


@pytest.mark.skipif(not SMPLX.is_file(), reason="needs SMPL-X")
def test_dposer_denoising_takes_rsgm_as_the_prior():
    from hp.evaluation.denoising import Body

    body = Body(str(SMPLX))
    torch.manual_seed(8)
    gt = roma.rotvec_to_rotmat(0.3 * torch.randn(1, 5, 21, 3))
    _, joints = body(gt.flatten(0, 1))
    out = dposer_denoise(body.model, joints.float().view(1, 5, 22, 3), torch.zeros(1, 5, 63), small_rsgm(), KEEP,
                         iterations=1, steps_per_iter=3, t_max=0.05, t_min=0.01, sds_weight=10.0)
    assert out.shape == (1, 5, 21, 3, 3) and torch.isfinite(out).all()


@pytest.mark.skipif(not SMPLX.is_file() or not Path("~/data/experiments/denoising/smpl_mean_params.npz").expanduser()
                    .is_file(), reason="needs SMPL-X and the mean pose")
def test_a_stopped_run_continues_to_the_result_of_a_run_without_a_stop(tmp_path):
    from hp.evaluation.denoising import Body, noisy_joints, run

    body = Body(str(SMPLX))
    torch.manual_seed(9)
    gt = roma.rotvec_to_rotmat(0.3 * torch.randn(5, 4, 21, 3))
    noisy = noisy_joints(body(gt.flatten(0, 1))[1].float().view(5, 4, 22, 3), 0.04)
    conf = dict(iterations=1, steps_per_iter=3, t_max=0.05, t_min=0.01, sds_weight=10.0, progress=False)
    model = small_rsgm()
    whole = run(model, gt, noisy, 0.04, body, "dposer", batch=2, **conf)
    partial = tmp_path / "test_40mm.partial"
    calls = []
    stopped = run(model, gt, noisy, 0.04, body, "dposer", batch=2, partial=partial, key="k",
                  stop=lambda: calls.append(1) or True, **conf)
    assert stopped is None and len(calls) == 1 and torch.load(partial, weights_only=True)["done"] == 2
    other = run(model, gt, noisy, 0.04, body, "dposer", batch=2, partial=partial, key="other", **conf)
    resumed = run(model, gt, noisy, 0.04, body, "dposer", batch=2, partial=partial, key="k", **conf)
    assert torch.equal(resumed, whole) and torch.equal(other, whole)
