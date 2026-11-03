"""DPoser, a score-based diffusion prior on z-scored axis-angle (hp/models/dposer.py)."""
import math

import pytest
import roma
import torch
from conftest import fixed_rotations

from hp.models import build_model
from hp.models.dposer import DPoser, normalization, timestep_embedding

torch.set_default_dtype(torch.float32)


def model(joints=2, **kwargs):
    torch.manual_seed(0)
    args = {"hidden": 32, "embed": 16, "blocks": 1, "sample_steps": 5, **kwargs}
    return build_model("dposer", num_joints=joints, **args).eval()


def test_the_sub_vp_marginal_and_diffusion():
    m = model()
    t = torch.tensor([0.0, 0.3, 1.0])
    alpha, std = m.marginal(t)
    assert torch.allclose(alpha[0], torch.tensor(1.0)) and std[0] == 0
    integral = m.beta_min * t + 0.5 * (m.beta_max - m.beta_min) * t ** 2  # int_0^t beta
    assert torch.allclose(alpha, torch.exp(-0.5 * integral)) and torch.allclose(std, 1 - torch.exp(-integral))
    # sub-VP: g^2 = beta (1 - exp(-2 int beta))
    assert torch.allclose(m.diffusion(t) ** 2, m.beta(t) * (1 - torch.exp(-2 * integral)))


def test_the_time_embedding_is_sinusoidal():
    emb = timestep_embedding(torch.tensor([0.0, 500.0]), 8)
    assert emb.shape == (2, 8)
    assert torch.equal(emb[0], torch.tensor([0.0] * 4 + [1.0] * 4))
    freqs = torch.exp(-math.log(10_000) * torch.arange(4) / 3)
    assert torch.allclose(emb[1], torch.cat([torch.sin(500 * freqs), torch.cos(500 * freqs)]))


def test_the_output_is_scaled_by_the_sigma_of_its_time_and_the_score_by_std():
    m = model()
    x, t = torch.randn(3, 6), torch.tensor([0.0, 0.5, 1.0])
    raw = m.net(x, t * 999)
    sigmas = torch.exp(torch.linspace(math.log(50.0), math.log(0.01), 1000))[(t * 999).long()]
    assert torch.allclose(m.noise_prediction(x, t), raw / sigmas[:, None], rtol=1e-5)
    t = torch.tensor([0.2, 0.5, 0.9])
    assert torch.allclose(m.score(x, t), -m.noise_prediction(x, t) / m.marginal(t)[1][:, None])


def test_the_normalization_round_trips_and_is_fitted_like_torch_std():
    poses = fixed_rotations(batch=50, joints=2)
    mean, std = normalization(poses, chunk=7)
    vectors = roma.rotmat_to_rotvec(poses).reshape(50, 6).double()
    assert torch.allclose(torch.tensor(mean, dtype=torch.float64), vectors.mean(0))
    assert torch.allclose(torch.tensor(std, dtype=torch.float64), vectors.std(0))
    m = model(mean=mean, std=std)
    x = m.normalize(poses)
    assert torch.allclose(x.mean(0), torch.zeros(6), atol=1e-5) and torch.allclose(x.std(0), torch.ones(6), atol=1e-5)
    assert torch.allclose(m.denormalize(x), poses, atol=1e-5)


def test_a_perfect_noise_prediction_has_zero_loss():
    m = model()
    pose = fixed_rotations(batch=16, joints=2)
    x0 = m.normalize(pose)
    g = torch.Generator().manual_seed(0)  # the same draws as loss() in eval mode
    t = torch.rand(16, generator=g) * (1 - m.loss_eps) + m.loss_eps
    z = torch.randn(x0.shape, generator=g)
    m.noise_prediction = lambda x, tt: z  # an oracle
    assert m.loss(pose) == 0


def test_the_loss_is_the_same_in_eval_mode_and_varies_in_training_mode():
    m = model()
    pose = fixed_rotations(batch=16, joints=2)
    with torch.no_grad():
        assert torch.equal(m.loss(pose), m.loss(pose))
        m.train()
        assert not torch.equal(m.loss(pose), m.loss(pose))


def test_sampling_takes_the_reverse_euler_maruyama_steps():
    # with a zero network the score is 0: x_mean = x (1 - beta dt / 2), x = x_mean + g sqrt(-dt) z; the last mean is returned
    m = model(joints=1, sample_steps=3, scale_by_sigma=False)
    with torch.no_grad():
        for p in m.net.parameters():
            p.zero_()
    torch.manual_seed(1)
    samples = m.sample(4)
    torch.manual_seed(1)
    x = torch.randn(4, 3)
    dt = -1 / 3
    for t in torch.linspace(1.0, 1e-3, 3):
        mean = x - 0.5 * m.beta(t) * x * dt
        x = mean + m.diffusion(t) * math.sqrt(-dt) * torch.randn_like(x)
    assert torch.allclose(samples, roma.rotvec_to_rotmat(mean)[:, None], atol=1e-6)


def test_wrong_arguments_are_refused():
    with pytest.raises(ValueError, match="both"):
        model(mean=[0.0] * 6)
    with pytest.raises(ValueError, match="6 entries"):
        model(mean=[0.0] * 5, std=[1.0] * 5)
    with pytest.raises(ValueError, match="multiple of 32"):
        model(hidden=48)
    with pytest.raises(ValueError, match="axis-angle"):
        model(representation="matrix")
    assert isinstance(model(), DPoser)
