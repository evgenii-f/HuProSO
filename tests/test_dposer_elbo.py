"""DPoser on the ELBO with a learned noise (hp/models/dposer_elbo.py, docs/elbo_noise_plan.md)."""
import math

import pytest
import torch
from conftest import fixed_rotations

from hp.models import build_model

torch.set_default_dtype(torch.float32)

D = [0.5, 2.0, 1.0, 3.0, 0.25, 1.5]


def elbo_model(joints=2, **kwargs):
    torch.manual_seed(0)
    args = {"hidden": 32, "embed": 16, "blocks": 1, "sample_steps": 5, **kwargs}
    return build_model("dposer_elbo", num_joints=joints, **args).eval()


def test_the_published_paths_refuse_the_learned_noise():
    from hp.models.dflow import DPoserPrior
    from hp.models.inpainting import dposer_inpaint

    m = elbo_model()
    assert not m.isotropic_noise
    with pytest.raises(NotImplementedError):
        DPoserPrior(m)
    with pytest.raises(NotImplementedError):
        dposer_inpaint(m, fixed_rotations(batch=2, joints=2), torch.tensor([True, False]), steps=2, replace=True)


# ---- the ELBO and the learned noise (docs/elbo_noise_plan.md) ----

def test_gamma_its_derivative_and_its_inverse():
    m = elbo_model().double()
    t = torch.tensor([1e-5, 1e-3, 0.1, 0.5, 1.0], dtype=torch.float64)
    alpha, std = m.marginal(t)
    assert torch.allclose(m.gamma(t), torch.log(std ** 2 / alpha ** 2))
    h = 1e-6
    numeric = (m.gamma(t[1:] + h) - m.gamma(t[1:] - h)) / (2 * h)
    assert torch.allclose(m.gamma_prime(t[1:]), numeric, rtol=1e-5)
    assert torch.allclose(m.time_of_gamma(m.gamma(t)), t, rtol=1e-9)
    t32 = t.float()
    assert torch.allclose(elbo_model().time_of_gamma(elbo_model().gamma(t32)), t32, rtol=1e-5)


def test_learned_noise_starts_at_dposer_and_keeps_the_mean_square_of_d():
    m = elbo_model()
    assert torch.allclose(m.noise_d, torch.ones(6)) and not m.isotropic_noise
    assert "noise_logits" in m.state_dict() and "noise_d" not in m.state_dict()
    with torch.no_grad():
        m.noise_logits.copy_(torch.randn(6))
    assert torch.allclose(m.noise_d.pow(2).mean(), torch.tensor(1.0))
    m.train()
    m.loss(fixed_rotations(batch=16, joints=2)).backward()
    assert m.noise_logits.grad is not None and m.noise_logits.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="not both"):
        elbo_model(noise_scale=D)
    fixed = elbo_model(learn_noise=False, noise_scale=D)
    assert torch.equal(fixed.noise_d, torch.tensor(D)) and "noise_logits" not in fixed.state_dict()


def gaussian_elbo(m, cov, n, denoiser_cov=None, seed=0):
    """E over x_0 ~ N(0, cov) of -ELBO with the exact noise prediction E[eps | x_t] of N(0, denoiser_cov) (default cov):
    the exact one, or one that ignores the correlations (denoiser_cov = diag(cov))."""
    dc = cov if denoiser_cov is None else denoiser_cov

    def prediction(x, t):
        alpha, std = m.marginal(t)
        sd = std[:, None] * m.noise_d                                                       # [B, n]
        var = alpha[:, None, None] ** 2 * dc + torch.diag_embed(sd ** 2)                    # Cov(x_t)
        return sd * torch.linalg.solve(var, x[..., None])[..., 0]                           # sD Cov^-1 x_t

    m.noise_prediction = prediction
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, cov.shape[0], dtype=torch.float64, generator=g) @ torch.linalg.cholesky(cov).T
    return sum(term.mean() for term in m.elbo_terms(x0, g)).item()


COV = torch.tensor([[1.0, 0.8, 0.0], [0.8, 1.0, 0.4], [0.0, 0.4, 1.0]], dtype=torch.float64)


@pytest.mark.parametrize("d2", [[1.0, 1.0, 1.0], [0.1, 0.4, 2.5], [2.8, 0.1, 0.1]])
def test_with_the_exact_denoiser_the_elbo_is_the_entropy_for_any_noise(d2):
    """Section 4 of the plan: the vector I-MMSE relation makes the ELBO of the optimal denoiser independent of d, also
    for correlated data, and equal to -log p (here its mean, the entropy), up to the boundaries."""
    m = elbo_model(joints=1).double()
    with torch.no_grad():
        m.noise_logits.copy_(torch.log(torch.tensor(d2, dtype=torch.float64)))
    entropy = 0.5 * torch.logdet(2 * math.pi * math.e * COV).item()
    assert gaussian_elbo(m, COV, 200_000) == pytest.approx(entropy, abs=0.4)


def test_an_exact_denoiser_of_another_distribution_is_also_invariant_to_the_noise():
    """The denoiser that ignores the correlations is the exact one of N(0, diag cov): its field is conservative too, so
    its ELBO does not depend on d either; it is the cross-entropy of that model, above the entropy."""
    m = elbo_model(joints=1).double()
    diagonal = torch.diag(torch.diag(COV))
    inverse = torch.linalg.solve(diagonal, COV)
    cross_entropy = (0.5 * torch.logdet(2 * math.pi * diagonal) + 0.5 * torch.trace(inverse)).item()
    for d2 in ([1.0, 1.0, 1.0], [0.1, 0.1, 2.8]):
        with torch.no_grad():
            m.noise_logits.copy_(torch.log(torch.tensor(d2, dtype=torch.float64)))
        assert gaussian_elbo(m, COV, 200_000, denoiser_cov=diagonal) == pytest.approx(cross_entropy, abs=0.4)


def test_a_denoiser_of_limited_form_makes_the_elbo_depend_on_the_noise():
    """MuLAN's point: a denoiser that is not the posterior mean of any distribution (here one shrinkage factor for all
    coordinates, the best of its form) gives a path-dependent ELBO: d is learned through what the network can do."""
    m = elbo_model(joints=1).double()

    def shared(x, t):
        alpha, std = m.marginal(t)
        sd = std[:, None] * m.noise_d
        return (sd.sum(-1) / (alpha ** 2 * torch.trace(COV) + (sd ** 2).sum(-1)))[:, None] * x

    values = []
    for d2 in ([1.0, 1.0, 1.0], [0.5, 0.5, 2.0], [0.1, 0.1, 2.8]):
        with torch.no_grad():
            m.noise_logits.copy_(torch.log(torch.tensor(d2, dtype=torch.float64)))
        m.noise_prediction = shared
        g = torch.Generator().manual_seed(0)
        x0 = torch.randn(200_000, 3, dtype=torch.float64, generator=g) @ torch.linalg.cholesky(COV).T
        values.append(sum(term.mean() for term in m.elbo_terms(x0, g)).item())
    assert values[0] + 1.0 < values[1] < values[2] - 1.0   # this family is best with the noise matched to it, d = 1
