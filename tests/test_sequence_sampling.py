"""Joint sampling of sequences with RSGM (hp/models/sequence_sampling.py)."""
import roma
import torch

from hp.models import build_model
from hp.models.rsgm import RSGM, exp_frame
from hp.models.sequence_sampling import coupling_energy, coupling_gradient, sample_sequence, velocities


def sequence(frames=6, joints=2, seed=0):
    return roma.random_rotmat(frames * joints, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).reshape(frames, joints, 3, 3)


def test_the_coupling_gradient_is_the_one_of_finite_differences():
    x = sequence()
    sigma1, sigma2 = torch.tensor([0.3, 0.5], dtype=torch.float64), 0.4
    gradient = coupling_gradient(x, sigma1, sigma2)
    numeric = torch.zeros_like(gradient)
    for index in [(0, 0, 0), (2, 1, 1), (3, 0, 2), (5, 1, 0)]:
        e = torch.zeros_like(gradient)
        e[index] = 1e-6
        numeric[index] = (coupling_energy(exp_frame(x, e), sigma1, sigma2) - coupling_energy(exp_frame(x, -e), sigma1, sigma2)) / 2e-6
        assert torch.isclose(gradient[index], numeric[index], rtol=1e-5, atol=1e-6)


def test_a_constant_sequence_has_no_coupling_energy():
    x = sequence(frames=1).expand(5, -1, -1, -1).clone()
    assert coupling_energy(x, 0.1, 0.1).item() < 1e-12
    assert torch.isfinite(coupling_gradient(x, 0.1, 0.1)).all()


class UniformPrior(RSGM):
    """RSGM whose score is 0: the uniform distribution on SO(3)^J."""

    def score(self, x, t, f=None):
        return torch.zeros(*x.shape[:-2], 3, dtype=x.dtype, device=x.device)


def test_the_coupling_makes_a_sequence_of_a_uniform_prior_smooth():
    """From pure noise the chain is smooth, though its steps are somewhat too large (see the module doc): the exact
    mean step of the random walk is 0.05 * sqrt(8 / pi) = 0.08."""
    torch.manual_seed(0)
    model = UniformPrior(2, hidden=[4], beta_f=10.0, sample_steps=100).double()
    free = sample_sequence(model, 40, sigma1=1e6)              # no coupling: independent uniform frames
    coupled = sample_sequence(model, 40, sigma1=0.05)
    step = lambda x: velocities(x).norm(dim=-1).mean().item()
    assert step(free) > 1.5                                      # about the mean angle of a uniform rotation (2.13)
    assert 0.05 < step(coupled) < 0.15


def test_from_a_coherent_start_the_sequence_stays_coherent():
    """Started at t_start from a constant sequence, the steps between frames are those of the random walk."""
    torch.manual_seed(0)
    model = UniformPrior(2, hidden=[4], beta_f=10.0, sample_steps=100).double()
    start = roma.random_rotmat(2, dtype=torch.float64)[None].expand(60, -1, -1, -1)
    x = sample_sequence(model, 60, sigma1=0.05, start=start, t_start=0.1)
    assert 0.05 < velocities(x).norm(dim=-1).mean().item() < 0.11


def test_a_conditional_model_takes_one_condition_per_frame():
    torch.manual_seed(0)
    model = build_model("rsgm", num_joints=19, hidden=[16], condition="3D", sample_steps=3)
    cond = torch.randn(7, 66)
    cond[3:, :9] = float("nan")                                  # hidden keypoints in some frames
    x = sample_sequence(model, 7, cond, sigma1=0.1, sigma2=0.1)
    assert x.shape == (7, 19, 3, 3) and torch.isfinite(x).all()
    start = model.sample_ode(7, cond, start=roma.random_rotmat(19)[None].expand(7, -1, -1, -1))
    assert torch.isfinite(sample_sequence(model, 7, cond, sigma1=0.1, start=start, t_start=0.2)).all()
