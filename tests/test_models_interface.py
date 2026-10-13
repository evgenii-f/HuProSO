import pytest
import torch
from torch import nn

from hp.models import PoseModel, available_models, build_model, get_model_class, register_model
from hp.models import registry


class Gaussian(PoseModel):
    """A trivial model for the tests: an independent Gaussian in the representation space."""

    supports_cond = True

    def __init__(self, num_joints, representation="6d", cond_dim=0):
        super().__init__(num_joints, representation, cond_dim)
        d = num_joints * self.representation.dim
        self.mean = nn.Parameter(torch.zeros(d))
        self.log_std = nn.Parameter(torch.zeros(d))

    def loss(self, pose, cond=None):
        self.check_cond(cond, len(pose))
        z = self.encode(pose)
        return (0.5 * ((z - self.mean) / self.log_std.exp()) ** 2 + self.log_std).mean()

    def sample(self, n, cond=None):
        self.check_cond(cond, n)
        z = self.mean + self.log_std.exp() * torch.randn(n, len(self.mean), device=self.device)
        return self.decode(z)


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch):
    """Every test starts with an empty registry, and the real models stay untouched."""
    monkeypatch.setattr(registry, "_REGISTRY", {})


def test_register_and_build():
    register_model("gaussian")(Gaussian)
    assert available_models() == ["gaussian"] and get_model_class("gaussian") is Gaussian
    model = build_model("gaussian", num_joints=19, representation="axis_angle")
    assert isinstance(model, Gaussian) and model.name == "gaussian"
    assert model.num_joints == 19 and model.representation.name == "axis_angle"
    assert model.build_args == {"name": "gaussian", "num_joints": 19, "representation": "axis_angle"}


def test_registering_a_name_twice_fails():
    register_model("gaussian")(Gaussian)
    with pytest.raises(ValueError, match="already registered"):
        register_model("gaussian")(Gaussian)


def test_only_pose_models_can_be_registered():
    with pytest.raises(TypeError):
        register_model("not_a_model")(nn.Linear)


def test_unknown_name_lists_the_available_models():
    register_model("gaussian")(Gaussian)
    with pytest.raises(ValueError, match="gaussian"):
        build_model("nope", num_joints=19)


def test_a_model_must_implement_loss_and_sample():
    class Incomplete(PoseModel):
        def loss(self, pose, cond=None):
            return pose.sum()

    with pytest.raises(TypeError):
        Incomplete(num_joints=19)


def test_log_prob_is_optional():
    model = Gaussian(num_joints=19)
    assert not model.has_log_prob
    with pytest.raises(NotImplementedError, match="no exact log-likelihood"):
        model.log_prob(torch.eye(3).expand(2, 19, 3, 3))


def test_loss_and_samples():
    torch.manual_seed(0)
    model = Gaussian(num_joints=19, representation="6d")
    pose = model.decode(torch.randn(16, 19 * 6))
    assert model.loss(pose).shape == ()
    samples = model.sample(5)
    assert samples.shape == (5, 19, 3, 3)
    assert torch.allclose(samples @ samples.transpose(-1, -2), torch.eye(3).expand_as(samples), atol=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.loss(pose).item()
    for _ in range(20):
        optimizer.zero_grad()
        model.loss(pose).backward()
        optimizer.step()
    assert model.loss(pose).item() < before  # the loss can be minimised


def test_encode_and_decode_round_trip():
    model = Gaussian(num_joints=19, representation="6d")
    pose = model.decode(torch.randn(4, 19 * 6))
    assert model.encode(pose).shape == (4, 19 * 6)
    assert torch.allclose(model.decode(model.encode(pose)), pose, atol=1e-4)


def test_conditioning_is_checked():
    plain = Gaussian(num_joints=19)
    conditional = Gaussian(num_joints=19, cond_dim=5)
    pose = conditional.decode(torch.randn(3, 19 * 6))
    with pytest.raises(ValueError, match="not conditional"):
        plain.loss(pose, cond=torch.zeros(3, 5))
    with pytest.raises(ValueError, match="needs a condition"):
        conditional.loss(pose)
    with pytest.raises(ValueError, match="shape"):
        conditional.loss(pose, cond=torch.zeros(3, 4))
    assert conditional.loss(pose, cond=torch.zeros(3, 5)).shape == ()
    assert conditional.sample(3, cond=torch.zeros(3, 5)).shape == (3, 19, 3, 3)


def test_a_model_that_cannot_be_conditioned_refuses_cond_dim():
    class Unconditional(Gaussian):
        supports_cond = False

    with pytest.raises(ValueError, match="cannot be conditioned"):
        Unconditional(num_joints=19, cond_dim=5)


def test_device_and_parameter_count():
    model = Gaussian(num_joints=2, representation="axis_angle")
    assert model.device.type == "cpu" and model.num_parameters() == 2 * 3 * 2
