import pytest
import torch
from conftest import PRETRAINED, fixed_rotations
from omegaconf import OmegaConf

from hp.models import available_models, build_model
from hp.models.so3_flow import PAPER_FLOW_CONFIG, SO3Flow

PAPER_PARAMETERS = 1_293_104


@pytest.fixture(scope="module")
def flow():
    """The wrapper with the weights of the pretrained prior id_0900."""
    model = build_model("so3_flow", num_joints=19)
    model.load_legacy_checkpoint(PRETRAINED)
    return model.eval()


def test_the_default_flow_is_the_paper_architecture():
    assert "so3_flow" in available_models()
    assert build_model("so3_flow", num_joints=19).num_parameters() == PAPER_PARAMETERS
    assert dict(OmegaConf.load(PRETRAINED / "config.yaml").so3) == PAPER_FLOW_CONFIG


def test_log_prob_and_loss_match_the_existing_trainer(flow, legacy_flow):
    x = fixed_rotations()
    with torch.no_grad():
        assert torch.equal(flow.log_prob(x), legacy_flow.forward_estimator(x))
        assert torch.equal(flow.loss(x), legacy_flow.forward_kld(x))


def test_samples_match_the_existing_trainer(flow, legacy_flow):
    torch.manual_seed(1)
    expected, expected_log_prob = legacy_flow.sample_from_estimator(6)
    torch.manual_seed(1)
    samples, log_prob = flow.sample_with_log_prob(6)
    assert torch.equal(samples, expected) and torch.equal(log_prob, expected_log_prob)
    torch.manual_seed(1)
    assert torch.equal(flow.sample(6), expected)  # the interface method gives the same samples


def test_state_dict_carries_the_dimension_order(flow, tmp_path):
    torch.save(flow.state_dict(), tmp_path / "model.pt")
    other = build_model("so3_flow", num_joints=19)  # starts with a different random order of the dimensions
    other.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    x = fixed_rotations()
    with torch.no_grad():
        assert torch.equal(other.eval().log_prob(x), flow.log_prob(x))


def test_a_small_flow_trains():
    torch.manual_seed(0)
    model = build_model("so3_flow", num_joints=4, flow={"layers": 2})
    x = fixed_rotations(batch=64, joints=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = model.loss(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(map(torch.isfinite, map(torch.tensor, losses))) and losses[-1] < losses[0]


def test_a_single_joint_flow():
    model = build_model("so3_flow", num_joints=1, flow={"layers": 2}).eval()
    x = fixed_rotations(batch=5, joints=1)
    assert model.log_prob(x).shape == (5,)
    assert model.sample(3).shape == (3, 1, 3, 3)


def test_the_flow_is_unconditional_and_works_on_matrices():
    with pytest.raises(ValueError, match="cannot be conditioned"):
        SO3Flow(num_joints=19, cond_dim=5)
    with pytest.raises(ValueError, match="matrix"):
        SO3Flow(num_joints=19, representation="6d")
    assert SO3Flow.has_log_prob and not SO3Flow.supports_cond
