"""What every model of the zoo must do. A new model is added to SMALL and has to pass all of these."""
import pytest
import torch
from conftest import fixed_rotations
from test_trainer import Poses

from hp.evaluation.evaluate import EvalConfig, evaluate
from hp.models import available_models, build_model, get_model_class
from hp.models.io import load_model, save_model
from hp.training import TrainConfig, Trainer

JOINTS = 4
SMALL = {  # arguments that make the model small enough for a test
    "so3_flow": {"flow": {"layers": 2}},
    "rsgm": {"hidden": [32, 32], "sample_steps": 5, "likelihood_steps": 5},
}
NAMES = available_models()


def small(name, **kwargs):
    return build_model(name, num_joints=JOINTS, **SMALL[name], **kwargs)


def is_rotation(r, atol=1e-4):
    eye = torch.eye(3).expand_as(r)
    return torch.allclose(r @ r.transpose(-1, -2), eye, atol=atol) and torch.allclose(r.det(), torch.ones(r.shape[:-2]), atol=atol)


def test_every_registered_model_is_in_the_contract_test():
    assert set(NAMES) <= set(SMALL), f"Add {sorted(set(NAMES) - set(SMALL))} to SMALL in tests/test_model_contract.py"


@pytest.mark.parametrize("name", NAMES)
def test_the_model_knows_its_name_and_size(name):
    model = small(name)
    assert model.name == name and type(model) is get_model_class(name)
    assert model.num_joints == JOINTS and model.num_parameters() > 0


@pytest.mark.parametrize("name", NAMES)
def test_samples_are_valid_rotations_and_follow_the_seed(name):
    model = small(name).eval()
    torch.manual_seed(0)
    first = model.sample(6)
    assert first.shape == (6, JOINTS, 3, 3) and is_rotation(first)
    torch.manual_seed(0)
    assert torch.equal(model.sample(6), first)


@pytest.mark.parametrize("name", NAMES)
def test_the_loss_is_a_finite_scalar_with_gradients(name):
    model = small(name)
    loss = model.loss(fixed_rotations(batch=16, joints=JOINTS))
    assert loss.shape == () and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("name", NAMES)
def test_the_model_trains_through_the_trainer(name):
    model = small(name)
    before = [p.detach().clone() for p in model.parameters()]
    history = Trainer(model, Poses(n=32, joints=JOINTS), Poses(n=16, joints=JOINTS),
                      TrainConfig(max_iterations=2, batch_size=16, valid_interval=2, checkpoint_every=0, progress=False)).fit()
    assert len(history.loss) == 2 and len(history.valid_loss) == 1
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


@pytest.mark.parametrize("name", NAMES)
def test_a_saved_model_loads_to_the_same_model(name, tmp_path):
    model = small(name).eval()
    save_model(model, tmp_path)
    loaded = load_model(tmp_path)
    pose = fixed_rotations(batch=8, joints=JOINTS)
    with torch.no_grad():
        assert torch.equal(loaded.loss(pose), model.loss(pose))
    torch.manual_seed(0)
    expected = model.sample(5)
    torch.manual_seed(0)
    assert torch.equal(loaded.sample(5), expected)


@pytest.mark.parametrize("name", NAMES)
def test_log_prob_exists_exactly_for_the_models_that_declare_it(name):
    model = small(name).eval()
    pose = fixed_rotations(batch=8, joints=JOINTS)
    if model.has_log_prob:
        with torch.no_grad():
            log_prob = model.log_prob(pose)
        assert log_prob.shape == (8,) and torch.isfinite(log_prob).all()
    else:
        with pytest.raises(NotImplementedError):
            model.log_prob(pose)


@pytest.mark.parametrize("name", NAMES)
def test_conditioning_is_supported_or_refused(name):
    if not get_model_class(name).supports_cond:
        with pytest.raises(ValueError, match="cannot be conditioned"):
            small(name, cond_dim=3)
        return
    model = small(name, cond_dim=3)
    cond = torch.randn(8, 3)
    assert torch.isfinite(model.loss(fixed_rotations(batch=8, joints=JOINTS), cond))
    assert model.sample(8, cond).shape == (8, JOINTS, 3, 3)


@pytest.mark.parametrize("name", NAMES)
def test_the_model_can_be_evaluated(name):
    model = small(name)
    config = EvalConfig(n_wasserstein=20, n_queries=20, n_generated=40, n_likelihood=20, repeats=1, sample_batch=20)
    report = evaluate(model, fixed_rotations(batch=64, joints=JOINTS), config)
    assert report["model"]["name"] == name and (report["likelihood"] is not None) == model.has_log_prob


@pytest.mark.parametrize("name", NAMES)
def test_a_model_with_a_log_prob_says_which_space_its_density_lives_on(name):
    cls = get_model_class(name)
    assert (cls.density_space is not None) == cls.has_log_prob
