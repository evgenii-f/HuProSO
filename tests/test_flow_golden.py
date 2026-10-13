"""Golden values of the existing flow (the pretrained prior id_0900 on fixed inputs).

They were computed with the SO3PriorTrainer before the model zoo refactoring and pin its behaviour: if one
of these tests fails, the numerics of the flow changed.
"""
import pytest
import torch
from conftest import fixed_rotations

LOG_PROB = [-95.4081, -92.073, -127.4304, -111.0177, -90.7592, -92.5976, -85.5501, -90.4316]
LOSS = 98.1585
SAMPLE_LOG_PROB = [-195.8919, -101.8751, -131.6588, -141.6504, -184.6569, -172.7959]
SAMPLE_0_0 = [[0.9994, 0.0345, -0.0009], [-0.0345, 0.9993, -0.0132], [0.0004, 0.0132, 0.9999]]
SAMPLE_5_18 = [[0.9834, -0.1312, 0.1254], [0.1504, 0.9758, -0.1586], [-0.1016, 0.1748, 0.9794]]


def test_log_prob_of_fixed_rotations(legacy_flow):
    with torch.no_grad():
        log_prob = legacy_flow.forward_estimator(fixed_rotations())
    assert log_prob.shape == (8,)
    assert log_prob.tolist() == pytest.approx(LOG_PROB, rel=1e-4, abs=1e-2)


def test_loss_is_the_negative_mean_log_prob(legacy_flow):
    with torch.no_grad():
        loss = legacy_flow.forward_kld(fixed_rotations())
    assert loss.item() == pytest.approx(LOSS, rel=1e-4, abs=1e-2)


def test_samples_with_a_fixed_seed(legacy_flow):
    torch.manual_seed(1)
    with torch.no_grad():
        samples, log_prob = legacy_flow.sample_from_estimator(6)
    assert samples.shape == (6, 19, 3, 3)
    assert log_prob.tolist() == pytest.approx(SAMPLE_LOG_PROB, rel=1e-4, abs=1e-2)
    assert torch.allclose(samples[0, 0], torch.tensor(SAMPLE_0_0), atol=1e-3)
    assert torch.allclose(samples[5, 18], torch.tensor(SAMPLE_5_18), atol=1e-3)
    eye = torch.eye(3).expand_as(samples)  # the samples are rotation matrices
    assert torch.allclose(samples @ samples.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(samples.det(), torch.ones(6, 19), atol=1e-5)
