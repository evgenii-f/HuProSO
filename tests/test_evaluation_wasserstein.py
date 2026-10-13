import numpy as np
import pytest
import roma
import torch

from hp.evaluation.wasserstein import wasserstein_distances, wasserstein_with_floor


def cloud(n, joints=3, sigma_deg=1.0, shift_deg=0.0, seed=0):
    """Poses scattered by about sigma_deg around the rotation of shift_deg about z (the same in every joint)."""
    g = torch.Generator().manual_seed(seed)
    noise = roma.rotvec_to_rotmat(torch.randn(n, joints, 3, generator=g, dtype=torch.float64) * np.deg2rad(sigma_deg))
    center = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, np.deg2rad(shift_deg)], dtype=torch.float64))
    return noise @ center


def test_the_distance_of_a_set_to_itself_is_zero():
    a = cloud(40)
    result = wasserstein_distances(a, a)
    # not exactly 0: arccos is steep at 1, so identical rotations still get an angle of about 1e-8 rad
    assert result["joint_sum_rad"] == pytest.approx(0, abs=1e-6) and np.allclose(result["per_joint_rad"], 0, atol=1e-6)


def test_a_known_shift():
    joints = 3
    result = wasserstein_distances(cloud(200, joints, seed=0), cloud(200, joints, shift_deg=20, seed=1))
    assert np.rad2deg(result["per_joint_rad"]) == pytest.approx([20] * joints, abs=1.5)
    assert np.rad2deg(result["joint_sum_rad"]) == pytest.approx(20 * joints, abs=4)


def test_the_distance_is_symmetric_and_the_pose_distance_is_at_least_the_sum_of_the_joint_distances():
    a, b = cloud(50, seed=0), cloud(60, shift_deg=15, seed=1)
    ab, ba = wasserstein_distances(a, b), wasserstein_distances(b, a)
    assert ab["joint_sum_rad"] == pytest.approx(ba["joint_sum_rad"], rel=1e-9)
    assert ab["joint_sum_rad"] >= ab["per_joint_rad"].sum() - 1e-9  # one coupling of whole poses is more constrained


def test_the_floor_is_the_distance_of_data_to_data():
    reference = cloud(400, seed=0)
    same = wasserstein_with_floor(reference, cloud(400, seed=1), n=150, rng=np.random.default_rng(0))
    ratio = same["generated"]["joint_sum_rad"] / same["floor"]["joint_sum_rad"]
    assert 0.7 < ratio < 1.4  # the same distribution is about as far from the data as the data from itself
    assert same["floor"]["joint_sum_rad"] > 0.01  # a finite sample is never at distance zero from another


def test_a_shifted_generator_is_far_above_the_floor():
    result = wasserstein_with_floor(cloud(400, seed=0), cloud(400, shift_deg=20, seed=1), n=150, rng=np.random.default_rng(0))
    assert result["generated"]["joint_sum_rad"] > 5 * result["floor"]["joint_sum_rad"]


def test_the_result_follows_the_generator_and_the_sample_size_shrinks_for_small_sets():
    reference, generated = cloud(100, seed=0), cloud(100, seed=1)
    first = wasserstein_with_floor(reference, generated, n=30, rng=np.random.default_rng(5))
    second = wasserstein_with_floor(reference, generated, n=30, rng=np.random.default_rng(5))
    assert first["generated"]["joint_sum_rad"] == second["generated"]["joint_sum_rad"]
    small = wasserstein_with_floor(cloud(10, seed=0), generated, n=30, rng=np.random.default_rng(0))
    assert small["n"] == 5  # half of the real poses


def test_different_pose_shapes_are_refused():
    with pytest.raises(ValueError, match="different shapes"):
        wasserstein_distances(cloud(5, joints=3), cloud(5, joints=4))


def test_a_contrast_set_is_compared_with_the_real_poses_too():
    reference = cloud(400, seed=0)
    same = wasserstein_with_floor(reference, cloud(400, seed=1), 150, np.random.default_rng(0), contrast=cloud(400, seed=2))
    assert same["contrast"]["joint_sum_rad"] / same["floor"]["joint_sum_rad"] < 1.5  # the same distribution
    other = wasserstein_with_floor(reference, cloud(400, seed=1), 150, np.random.default_rng(0), contrast=cloud(400, shift_deg=20, seed=2))
    assert other["contrast"]["joint_sum_rad"] > 5 * other["floor"]["joint_sum_rad"]
    assert "contrast" not in wasserstein_with_floor(reference, cloud(400, seed=1), 150, np.random.default_rng(0))
    assert wasserstein_with_floor(reference, cloud(400, seed=1), 150, np.random.default_rng(0), contrast=cloud(20, seed=2))["n"] == 20
