import numpy as np
import pytest
import roma
import torch
from conftest import fixed_rotations

from hp.evaluation.geodesic import flatten_joints, pose_distances
from hp.evaluation.neighbours import nearest_distances, summarize
from hp.utils.rotation_tools import batch_geodesic_distance_matrix


def rotations64(batch, joints, seed=0):
    """Random rotations in float64, orthogonal to 1e-16 (float32 ones are only orthogonal to about 1e-7)."""
    torch.manual_seed(seed)
    return roma.random_rotmat(batch * joints, dtype=torch.float64).reshape(batch, joints, 3, 3)


def brute_force(queries, candidates, reduction):
    """The matrix of distances by the geodesic helper of the repository (numpy, one joint at a time)."""
    total = sum(batch_geodesic_distance_matrix(queries[:, j].numpy().astype(np.float64), candidates[:, j].numpy().astype(np.float64))
                for j in range(queries.shape[1]))
    return total / queries.shape[1] if reduction == "mean" else total


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_distances_equal_the_geodesic_helper_of_the_repository(reduction):
    a, b = fixed_rotations(batch=7, joints=5, seed=0), fixed_rotations(batch=9, joints=5, seed=1)
    ours = pose_distances(flatten_joints(a), flatten_joints(b), reduction).numpy()
    np.testing.assert_allclose(ours, brute_force(a, b, reduction), atol=2e-3)  # float32 near the ends of arccos


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_nearest_distances_are_exact(reduction):
    queries, candidates = fixed_rotations(batch=20, joints=5, seed=0), fixed_rotations(batch=50, joints=5, seed=1)
    expected = brute_force(queries, candidates, reduction).min(axis=1)
    np.testing.assert_allclose(nearest_distances(queries, candidates, reduction).numpy(), expected, atol=2e-3)


def test_the_result_does_not_depend_on_the_chunk_size():
    queries, candidates = rotations64(10, 4, seed=0), rotations64(101, 4, seed=1)
    reference = nearest_distances(queries, candidates, chunk=1000, dtype=torch.float64)
    for chunk in (1, 7, 100, 101):
        torch.testing.assert_close(nearest_distances(queries, candidates, chunk=chunk, dtype=torch.float64), reference, rtol=0, atol=1e-9)


def test_a_pose_among_the_candidates_has_distance_zero():
    poses = rotations64(30, 6)
    assert nearest_distances(poses[:10], poses, dtype=torch.float64).max() < 1e-6
    single = fixed_rotations(batch=30, joints=6)  # float32 poses: resolved down to about 0.03 degrees
    assert nearest_distances(single[:10], single).max() < np.deg2rad(0.1)


def test_more_candidates_never_make_it_farther():
    queries, candidates = fixed_rotations(batch=10, joints=4, seed=0), fixed_rotations(batch=60, joints=4, seed=1)
    few, many = nearest_distances(queries, candidates[:20]), nearest_distances(queries, candidates)
    assert (many <= few).all()


def test_a_known_angle():
    angle = np.deg2rad(10)
    rotated = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, angle])).expand(1, 6, 3, 3)
    identity = torch.eye(3).expand(1, 6, 3, 3)
    assert nearest_distances(rotated, identity, dtype=torch.float64).item() == pytest.approx(angle, abs=1e-6)
    assert nearest_distances(rotated, identity, reduction="sum", dtype=torch.float64).item() == pytest.approx(6 * angle, abs=1e-6)


def test_different_pose_shapes_and_reductions_are_refused():
    with pytest.raises(ValueError, match="different shapes"):
        nearest_distances(fixed_rotations(joints=4), fixed_rotations(joints=5))
    with pytest.raises(ValueError, match="reduction"):
        nearest_distances(fixed_rotations(), fixed_rotations(), reduction="max")


def test_summary_in_degrees():
    summary = summarize(np.deg2rad([1.0, 2.0, 3.0, 10.0]))
    assert summary["mean_deg"] == pytest.approx(4.0) and summary["median_deg"] == pytest.approx(2.5)
    assert summary["within_5deg"] == 0.75 and summary["p95_deg"] == pytest.approx(np.percentile([1, 2, 3, 10], 95))
