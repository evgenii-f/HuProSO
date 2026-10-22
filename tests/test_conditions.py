"""Keypoint conditions (hp/data/conditions.py)."""
import pytest
import roma
import torch

from hp.data.amass_cfg import AmassJoints4Learning
from hp.data.conditions import condition_dim, keypoints
from hp.utils.skeleton_forward_kinematics import perform_forward_kinematics

J = len(AmassJoints4Learning)


def random_poses(n, seed=0, dtype=torch.float32):
    return roma.random_rotmat(n * J, generator=torch.Generator().manual_seed(seed), dtype=dtype).reshape(n, J, 3, 3)


def test_keypoints_are_the_forward_kinematics_of_the_papers_code():
    poses = random_poses(200)
    assert torch.allclose(keypoints(poses), perform_forward_kinematics(poses, selected_joints=AmassJoints4Learning), atol=1e-5)


def test_the_rest_pose_has_its_root_at_the_origin_and_2d_is_x_and_y():
    rest = torch.eye(3).expand(1, J, 3, 3)
    p = keypoints(rest)
    assert p.shape == (1, 22, 3) and torch.equal(p[0, 0], torch.zeros(3))
    assert torch.equal(keypoints(rest, "2D"), p[..., :2])
    assert condition_dim("3D") == 66 and condition_dim("2D") == 44


def test_keypoints_keep_dtype_and_are_differentiable():
    poses = random_poses(4, dtype=torch.float64).requires_grad_()
    p = keypoints(poses)
    assert p.dtype == torch.float64
    p.sum().backward()
    assert torch.isfinite(poses.grad).all()


def test_wrists_and_head_do_not_move_any_keypoint():
    poses = random_poses(3)
    names = [j.name for j in AmassJoints4Learning]
    changed = poses.clone()
    for name in ("LeftHand", "RightHand", "Head"):
        changed[:, names.index(name)] = random_poses(3, seed=1)[:, 0]
    assert torch.allclose(keypoints(changed), keypoints(poses), atol=1e-6)


def test_unknown_modality_and_wrong_shapes_are_refused():
    with pytest.raises(ValueError):
        keypoints(random_poses(1), "1D")
    with pytest.raises(ValueError):
        keypoints(torch.eye(3).expand(1, 21, 3, 3))
