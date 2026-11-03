"""The dNN of the PoseD-Flow paper (hp/evaluation/posedflow_protocol.py)."""
import math

import pytest
import roma
import torch

from hp.evaluation.posedflow_protocol import confidence, nearest_neighbour_distance


def test_dnn_is_zero_on_training_poses_and_half_the_angle_otherwise():
    torch.manual_seed(0)
    train = roma.random_rotmat(300 * 5, dtype=torch.float64).reshape(300, 5, 3, 3)
    assert nearest_neighbour_distance(train[:20], train, query_chunk=7, train_chunk=50).abs().max() < 1e-6
    # rotate one joint of a pose by 0.2 rad: its distance is 0.1 (half-angle) / 5 joints, if nothing else is nearer
    moved = train[:1].clone()
    moved[0, 2] = moved[0, 2] @ roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64))
    assert nearest_neighbour_distance(moved, train).item() == pytest.approx(0.1 / 5, abs=1e-6)


def test_dnn_takes_quaternions_too():
    train = roma.random_rotmat(50 * 3, dtype=torch.float64).reshape(50, 3, 3, 3)
    poses = roma.random_rotmat(4 * 3, dtype=torch.float64).reshape(4, 3, 3, 3)
    assert torch.allclose(nearest_neighbour_distance(poses, train), nearest_neighbour_distance(poses, roma.rotmat_to_unitquat(train)))


def test_confidence_is_the_t_interval():
    c = confidence([1.0, 2.0, 3.0])
    assert c["mean"] == 2.0 and c["ci95"] == pytest.approx(4.302653 * 1 / math.sqrt(3), rel=1e-5)


from hp.evaluation.posedflow_protocol import SmplSkeleton, apd, fid, frechet_distance  # noqa: E402


def skeleton(tmp_path):
    torch.manual_seed(3)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    import numpy as np
    np.savez(tmp_path / "s.npz", joints=torch.randn(24, 3).numpy(), parents=np.array(parents))
    return SmplSkeleton(tmp_path / "s.npz")


def reference_fk(sk, rot21):
    """smplx's batch_rigid_transform with homogeneous transforms, for the first 22 joints."""
    b = len(rot21)
    rots = torch.cat([torch.eye(3, dtype=torch.float64).expand(b, 1, 3, 3), rot21], 1)
    rel = sk.rest.clone()
    rel[1:] = sk.rest[1:] - sk.rest[sk.parents[1:]]
    t = torch.zeros(b, 22, 4, 4, dtype=torch.float64)
    t[:, :, :3, :3], t[:, :, :3, 3], t[:, :, 3, 3] = rots, rel, 1
    chain = [t[:, 0]]
    for j in range(1, 22):
        chain.append(chain[sk.parents[j]] @ t[:, j])
    return torch.stack(chain, 1)[:, :, :3, 3]


def test_fk_matches_the_rigid_transforms_of_smplx(tmp_path):
    sk = skeleton(tmp_path)
    rot = roma.random_rotmat(5 * 21, dtype=torch.float64).reshape(5, 21, 3, 3)
    assert torch.allclose(sk.joints(rot), reference_fk(sk, rot), atol=1e-12)
    assert torch.allclose(sk.joints(torch.eye(3).repeat(2, 21, 1, 1)), sk.rest.expand(2, 22, 3).double(), atol=1e-6)


def test_the_toes_move_no_joint(tmp_path):
    sk = skeleton(tmp_path)
    rot = roma.random_rotmat(3 * 21, dtype=torch.float64).reshape(3, 21, 3, 3)
    keep = [j for j in range(21) if j not in (9, 10)]
    assert torch.allclose(sk.joints(rot), sk.joints(rot[:, keep]), atol=1e-12)


def test_fid_and_apd():
    torch.manual_seed(4)
    x = torch.randn(400, 22, 3, dtype=torch.float64)
    flat = x.reshape(400, -1).numpy()
    import numpy as np
    from scipy.spatial import distance
    assert fid(x, flat.mean(0), np.cov(flat, rowvar=False)) == pytest.approx(0.0, abs=1e-6)
    assert frechet_distance(np.zeros(2), np.eye(2), np.ones(2), np.eye(2)) == pytest.approx(2.0)
    small = x[:30].numpy()
    ref = distance.pdist(small.reshape(30, -1), lambda u, v: np.sqrt(((u.reshape(22, 3) - v.reshape(22, 3)) ** 2).sum(1)).mean()).mean()
    assert apd(x[:30]) == pytest.approx(ref * 100)
