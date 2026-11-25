"""Harmonics of SO(3) (hp.models.so3_harmonics); the tests of experimental/wigner-inputs, moved with the code."""
import math

import pytest
import roma
import torch

from hp.data.amass_cfg import AmassJoints4Learning
from hp.models.rsgm import global_rotations
from hp.models.so3_harmonics import WIGNER_PAIRS, pair_wigner_features, wigner_d, wigner_features

J19 = len(AmassJoints4Learning)


def poses19(n, seed=0):
    return roma.random_rotmat(n * J19, generator=torch.Generator().manual_seed(seed),
                              dtype=torch.float64).reshape(n, J19, 3, 3)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_the_wigner_matrices_are_an_orthogonal_representation(l):
    g = torch.Generator().manual_seed(l)
    r1, r2 = (roma.random_rotmat(500, generator=g, dtype=torch.float64) for _ in range(2))
    d1, d2, d12 = wigner_d(r1, l), wigner_d(r2, l), wigner_d(r1 @ r2, l)
    eye = torch.eye(2 * l + 1, dtype=torch.float64)
    assert d1.shape == (500, 2 * l + 1, 2 * l + 1)
    assert torch.allclose(d12, d1 @ d2, atol=1e-10)  # D(R1 R2) = D(R1) D(R2)
    assert torch.allclose(d1 @ d1.transpose(-1, -2), eye.expand_as(d1), atol=1e-10)
    assert torch.allclose(wigner_d(torch.eye(3, dtype=torch.float64), l), eye, atol=1e-10)


def test_the_first_wigner_matrix_is_the_rotation_in_the_order_y_z_x():
    r = roma.random_rotmat(100, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    order = [1, 2, 0]  # the real harmonics of degree 1 are y, z, x
    assert torch.allclose(wigner_d(r, 1), r[:, order][:, :, order], atol=1e-10)


def test_the_wigner_entries_are_orthogonal_over_the_haar_measure():
    """Schur orthogonality: E[D^l_mn D^l'_m'n'] = delta / (2l + 1) for uniform rotations, so the entries of different
    degrees are uncorrelated and none is a function of the others."""
    n = 200_000
    r = roma.random_rotmat(n, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    f = torch.cat([wigner_d(r, l).reshape(n, -1) * math.sqrt(2 * l + 1) for l in (1, 2, 3)], dim=1)  # [n, 83]
    gram = f.T @ f / n
    assert torch.allclose(gram, torch.eye(83, dtype=torch.float64), atol=0.03)


def test_the_joint_features_are_the_wigner_matrices_of_the_joints():
    x = poses19(3)
    f = wigner_features(x, (2, 3))
    assert f.shape == (3, J19 * (25 + 49))
    assert torch.allclose(f[:, :J19 * 25].reshape(3, J19, 5, 5), wigner_d(x, 2), atol=1e-12)


def test_the_pair_harmonics_are_the_contracted_products_of_the_two_joints():
    """D^l(G_i^T G_k) = D^l(G_i)^T D^l(G_k) = sum_m D^l_mn(G_i) D^l_mn'(G_k): the two-body products of the joints'
    harmonics, contracted; the joints are the named pairs and the root."""
    x = poses19(4)
    g = global_rotations(x)
    f = pair_wigner_features(g, (1, 2, 3))
    assert f.shape == (4, (J19 + len(WIGNER_PAIRS)) * (9 + 25 + 49))
    names = [j.name for j in AmassJoints4Learning]
    a, b = names.index("LeftLeg"), names.index("RightLeg")
    p = len(WIGNER_PAIRS)  # the layout: the joints' blocks of l = 1, 2, 3, then the pairs' blocks of l = 1, 2, 3
    joints2 = f[:, J19 * 9:J19 * (9 + 25)].reshape(4, J19, 5, 5)
    start = J19 * 83 + p * 9
    pairs2 = f[:, start:start + p * 25].reshape(4, p, 5, 5)
    knees = list(WIGNER_PAIRS).index(("LeftLeg", "RightLeg"))
    assert torch.allclose(pairs2[:, knees], wigner_d(g[:, a], 2).transpose(-1, -2) @ wigner_d(g[:, b], 2), atol=1e-10)
    assert torch.allclose(joints2[:, a], wigner_d(g[:, a], 2), atol=1e-10)  # a joint with the root: its own G


def test_torch_func_differentiates_the_harmonics():
    from torch.func import jacfwd, vmap
    x = poses19(2)
    jac = vmap(jacfwd(lambda y: wigner_features(y[None], (3,))[0]))(x)
    assert jac.shape == (2, J19 * 49, J19, 3, 3) and torch.isfinite(jac).all()
