"""The ACE descriptors of a pose (hp.models.ace_features)."""
import math

import pytest
import roma
import torch

from hp.data.conditions import keypoints
from hp.models.ace_features import AceFeatures, _random_rotations, ace_statistics, chebyshev, legendre


@pytest.fixture(scope="module")
def ace():
    return AceFeatures()


def poses(n, seed=0):
    return _random_rotations(n, 19, torch.Generator().manual_seed(seed))


def test_polynomials():
    c = torch.linspace(-1, 1, 101, dtype=torch.float64)
    theta = torch.arccos(c)
    for k, t in enumerate(chebyshev(c, 5)):
        assert torch.allclose(t, torch.cos(k * theta), atol=1e-12)
    p1, p2, p3 = legendre(c, 3)
    assert torch.allclose(p1, c)
    assert torch.allclose(p2, (3 * c ** 2 - 1) / 2)
    assert torch.allclose(p3, (5 * c ** 3 - 3 * c) / 2)


def test_size_and_range(ace):
    assert ace.dim == len(ace.pairs) * 4 + len(ace.triples) * 2 ** 2 * 2 + len(ace.twist_joints) * 2
    assert len(ace.twist_joints) == 18  # every joint with a rotation but Spine2 (three children)
    b = ace(poses(512))
    assert b.shape == (512, ace.dim)
    assert torch.isfinite(b).all()
    assert b.abs().max() <= 1 + 1e-5  # the scaled distances stay inside [-1, 1], cos and sin of the twist too


def test_positions_are_the_keypoints(ace):
    x = poses(64)
    assert torch.allclose(ace.positions(x)[:, :22], keypoints(x), atol=1e-6)  # offsets kept in float32


def test_rigid_pairs_and_triples_are_left_out(ace):
    p = ace.positions(poses(256, seed=3))
    for j, k in ace.pairs:
        assert (p[:, k] - p[:, j]).norm(dim=-1).std() > 1e-6
    bones = {(min(i, q), max(i, q)) for i, q in enumerate(ace.parents) if q >= 0}
    assert not bones & set(ace.pairs)  # a bone has a constant length


def test_point_descriptors_are_rotation_invariant(ace):
    p = ace.positions(poses(32))
    r = roma.random_rotmat(1, dtype=torch.float64)
    assert torch.allclose(ace.point_descriptors(p), ace.point_descriptors(p @ r[0].T), atol=1e-10)


def test_twist_angle(ace):
    """x_j = swing R_a(psi): the descriptors give back psi for any swing that does not turn a into -a."""
    g = torch.Generator().manual_seed(5)
    n = 200
    psi = (torch.rand(n, dtype=torch.float64, generator=g) * 2 - 1) * math.pi
    for slot, axis in zip(ace.twist_slot.tolist(), ace.twist_axis.double()):
        swing_axis = torch.cross(axis.expand(n, 3), torch.randn(n, 3, dtype=torch.float64, generator=g), dim=-1)
        swing_axis = swing_axis / swing_axis.norm(dim=-1, keepdim=True)
        swing_angle = torch.rand(n, dtype=torch.float64, generator=g) * 2.5  # < pi
        swing = roma.rotvec_to_rotmat(swing_angle[:, None] * swing_axis)
        x = roma.rotvec_to_rotmat(torch.zeros(n, 19, 3, dtype=torch.float64))
        x = x.clone()
        x[:, slot] = swing @ roma.rotvec_to_rotmat(psi[:, None] * axis)
        c, s = ace.twist_cos_sin(x)
        k = ace.twist_slot.tolist().index(slot)
        assert torch.allclose(c[:, k], torch.cos(psi), atol=1e-6)
        assert torch.allclose(s[:, k], torch.sin(psi), atol=1e-6)


def test_twist_of_a_leaf_moves_no_point_but_a_descriptor(ace):
    """The twist of the wrist about the hand axis: invisible to the points, visible to the twist descriptors."""
    x = poses(8)
    wrist = ace.twist_joints.index(20)
    slot, axis = ace.twist_slot[wrist], ace.twist_axis[wrist].double()
    y = x.clone()
    y[:, slot] = x[:, slot] @ roma.rotvec_to_rotmat(0.7 * axis)
    assert torch.allclose(ace.positions(x), ace.positions(y), atol=1e-12)
    assert not torch.allclose(ace(x), ace(y), atol=1e-3)


def test_descriptors_separate_poses(ace):
    """Different poses give different descriptors: no two of 2000 random poses are closer than a small margin."""
    b = ace(poses(2000, seed=7))
    d = torch.cdist(b, b) + torch.eye(len(b), dtype=b.dtype) * 1e9
    assert d.min() > 1e-3


def test_gradient(ace):
    """Autograd through the descriptors agrees with finite differences along the tangent directions."""
    small = AceFeatures(radial=2, angular=2, triple_radial=2, twist=2)
    x0 = poses(2, seed=11)
    v = torch.randn(2, 19, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    w = torch.randn(small.dim, dtype=torch.float64, generator=torch.Generator().manual_seed(3))

    def f(eps):
        return (small(x0 @ roma.rotvec_to_rotmat(eps * v)) * w).sum()

    eps = torch.zeros((), dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(f(eps), eps)
    h = 1e-6
    assert torch.allclose(grad, (f(torch.tensor(h)) - f(torch.tensor(-h))) / (2 * h), rtol=1e-5, atol=1e-7)


def test_vmap_and_jacfwd(ace):
    """torch.func can differentiate the descriptors per sample (the exact divergence of RSGM's log_prob)."""
    from torch.func import jacfwd, vmap
    x = poses(3, seed=4).float()
    jac = vmap(jacfwd(lambda y: ace(y[None])[0]))(x)
    assert jac.shape == (3, ace.dim, 19, 3, 3)
    assert torch.isfinite(jac).all()


def test_statistics(ace):
    x = poses(3000, seed=9)
    mean, std = ace_statistics(ace, x, chunk=1000, std_floor=0.0)
    z = (ace(x) - torch.tensor(mean)) / torch.tensor(std)
    assert z.mean(0).abs().max() < 1e-5
    assert ((z.std(0) - 1).abs() < 1e-4).all()


def test_the_std_floor_bounds_the_z_scores(ace):
    """With the floor, a descriptor of tiny spread on the data cannot blow up the input at large noise."""
    tight = roma.rotvec_to_rotmat(0.05 * torch.randn(2000, 19, 3, dtype=torch.float64,
                                                      generator=torch.Generator().manual_seed(1)))
    mean, std = ace_statistics(ace, tight, chunk=1000, std_floor=0.1)
    assert min(std) >= 0.1
    z = (ace(poses(2000, seed=2)) - torch.tensor(mean)) / torch.tensor(std)
    assert z.abs().max() <= 2 / 0.1 + 1e-6


def test_smooth_descriptors_are_polynomials_with_bounded_gradients():
    """smooth: the same descriptors up to their form, with bounded tangent gradients also where points meet."""
    from torch.func import jacfwd, vmap
    from hp.models.rsgm import exp_frame
    smooth = AceFeatures(smooth=True)
    assert smooth.dim == AceFeatures().dim
    x = poses(200, seed=12)
    b = smooth(x)
    assert torch.isfinite(b).all() and b.abs().max() <= 1 + 1e-5
    p = smooth.positions(x)
    r = smooth.point_descriptors(p @ roma.random_rotmat(1, dtype=torch.float64)[0].T)
    assert torch.allclose(smooth.point_descriptors(p), r, atol=1e-10)  # still rotation invariant

    def jac(ace, y):
        return vmap(jacfwd(lambda d, yi: ace(exp_frame(yi, d)[None])[0]))(torch.zeros(len(y), 19, 3, dtype=y.dtype), y)

    j_smooth, j_plain = jac(smooth, x).abs().amax((1, 2, 3)), jac(AceFeatures(), x).abs().amax((1, 2, 3))
    assert j_smooth.max() < 20  # bounded by the polynomial degree and the scaled lengths
    assert j_plain.max() > j_smooth.max()
