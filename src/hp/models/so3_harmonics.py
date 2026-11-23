"""Harmonics of SO(3) for features and energies of poses: real spherical harmonics, real Wigner D-matrices, and their
contracted two-body products for pairs of joints. Shared by RSGM's Wigner inputs (experimental/wigner-inputs) and the
ACE energies (explore/ace_energy, experimental/ace-descriptors).

    d = wigner_d(r, 2)                        # [..., 5, 5]: D^2(r) of rotations r [..., 3, 3]
    f = wigner_features(x, (2, 3))            # [B, J * (25 + 49)]: D^l(x_j) of every joint
    p = pair_wigner_features(g, (1, 2, 3))    # [B, (J + 11) * 83]: D^l(G_j) and D^l(G_i^T G_k) of WIGNER_PAIRS

The real basis is that of the orthonormal real spherical harmonics Y_lm (m = -l..l; for l = 1 the order y, z, x).
wigner_d solves Y_l(R p) = D^l(R) Y_l(p) on fixed points p of the sphere: exact (the harmonics of degree l span a
space that rotations map to itself), polynomials of degree l in the entries of R, orthogonal, D^l(R1 R2) =
D^l(R1) D^l(R2), and D^1(R) is R itself with rows and columns in the order y, z, x. For a pair of joints with
rotations G_i, G_k relative to the root, D^l(G_i^T G_k) = D^l(G_i)^T D^l(G_k): the products of the harmonics of the
two joints contracted over m, the two-body terms of the atomic cluster expansion. Plain out-of-place ops, so that
torch.func can vmap and differentiate everything here.
"""
import math
from typing import Sequence

import torch

from hp.data.amass_cfg import AmassJoints4Learning


def real_harmonics(v: torch.Tensor, l: int) -> torch.Tensor:
    """The orthonormal real spherical harmonics Y_lm, m = -l..l, of unit vectors v [..., 3, K], as [..., 2l+1, K]
    (l = 1, 2, 3)."""
    x, y, z = v[..., 0, :], v[..., 1, :], v[..., 2, :]
    pi = math.pi
    if l == 1:
        c = math.sqrt(3 / (4 * pi))
        ys = [c * y, c * z, c * x]
    elif l == 2:
        a, b = 0.5 * math.sqrt(15 / pi), 0.25 * math.sqrt(5 / pi)
        ys = [a * x * y, a * y * z, b * (3 * z * z - 1), a * x * z, 0.5 * a * (x * x - y * y)]
    elif l == 3:
        a, b = 0.25 * math.sqrt(35 / (2 * pi)), 0.5 * math.sqrt(105 / pi)
        c, d = 0.25 * math.sqrt(21 / (2 * pi)), 0.25 * math.sqrt(7 / pi)
        ys = [a * y * (3 * x * x - y * y), b * x * y * z, c * y * (5 * z * z - 1), d * z * (5 * z * z - 3),
              c * x * (5 * z * z - 1), 0.5 * b * z * (x * x - y * y), a * x * (x * x - 3 * y * y)]
    else:
        raise ValueError(f"real_harmonics has the degrees 1, 2 and 3, got {l}.")
    return torch.stack(ys, dim=-2)


# fixed points on the sphere and the pseudo-inverses of their harmonics [K, 2l+1], for wigner_d
_WIGNER_POINTS = torch.nn.functional.normalize(
    torch.randn(3, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(0)), dim=0)
_WIGNER_PINV = {l: torch.linalg.pinv(real_harmonics(_WIGNER_POINTS, l)) for l in (1, 2, 3)}


def wigner_d(r: torch.Tensor, l: int) -> torch.Tensor:
    """The real Wigner D-matrix D^l(r) [..., 2l+1, 2l+1] of rotations r [..., 3, 3]: Y_l(r p) = D^l(r) Y_l(p) for
    every point p of the sphere, solved on fixed points (module doc)."""
    p = _WIGNER_POINTS.to(device=r.device, dtype=r.dtype)
    return real_harmonics(r @ p, l) @ _WIGNER_PINV[l].to(device=r.device, dtype=r.dtype)


def wigner_features(x: torch.Tensor, degrees: Sequence[int]) -> torch.Tensor:
    """The entries of D^l(x_j) for l in degrees, [B, J * sum (2l+1)^2], of rotations x [B, J, 3, 3]."""
    return torch.cat([wigner_d(x, l).reshape(x.shape[0], -1) for l in degrees], dim=1)


# pairs of joints (AmassJoints4Learning names) that are not parent and child: left and right, and the siblings that
# are not left and right (the children of the root and of Spine2)
WIGNER_PAIRS = (("LeftUpLeg", "RightUpLeg"), ("LeftLeg", "RightLeg"), ("LeftFoot", "RightFoot"),
                ("LeftShoulder", "RightShoulder"), ("LeftArm", "RightArm"), ("LeftElbow", "RightElbow"),
                ("LeftHand", "RightHand"), ("LeftUpLeg", "Spine0"), ("RightUpLeg", "Spine0"),
                ("Neck", "LeftShoulder"), ("Neck", "RightShoulder"))
_NAMES = [joint.name for joint in AmassJoints4Learning]
PAIR_INDEX = tuple((_NAMES.index(a), _NAMES.index(b)) for a, b in WIGNER_PAIRS)


def pair_wigner_features(g: torch.Tensor, degrees: Sequence[int]) -> torch.Tensor:
    """The entries of D^l(G_j) for every joint and of D^l(G_i^T G_k) for the pairs of WIGNER_PAIRS, l in degrees,
    [B, (J + len(WIGNER_PAIRS)) * sum (2l+1)^2], of the rotations relative to the root g [B, J, 3, 3] (for RSGM's
    poses: hp.models.rsgm.global_rotations(x)). The layout: the joints' blocks of each l, then the pairs' blocks."""
    i, k = (list(v) for v in zip(*PAIR_INDEX))
    rel = g[:, i].transpose(-1, -2) @ g[:, k]
    return torch.cat([wigner_features(g, degrees), wigner_features(rel, degrees)], dim=1)
