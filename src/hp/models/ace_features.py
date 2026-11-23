"""ACE-like descriptors of a pose: atomic cluster expansion on the joint positions, plus the twist of the bones.

    ace = AceFeatures()                     # buffers only, nothing trained
    b = ace(x)                              # x [..., 19, 3, 3] (AmassJoints4Learning) -> [..., ace.dim]
    mean, std = ace_statistics(ace, train)  # for the z-scoring (RSGM inputs="local+ace")

The points are the 22 body joints of SMPL (the fixed skeleton of hp.data.conditions, root at the origin, root and toe
rotations the identity) and three end points that make the rotations of the leaves visible: the tips of the hands
(8 cm from the wrists along the forearm) and of the head (10 cm from the head along the neck). 25 points in all.

Every point is its own species (the joints are distinguishable), and the graph is complete. With x(r) a scaling of the
distance into [-1, 1] and T_n the Chebyshev polynomials, the descriptors are, for every point pair (j, k) and triple
(k, j, k') with the centre j:

    2-body  B1[jk, n]         = T_n(x_jk(r_jk)),                                n = 1..radial
    3-body  B2[k j k', n n' l] = T_n(x_jk(r_jk)) T_n'(x_jk'(r_jk')) P_l(cos theta_kjk'),
                                                    n, n' = 0..triple_radial - 1, l = 1..angular
    twist   cos(q psi_j), sin(q psi_j),            q = 1..twist

The 3-body term is the rotation invariant sum_m A_jknlm A*_jk'n'lm of ACE with the one-particle basis
R_n(r) Y_lm(r_hat), by the addition theorem (up to (2l + 1) / 4 pi). x_jk(r) = 2 r / r_max_jk - 1, r_max_jk the length
of the path from j to k along the bones, so that every pair uses the whole range of the polynomials and stays inside
it for any rotations. No cutoff function: the set of points is fixed.

Pairs whose distance does not depend on the pose (a bone, two points on one rigid frame) are left out of B1, triples
whose angle does not depend on it out of B2; which ones is found on random rotations at construction. The triples are
those with k and k' within `hops` bones of the centre j.

The twist psi_j of a joint with one child c is the angle of x_j about the axis a = b_c / |b_c| of the bone to the
child (the swing-twist decomposition x_j = swing R_a(psi_j)). From the quaternion (w, v) of x_j and u = a.v,
psi = 2 atan2(u, w), so cos psi = (w^2 - u^2) / (w^2 + u^2), sin psi = 2 w u / (w^2 + u^2), with
w^2 = (1 + tr x) / 4, w u = a.vee(x - x^T) / 4, w^2 + u^2 = (1 + a.x a) / 2: polynomials in x, invariant to the sign
of the quaternion; undefined only where x turns a into -a (a swing of pi), where the denominator is clamped. Positions
cannot see this angle (it does not move the child), and the leaves have no other way to show it. Spine2, with three
children, has no twist axis; its rotation is fixed by the positions of the children.

All operations are out of place and without data-dependent branches, so torch.func can vmap and differentiate them
(the exact divergence of RSGM's log_prob).
"""
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from hp.data.amass_cfg import AmassJoints4Learning
from hp.data.conditions import _skeleton

LEARNED = [joint.value for joint in AmassJoints4Learning]  # the 19 joints with a rotation, in the order of x
# end points: (their parent joint, length in meters along the bone to that joint)
END_POINTS = ((20, 0.08), (21, 0.08), (15, 0.10))


def chebyshev(x: torch.Tensor, n: int) -> List[torch.Tensor]:
    """[T_0(x), ..., T_n(x)] by the recurrence T_{k+1} = 2 x T_k - T_{k-1}."""
    out = [torch.ones_like(x), x]
    for _ in range(n - 1):
        out.append(2 * x * out[-1] - out[-2])
    return out[:n + 1]


def legendre(c: torch.Tensor, n: int) -> List[torch.Tensor]:
    """[P_1(c), ..., P_n(c)] by Bonnet's recurrence (k + 1) P_{k+1} = (2k + 1) c P_k - k P_{k-1}."""
    out = [torch.ones_like(c), c]
    for k in range(1, n):
        out.append(((2 * k + 1) * c * out[-1] - k * out[-2]) / (k + 1))
    return out[1:n + 1]


def _points() -> Tuple[Tuple[int, ...], np.ndarray]:
    """(parents of the 25 points, their offsets [25, 3] from the parent in the parent's frame)."""
    parents, offsets = _skeleton()
    parents, offsets = list(parents), [o for o in offsets]
    for joint, length in END_POINTS:
        bone = offsets[joint]
        parents.append(joint)
        offsets.append(length * bone / np.linalg.norm(bone))
    return tuple(parents), np.stack(offsets)


class AceFeatures(nn.Module):
    """Descriptors [..., dim] of poses [..., 19, 3, 3] (module doc); every option fixes the set of descriptors."""

    def __init__(self, radial: int = 4, angular: int = 2, triple_radial: int = 2, twist: int = 1, hops: int = 2,
                 seed: int = 0, probes: int = 256):
        super().__init__()
        if radial < 1 or angular < 0 or triple_radial < 1 or twist < 0 or hops < 1:
            raise ValueError("radial, triple_radial and hops must be >= 1, angular and twist >= 0.")
        self.radial, self.angular, self.triple_radial, self.twist, self.hops = radial, angular, triple_radial, twist, hops
        parents, offsets = _points()
        self.parents = parents
        n = len(parents)
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.float32), persistent=False)  # float32: mps
        # the local rotation of every point: its index in x, or -1 (the identity: root, toes, end points)
        self.slot = [LEARNED.index(i) if i in LEARNED else -1 for i in range(n)]

        # tree distances (in bones) and path lengths (in meters) between all points
        adjacency = {i: set() for i in range(n)}
        for i, p in enumerate(parents):
            if p >= 0:
                adjacency[i].add(p)
                adjacency[p].add(i)
        hop = np.full((n, n), -1, dtype=int)
        length = np.zeros((n, n))
        bone = np.linalg.norm(offsets, axis=1)
        for s in range(n):
            hop[s, s], frontier = 0, [s]
            while frontier:
                nxt = []
                for a in frontier:
                    for b in adjacency[a]:
                        if hop[s, b] < 0:
                            hop[s, b] = hop[s, a] + 1
                            length[s, b] = length[s, a] + bone[b if parents[b] == a else a]
                            nxt.append(b)
                frontier = nxt

        g = torch.Generator().manual_seed(seed)
        probe = self.positions(_random_rotations(probes, len(LEARNED), g))  # [probes, n, 3], float64
        tol = 1e-9

        pairs = [(j, k) for j in range(n) for k in range(j + 1, n)]
        dist = torch.stack([(probe[:, k] - probe[:, j]).norm(dim=-1) for j, k in pairs], 1)
        keep = (dist.std(0) > tol).tolist()
        self.pairs = [pq for pq, ok in zip(pairs, keep) if ok]
        self.register_buffer("pair_j", torch.tensor([j for j, _ in self.pairs]), persistent=False)
        self.register_buffer("pair_k", torch.tensor([k for _, k in self.pairs]), persistent=False)
        self.register_buffer("pair_rmax", torch.tensor([length[j, k] for j, k in self.pairs], dtype=torch.float32),
                             persistent=False)

        triples = []  # (k, j, k'): centre j, k < k' within `hops` bones of j
        for j in range(n):
            near = [k for k in range(n) if 0 < hop[j, k] <= hops]
            triples += [(k, j, q) for a, k in enumerate(near) for q in near[a + 1:]]
        if angular and triples:
            u = probe[:, [k for k, _, _ in triples]] - probe[:, [j for _, j, _ in triples]]
            v = probe[:, [q for _, _, q in triples]] - probe[:, [j for _, j, _ in triples]]
            cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
            keep = (cos.std(0) > tol).tolist()
            triples = [tr for tr, ok in zip(triples, keep) if ok]
        else:
            triples = []
        self.triples = triples
        self.register_buffer("tri_k", torch.tensor([k for k, _, _ in triples], dtype=torch.long), persistent=False)
        self.register_buffer("tri_j", torch.tensor([j for _, j, _ in triples], dtype=torch.long), persistent=False)
        self.register_buffer("tri_q", torch.tensor([q for _, _, q in triples], dtype=torch.long), persistent=False)
        self.register_buffer("tri_rmax_k", torch.tensor([length[j, k] for k, j, _ in triples], dtype=torch.float32),
                             persistent=False)
        self.register_buffer("tri_rmax_q", torch.tensor([length[j, q] for _, j, q in triples], dtype=torch.float32),
                             persistent=False)

        # twist: joints with a rotation and exactly one child among the points
        children = {i: [c for c, p in enumerate(parents) if p == i] for i in range(n)}
        self.twist_joints = [i for i in LEARNED if len(children[i]) == 1] if twist else []
        axes = [offsets[children[i][0]] / np.linalg.norm(offsets[children[i][0]]) for i in self.twist_joints]
        self.register_buffer("twist_slot", torch.tensor([LEARNED.index(i) for i in self.twist_joints], dtype=torch.long),
                             persistent=False)
        self.register_buffer("twist_axis", torch.as_tensor(np.array(axes).reshape(-1, 3), dtype=torch.float32),
                             persistent=False)

        self.dim = (len(self.pairs) * radial + len(triples) * triple_radial ** 2 * angular
                    + len(self.twist_joints) * 2 * twist)

    def positions(self, x: torch.Tensor) -> torch.Tensor:
        """Positions [..., 25, 3] in meters of the points of poses x [..., 19, 3, 3]."""
        offsets = self.offsets.to(dtype=x.dtype, device=x.device)
        rotations, positions = [], []
        for i, p in enumerate(self.parents):
            local = None if self.slot[i] < 0 else x[..., self.slot[i], :, :]
            if p < 0:
                rotations.append(local)
                positions.append(torch.zeros_like(x[..., 0, :, 0]))
                continue
            parent = rotations[p]
            bone = offsets[i].expand_as(positions[p]) if parent is None else parent @ offsets[i]
            positions.append(positions[p] + bone)
            rotations.append(local if parent is None else (parent if local is None else parent @ local))
        return torch.stack(positions, dim=-2)

    def twist_cos_sin(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(cos psi, sin psi) [..., T] of the twist joints (module doc)."""
        r = x[..., self.twist_slot, :, :]
        a = self.twist_axis.to(dtype=x.dtype, device=x.device)
        w2 = (1 + r.diagonal(dim1=-2, dim2=-1).sum(-1)) / 4
        vee = torch.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1)
        wu = (vee * a).sum(-1) / 4
        ara = (a * (r @ a[..., None])[..., 0]).sum(-1)
        den = ((1 + ara) / 2).clamp_min(1e-6)
        return (2 * w2 - den) / den, 2 * wu / den  # w^2 - u^2 = 2 w^2 - (w^2 + u^2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [self.point_descriptors(self.positions(x))]
        if self.twist_joints:
            c, sn = self.twist_cos_sin(x)
            ck, sk = c, sn
            for q in range(self.twist):
                if q:  # cos and sin of (q + 1) psi from those of q psi
                    ck, sk = ck * c - sk * sn, sk * c + ck * sn
                out += [ck, sk]
        return torch.cat(out, dim=-1)

    def point_descriptors(self, p: torch.Tensor) -> torch.Tensor:
        """The 2- and 3-body descriptors [..., D] of the points p [..., 25, 3]: invariant to rotations of p."""
        x = p
        out = []
        d = p[..., self.pair_k, :] - p[..., self.pair_j, :]
        s = 2 * d.norm(dim=-1) / self.pair_rmax.to(x.dtype) - 1
        out += chebyshev(s, self.radial)[1:]
        if len(self.triples):
            u = p[..., self.tri_k, :] - p[..., self.tri_j, :]
            v = p[..., self.tri_q, :] - p[..., self.tri_j, :]
            ru, rv = u.norm(dim=-1), v.norm(dim=-1)
            cos = (u * v).sum(-1) / (ru * rv)
            tu = chebyshev(2 * ru / self.tri_rmax_k.to(x.dtype) - 1, self.triple_radial - 1)
            tv = chebyshev(2 * rv / self.tri_rmax_q.to(x.dtype) - 1, self.triple_radial - 1)
            for pl in legendre(cos, self.angular):
                out += [a * b * pl for a in tu[:self.triple_radial] for b in tv[:self.triple_radial]]
        return torch.cat(out, dim=-1)


def _random_rotations(n: int, joints: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform random rotations [n, joints, 3, 3] in float64 (normalized Gaussian quaternions)."""
    q = torch.randn(n, joints, 4, dtype=torch.float64, generator=generator)
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


@torch.no_grad()
def ace_statistics(ace: AceFeatures, poses: torch.Tensor, chunk: int = 200_000,
                   std_floor: float = 0.1) -> Tuple[list, list]:
    """(mean, std) [dim] of the descriptors of training poses [N, 19, 3, 3], as lists for the model's arguments; in
    float64. The std is at least std_floor: every descriptor lies in [-1, 1], so the z-scored input stays within
    2 / std_floor (20) even at the large noise of RSGM, where the poses are far from the data and a descriptor of tiny
    spread on the data (the twist of the spine) would otherwise reach z of several hundred."""
    total = torch.zeros(ace.dim, dtype=torch.float64)
    square = torch.zeros(ace.dim, dtype=torch.float64)
    for i in range(0, len(poses), chunk):
        b = ace(poses[i:i + chunk].double())
        total += b.sum(0)
        square += (b * b).sum(0)
    n = len(poses)
    mean = total / n
    std = ((square - n * mean ** 2) / max(n - 1, 1)).clamp_min(0).sqrt()
    std = std.clamp_min(std_floor)
    return mean.tolist(), std.tolist()
