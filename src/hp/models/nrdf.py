"""NRDF (He, Tiwari, Dai, Pons-Moll, CVPR 2024, github.com/hynann/NRDF): their released checkpoint as a PoseModel, for
sampling only. The code of the network and of the projection is rewritten from theirs (non-commercial research license,
the weights are not in this repository).

The model is a neural field: a network that predicts the distance of a pose to the manifold of plausible poses. It is
not a density, so has_log_prob is False. The poses are unit quaternions [w, x, y, z] of the 21 body joints of SMPL
(SMPL+H AMASS, the official split). Sampling as their lib/exp/sample_poses.py:
  - start: dist ~ N(0, 1) per pose, shared out over the joints with random weights: per joint the rotation by the
    angle 21 * weight * dist about a random axis (nrdf_sampling);
  - 200 steps of Riemannian gradient descent on the field, q <- q exp(-step_size * dist * unit gradient) (rdfgrad),
    step_size 0.01 (the alpha of Tab. 2 of their paper).
"""
from pathlib import Path
from typing import Optional

import roma
import torch
from torch import nn
from torch.nn import functional as F

from hp.models.base import PoseModel
from hp.models.conditioning import move
from hp.models.posedflow import KEEP, NUM_JOINTS

PARENTS = (-1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 11, 12, 13, 15, 16, 17, 18)  # among the 21 joints
BONE_FEATURES = 6
HIDDEN = (256, 512, 1024, 512, 256, 64)
BETA = 100.0  # of the softplus activations


def hamilton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The product a b of quaternions [..., 4] with the real part first."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def rdf_step(q: torch.Tensor, egrad: torch.Tensor, dist: torch.Tensor, step_size: float) -> torch.Tensor:
    """One step of their rdfgrad from q [B, J, 4], with the Euclidean gradient egrad [B, J, 4] of the field and its
    value dist [B]: the unit tangent per joint, moved to the identity, scaled by -step_size * dist; the real part of
    the result is made positive."""
    v = F.normalize(egrad - (q * egrad).sum(-1, keepdim=True) * q, dim=-1)
    imaginary = v[..., 1:] - q[..., 1:] / (1 + q[..., :1]) * v[..., :1]  # the first column of their rmat
    tangent = -step_size * dist[:, None, None] * imaginary
    alpha = tangent.norm(dim=-1, keepdim=True)
    sinc = torch.where(alpha > 0, torch.sin(alpha) / alpha, torch.ones_like(alpha))
    out = F.normalize(hamilton(q, torch.cat([torch.cos(alpha), tangent * sinc], dim=-1)), dim=-1)
    return torch.where(out[..., :1] < 0, -out, out)


def random_rotations(dist: torch.Tensor, joints: int = NUM_JOINTS) -> torch.Tensor:
    """Quaternions [B, joints, 4] for dist [B]: dist shared out over the joints with random weights, per joint the
    rotation by the angle joints * weight * dist about a random axis (their nrdf_sampling)."""
    n = dist.shape[0]
    axes = F.normalize(torch.randn(n, joints, 3, dtype=dist.dtype), dim=-1)
    weights = torch.rand(n, joints, dtype=dist.dtype)
    angles = dist[:, None] * weights / weights.sum(-1, keepdim=True) * joints
    return torch.cat([torch.cos(angles / 2)[..., None], torch.sin(angles / 2)[..., None] * axes], dim=-1)


def perturb(clean: torch.Tensor, sigma: float) -> torch.Tensor:
    """Noisy poses [B, J, 4] around the poses clean [B, J, 4], with dist ~ N(0, sigma^2) (their PerturbData)."""
    dist = sigma * torch.randn(clean.shape[0], dtype=clean.dtype)
    return hamilton(clean, random_rotations(dist, clean.shape[1]))


def geodesic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Their distance of poses [..., J, 4]: over the joints the mean of acos|<a, b>|, half the rotation angle."""
    return torch.acos((a * b).sum(-1).abs().clamp(max=1)).mean(-1)


@torch.no_grad()
def nearest_distances(noisy: torch.Tensor, origin: torch.Tensor, clean: torch.Tensor, k: int = 1000,
                      block: int = 2_000_000) -> torch.Tensor:
    """The distance [Q] of noisy poses [Q, J, 4] to the clean poses [M, J, 4], as their nn_search: the k candidates of
    the smallest Euclidean distance of the quaternions (the largest dot product, as all have the same norm; their
    FAISS Flat index) and the pose each one came from, origin [Q, J, 4]; the minimum of geodesic over these. The clean
    poses are searched in blocks, to limit the memory of the matrix of dot products."""
    queries = noisy.flatten(1)
    scores, indices = [], []
    for start in range(0, clean.shape[0], block):
        dots = queries @ clean[start:start + block].flatten(1).T
        s, i = dots.topk(min(k, dots.shape[1]), dim=1)
        scores.append(s)
        indices.append(i + start)
    chosen = torch.cat(indices, dim=1).gather(1, torch.cat(scores, dim=1).topk(min(k, clean.shape[0]), dim=1).indices)
    candidates = torch.cat([clean[chosen], origin[:, None]], dim=1)
    return geodesic(noisy[:, None], candidates).amin(1)


class BoneMLP(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 4 + BONE_FEATURES), nn.Softplus(beta=BETA),
                                 nn.Linear(4 + BONE_FEATURES, BONE_FEATURES), nn.Softplus(beta=BETA))


class StructureEncoder(nn.Module):
    """Features of a pose along the kinematic tree: each joint from its quaternion and the features of its parent."""

    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([BoneMLP(4 if p < 0 else 4 + BONE_FEATURES) for p in PARENTS])

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        features = []
        for j, (mlp, p) in enumerate(zip(self.net, PARENTS)):
            features.append(mlp.net(q[:, j] if p < 0 else torch.cat([q[:, j], features[p]], dim=-1)))
        return torch.cat(features, dim=-1)


class DistanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        dims = (NUM_JOINTS * BONE_FEATURES, *HIDDEN, 1)
        self.count = len(dims) - 1
        for k in range(self.count):
            setattr(self, f"lin{k}", nn.Linear(dims[k], dims[k + 1]))  # the names of their checkpoint
        self.activation = nn.Softplus(beta=BETA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for k in range(self.count):
            x = getattr(self, f"lin{k}")(x)
            if k < self.count - 1:
                x = self.activation(x)
        return self.activation(x)


class NRDFField(nn.Module):
    """The distance field on poses of 21 joints; state_dict keys as in their checkpoints (without the unused decoder)."""

    def __init__(self):
        super().__init__()
        self.enc = StructureEncoder()
        self.dfnet = DistanceNet()

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """The predicted distance [B] of poses q [B, 21, 4] to the manifold."""
        return self.dfnet(self.enc(q))[:, 0]

    def loss(self, noisy: torch.Tensor, dist: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Their loss (the weights of the released model): L1 to the distance, and the field on clean poses to zero."""
        return 0.5 * (self(noisy) - dist).abs().mean() + 0.1 * self(clean).abs().mean()


class NRDFPretrained(PoseModel):
    """Their checkpoint as a PoseModel of our 19 joints (the toes are leaves of the tree; sample_21 has all 21)."""

    name = "nrdf_pretrained"  # not in the registry: it is loaded from their checkpoint, not trained here
    has_log_prob = False

    def __init__(self, checkpoint: str, step_size: float = 0.01, iterations: int = 200):
        super().__init__(len(KEEP), "matrix")
        self.step_size, self.iterations = step_size, iterations
        self.field = NRDFField()
        state = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)["model_state_dict"]
        self.field.load_state_dict({k: v for k, v in state.items() if not k.startswith("dec.")})  # the decoder is not used
        self.requires_grad_(False)
        self.build_args = dict(checkpoint=str(checkpoint), step_size=step_size, iterations=iterations)

    @property
    def dtype(self) -> torch.dtype:
        return self.field.dfnet.lin0.weight.dtype

    def loss(self, pose, cond=None):
        raise NotImplementedError("The pretrained NRDF model is not trained here.")

    def distance(self, q: torch.Tensor) -> torch.Tensor:
        """The predicted distance [B] of poses q [B, 21, 4] to the manifold."""
        return self.field(q)

    def start(self, n: int) -> torch.Tensor:
        """n starting poses [n, 21, 4], their nrdf_sampling with dist ~ N(0, 1)."""
        return move(random_rotations(torch.randn(n, dtype=torch.float64)), self.device, self.dtype)

    def project(self, q: torch.Tensor) -> torch.Tensor:
        """The poses q [B, 21, 4] after the steps of gradient descent on the field."""
        with torch.enable_grad():  # the evaluation samples under no_grad
            for _ in range(self.iterations):
                q = q.detach().requires_grad_(True)
                dist = self.distance(q)
                egrad = torch.autograd.grad(dist.sum(), q)[0]
                q = rdf_step(q.detach(), egrad, dist.detach(), self.step_size)
        return q.detach()

    def sample_21(self, n: int) -> torch.Tensor:
        """n poses of all 21 joints [n, 21, 3, 3]."""
        return roma.unitquat_to_rotmat(self.project(self.start(n))[..., [1, 2, 3, 0]])

    def sample(self, n: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_cond(cond, n)
        return self.sample_21(n)[:, KEEP]
