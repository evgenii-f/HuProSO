"""VPoser (Pavlakos et al., CVPR 2019; the V02_05 of github.com/nghorbani/human_body_prior), trained here on our data.

The network and the loss are those of human_body_prior/models/vposer_model.py and train/vposer_trainer.py, V02_05.yaml:
- encoder: the axis-angle of the joints [B, 3J], BatchNorm, Linear 512, LeakyReLU, BatchNorm, Dropout 0.1, Linear 512,
  Linear 512 (no activation between the two, as theirs), then the mean and softplus(.) scale of a normal in R^32;
- decoder: Linear 512, LeakyReLU, Dropout 0.1, Linear 512, LeakyReLU, Linear 6J, then their ContinousRotReprDecoder
  (Gram-Schmidt of the two columns of each 3x2 block) to rotation matrices;
- loss: 0.005 KL(q(z|x) || N(0, I)) (summed over z, mean over the batch) + 4 L1 between the vertices of the SMPL-X
  bodies of the pose and of its reconstruction; during the first epochs (their keep_extra_loss_terms_until_epoch 15)
  also 2 geodesic distance of the rotations (mean over the joints) + 2 L1 between the joints of the two bodies.
  Here the first epochs are the first `extra_terms_steps` training steps, counted in the buffer `steps`;
- their validation loss: the mean Euclidean distance of the vertices (v2v), which is what loss() returns in eval mode
  (with fixed noise, so that it is the same number every time);
- sampling: z ~ N(0, I), decoded with dropout off (their sample_poses).

Their body model is the SMPL-X neutral with the shape zero (BodyModel(bm_fname) with betas zero): the bodies of the
loss are posed by the body joints only, every other joint (root, jaw, eyes, hands) the identity. `body_model` is the
SMPL-X .npz (the public one, theirs is the locked-head SMPL-X of AMASS); it is read the first time loss() needs it,
so a trained model samples without it. `joints` are the SMPL-X joints of our joints (default: our 19 joints for 19,
the body joints 1 ... 21 for 21); the toes of our 19-joint data are the identity in both bodies.

A VAE has no exact log-likelihood (only a bound), so has_log_prob is False.
"""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import roma
import torch
from torch import nn
from torch.nn import functional as F

from hp.models.base import PoseModel
from hp.models.conditioning import move
from hp.models.registry import register_model

BODY_JOINTS = 21  # SMPL-X joints 1 ... 21, the body (their pose_body)
OUR_19 = [j for j in range(1, BODY_JOINTS + 1) if j not in (10, 11)]  # AmassJoints4Learning: without the toes
DEFAULT_BODY_MODEL = "~/data/body_models/smplx/SMPLX_NEUTRAL.npz"


class NormalDistDecoder(nn.Module):
    """Their NormalDistDecoder: N(mu(x), softplus(logvar(x))), the second head being a scale despite its name."""

    def __init__(self, num_feat_in: int, latent_dim: int):
        super().__init__()
        self.mu = nn.Linear(num_feat_in, latent_dim)
        self.logvar = nn.Linear(num_feat_in, latent_dim)

    def forward(self, x):
        return self.mu(x), F.softplus(self.logvar(x))


def continuous_rotation(x: torch.Tensor) -> torch.Tensor:
    """Their ContinousRotReprDecoder: [..., 6] read as a 3x2 matrix (row-major), Gram-Schmidt of its columns -> [N, 3, 3]
    with the columns b1, b2, b1 x b2."""
    m = x.reshape(-1, 3, 2)
    b1 = F.normalize(m[:, :, 0], dim=1)
    b2 = F.normalize(m[:, :, 1] - (b1 * m[:, :, 1]).sum(1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=1)], dim=-1)


def geodesic(r1: torch.Tensor, r2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Their geodesic_loss_R (reduction mean): acos((tr(r1 r2^T) - 1) / 2), the cosine clamped. Here the clamp keeps
    eps from +-1, where the gradient of acos is infinite."""
    m = r1 @ r2.transpose(-1, -2)
    cos = (m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(cos.clamp(-1 + eps, 1 - eps)).mean()


class BodyMesh(nn.Module):
    """The vertices and joints of an SMPL-X body with the shape zero, posed by some of its joints (linear blend
    skinning with the pose blend shapes, as the lbs of smplx and human_body_prior). The arrays are not saved with the
    model (non-persistent buffers)."""

    def __init__(self, path, joints: Sequence[int]):
        super().__init__()
        data = np.load(Path(path).expanduser(), allow_pickle=True)
        v = torch.as_tensor(data["v_template"], dtype=torch.float32)  # [V, 3]
        regressor = torch.as_tensor(np.asarray(data["J_regressor"], dtype=np.float32))  # [K, V]
        parents = np.asarray(data["kintree_table"])[0].astype(np.int64)
        parents[0] = -1
        k = len(parents)
        self.body = min(BODY_JOINTS, k - 1)  # the joints with pose blend shapes here: 1 ... body
        if max(joints) > self.body or min(joints) < 1:
            raise ValueError(f"joints must be body joints 1 ... {self.body} of the body model, got {list(joints)}.")
        posedirs = np.asarray(data["posedirs"], dtype=np.float32)[:, :, : 9 * self.body]  # [V, 3, 9 body]
        self.joints, self.parents = list(joints), parents.tolist()
        self.register_buffer("v_template", v, persistent=False)
        self.register_buffer("j_rest", regressor @ v, persistent=False)  # [K, 3]
        self.register_buffer("posedirs", torch.as_tensor(posedirs).reshape(-1, 9 * self.body).T.contiguous(), persistent=False)
        self.register_buffer("weights", torch.as_tensor(np.asarray(data["weights"], dtype=np.float32)), persistent=False)

    def forward(self, rotations: torch.Tensor):
        """rotations [B, len(joints), 3, 3] of `joints` -> (vertices [B, V, 3], joints [B, K, 3])."""
        b, dtype = rotations.shape[0], self.v_template.dtype
        k = len(self.parents)
        local = torch.eye(3, dtype=dtype, device=rotations.device).repeat(b, k, 1, 1)
        local[:, self.joints] = rotations.to(dtype)
        feature = (local[:, 1:self.body + 1] - torch.eye(3, dtype=dtype, device=rotations.device)).reshape(b, -1)
        posed = self.v_template + (feature @ self.posedirs).reshape(b, -1, 3)
        rotation, position = [local[:, 0]], [self.j_rest[0].expand(b, 3)]
        for i in range(1, k):  # forward kinematics
            p = self.parents[i]
            rotation.append(rotation[p] @ local[:, i])
            position.append(position[p] + rotation[p] @ (self.j_rest[i] - self.j_rest[p]))
        rotation, position = torch.stack(rotation, 1), torch.stack(position, 1)  # [B, K, 3, 3], [B, K, 3]
        translation = position - (rotation @ self.j_rest[..., None]).squeeze(-1)  # moves the rest joint to its place
        transform = torch.cat([rotation, translation[..., None]], -1).reshape(b, k, 12)
        blended = (self.weights @ transform).reshape(b, -1, 3, 4)  # [B, V, 3, 4]
        vertices = (blended[..., :3] @ posed[..., None]).squeeze(-1) + blended[..., 3]
        return vertices, position


@register_model("vposer")
class VPoser(PoseModel):
    has_log_prob = False

    def __init__(
        self,
        num_joints: int,
        num_neurons: int = 512,
        latent_dim: int = 32,
        loss_kl_wt: float = 0.005,
        loss_rec_wt: float = 4.0,
        loss_matrot_wt: float = 2.0,
        loss_jtr_wt: float = 2.0,
        extra_terms_steps: int = 0,  # training steps with the geodesic and joint terms (their first 15 epochs)
        body_model: str = DEFAULT_BODY_MODEL,
        joints: Optional[Sequence[int]] = None,  # the SMPL-X joints of ours; None: our 19 (19) or 1 ... 21 (21)
        representation: str = "axis_angle",
        cond_dim: int = 0,
    ):
        if representation != "axis_angle":
            raise ValueError("VPoser encodes axis-angle vectors; use representation='axis_angle'.")
        if joints is None:
            defaults = {len(OUR_19): OUR_19, BODY_JOINTS: list(range(1, BODY_JOINTS + 1))}
            if num_joints not in defaults:
                raise ValueError(f"Give the SMPL-X joints of the {num_joints} joints (joints=[...]).")
            joints = defaults[num_joints]
        if len(joints) != num_joints:
            raise ValueError(f"joints has {len(joints)} entries for {num_joints} joints.")
        super().__init__(num_joints, representation, cond_dim)
        self.latent_dim = latent_dim
        self.weights = dict(kl=loss_kl_wt, rec=loss_rec_wt, matrot=loss_matrot_wt, jtr=loss_jtr_wt)
        self.extra_terms_steps = extra_terms_steps
        self.body_model, self.smplx_joints = body_model, [int(j) for j in joints]
        n = 3 * num_joints
        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(n),
            nn.Linear(n, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, latent_dim),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, 6 * num_joints),
        )
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))  # training steps so far (loss in train mode)
        self._mesh = None  # BodyMesh, built when the loss first needs it (not a submodule: not saved, not in EMA copies)

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder_net[0].weight.dtype

    def mesh(self) -> BodyMesh:
        device = self.device
        if self._mesh is None or self._mesh.v_template.device != device:
            mesh = self._mesh or BodyMesh(self.body_model, self.smplx_joints)
            self.__dict__["_mesh"] = mesh.to(device)  # a plain attribute, not a registered submodule
        return self._mesh

    def encode_distribution(self, pose: torch.Tensor):
        """(mean, scale) [B, latent_dim] of q(z | pose), pose [B, J, 3, 3]."""
        return self.encoder_net(roma.rotmat_to_rotvec(pose).reshape(len(pose), -1))

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim] -> rotations [B, J, 3, 3]."""
        return continuous_rotation(self.decoder_net(z)).reshape(len(z), self.num_joints, 3, 3)

    def loss(self, pose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Their training loss (train mode, counting the step) or their validation loss v2v (eval mode)."""
        self.check_cond(cond, len(pose))
        device, dtype = self.device, self.dtype
        x = move(pose.detach().reshape(-1, self.num_joints, 3, 3), device, dtype)
        mean, scale = self.encode_distribution(x)
        generator = None if self.training else torch.Generator(device=device).manual_seed(0)
        z = mean + scale * torch.randn(mean.shape, generator=generator, device=device, dtype=dtype)
        rec = self.decode_latent(z)
        mesh = self.mesh()
        with torch.no_grad():
            v_orig, j_orig = mesh(x)
        v_rec, j_rec = mesh(rec)
        if not self.training:
            return (v_rec - v_orig).norm(dim=-1).mean()
        w = self.weights
        kl = (0.5 * (mean ** 2 + scale ** 2) - torch.log(scale) - 0.5).sum(-1).mean()  # KL(N(mean, scale) || N(0, I))
        loss = w["kl"] * kl + w["rec"] * (v_rec - v_orig).abs().mean()
        if self.steps < self.extra_terms_steps:
            loss = loss + w["matrot"] * geodesic(rec, x) + w["jtr"] * (j_rec - j_orig).abs().mean()
        self.steps += 1
        return loss

    @torch.no_grad()
    def sample(self, n: int, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.check_cond(cond, n)
        z = torch.randn(n, self.latent_dim, device=self.device, dtype=self.dtype)
        training = self.training
        self.decoder_net.eval()  # their sample_poses: dropout off
        try:
            return self.decode_latent(z)
        finally:
            self.decoder_net.train(training)
