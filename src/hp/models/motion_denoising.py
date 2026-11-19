"""Motion denoising with a pose prior: the body poses of a sequence of frames are recovered from 3D joints with
Gaussian noise (the task of PoseD-Flow, Nadar, Foti, Birdal, CVPR 2026, their motion_denoising.py and Tab. 3, which
takes it from DPoser, Lu et al., DPoser-X, run/tester/body/motion_denoising.py there). The root orientation is the
identity and betas are 0, on SMPL-X neutral; the joints are the first 22 of SMPL-X.

Three procedures:

- dflow_denoise: Riemannian D-Flow (hp.models.dflow) as PoseD-Flow, one latent per frame:
      loss = mean (J(x_1) - noisy)^2 + temp_weight * mean_frames,joints d(x_1[f], x_1[f + 1]) + running_weight * running
  the start projx(sqrt(0.75) F(rest) + sqrt(0.25) noise), F(rest) the sampling ODE run from the rest pose, as theirs
  (their T-pose pushed through the flow); defaults of their README for sigma 40 mm: lr 0.25, 300 steps, temp_weight
  1e-2, running_weight 1e-5, and their optimizer (RiemannianAdam on SO3Power: one second moment per joint, see
  hp.models.dflow.riemannian_adam_dflow), whose steps are about sqrt(6) times smaller than those of Adam per
  coordinate at the same lr.
- dposer_denoise: DPoser's own optimization (MotionDenoise.optimize, time strategy "3"): the axis-angle pose itself,
  Adam lr 0.03, `iterations` x `steps_per_iter` steps with t linear from t_max down to t_min, every step
      loss = 100 / (1 + it^2) * mean_frames,joints |J(x) - noisy|              (the norm, not squared)
           + 10 (1 + it) * mean_pairs,vertices |V(x[f]) - V(x[f + 1])|
           + 0.1 (1 + it) * sds_weight * sum 0.5 (x_n - x0_hat(x_n))^2 / frames   (x_n: x z-scored, x0_hat the Tweedie
                                                                                   step from its noised copy, no grad)
  started at the SMPL mean pose + 0.01 N(0, I). prior=None: the same without the last term ("w/o prior").
- Sequences are batched: S sequences of T frames are one batch of S T poses, the loss the sum over the sequences of
  their losses (each a mean over its own frames), and the temporal terms do not cross sequences. Adam works per
  coordinate, so a sequence gets the same updates as alone.

The joints J come from Skeleton (forward kinematics of the rest joints of SMPL-X with betas 0: the pose blend shapes
do not move the joints), the vertices V for DPoser's temporal term from the body model.
"""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import roma
import torch

from hp.models import igso3
from hp.models.dflow import DPoserPrior, dflow, geodesic_loss
from hp.models.rsgm import RSGM, exp_frame
from hp.models.score_completion import _angle2, rsgm_denoise

BODY_JOINTS = 22  # pelvis + the 21 joints of pose_body


class Skeleton(torch.nn.Module):
    """The 22 body joints [B, 22, 3] of SMPL-X with betas 0, the root at the identity, from the 21 body rotations
    [B, 21, 3, 3] (SMPL joints 1 ... 21): forward kinematics of the rest joints J_regressor @ v_template."""

    def __init__(self, rest: torch.Tensor, parents: Sequence[int]):
        super().__init__()
        self.register_buffer("rest", torch.as_tensor(rest, dtype=torch.float32)[:BODY_JOINTS])
        self.parents = [int(p) for p in parents[:BODY_JOINTS]]

    @classmethod
    def from_file(cls, model_file) -> "Skeleton":
        data = np.load(Path(model_file).expanduser(), allow_pickle=True)
        rest = torch.from_numpy(data["J_regressor"].astype(np.float64) @ data["v_template"].astype(np.float64))
        parents = data["kintree_table"][0].astype(np.int64)
        parents[0] = -1
        return cls(rest.float(), parents.tolist())

    def forward(self, rotations: torch.Tensor) -> torch.Tensor:
        rest = self.rest.to(rotations.dtype)
        n = rotations.shape[0]
        eye = torch.eye(3, dtype=rotations.dtype, device=rotations.device).expand(n, 3, 3)
        frames, positions = [eye], [rest[0].expand(n, 3)]
        for j in range(1, BODY_JOINTS):
            p = self.parents[j]
            positions.append(positions[p] + (frames[p] @ (rest[j] - rest[p])[:, None])[..., 0])
            frames.append(frames[p] @ rotations[:, j - 1])
        return torch.stack(positions, 1)


def to_body(x: torch.Tensor, joints: Optional[Sequence[int]]) -> torch.Tensor:
    """Poses [B, J, 3, 3] of a model's joints (indices among the 21 body joints; None: all 21) as [B, 21, 3, 3], the
    other joints (the toes for our 19-joint models) at rest."""
    if joints is None or len(joints) == 21:
        return x
    out = torch.eye(3, dtype=x.dtype, device=x.device).repeat(x.shape[0], 21, 1, 1)
    return out.index_copy(1, torch.as_tensor(list(joints), device=x.device), x)


def dflow_denoise(prior, skeleton: Skeleton, noisy: torch.Tensor, joints: Optional[Sequence[int]] = None,
                  steps: int = 300, lr: float = 0.25, temp_weight: float = 1e-2, running_weight: float = 1e-5,
                  alpha: float = 0.75, optimizer: Optional[str] = None) -> torch.Tensor:
    """Poses [S, T, 21, 3, 3] of S sequences of T frames of noisy joints [S, T, 22, 3] by D-Flow on `prior`
    (hp.models.dflow.prior_of); `joints`: the body joints of the prior (None: all 21). optimizer (hp.models.dflow.dflow):
    None is theirs, "riemannian", for the priors on SO(3)^J, and "adam" for DPoser's Euclidean latent."""
    if optimizer is None:
        optimizer = "adam" if isinstance(prior, DPoserPrior) else "riemannian"
    s, t = noisy.shape[:2]
    device, dtype = prior.model.device, prior.model.dtype
    noisy = noisy.to(device, dtype)
    skeleton = skeleton.to(device)
    rest = torch.eye(3, device=device, dtype=dtype).repeat(s * t, prior.joints, 1, 1)
    start = prior.start(rest, alpha)

    def loss_fn(x1):
        body = to_body(x1, joints)
        data = (skeleton(body).view(s, t, BODY_JOINTS, 3) - noisy).pow(2).mean((1, 2, 3)).sum()
        body = body.view(s, t, 21, 3, 3)
        temporal = torch.stack([geodesic_loss(b[:-1], b[1:]) for b in body]).sum()
        return data + temp_weight * temporal

    # the running cost of the prior is a mean over all S T poses; theirs is the mean over the T poses of one sequence
    x1 = dflow(prior, loss_fn, start, steps, lr, running_weight * s, optimizer)
    return to_body(x1, joints).view(s, t, 21, 3, 3)


def mean_pose(file) -> torch.Tensor:
    """The SMPL mean body pose [63] (axis-angle of joints 1 ... 21) of smpl_mean_params.npz (rot6d of 24 joints), the
    start of DPoser's denoising."""
    six = torch.as_tensor(np.load(Path(file).expanduser())["pose"], dtype=torch.float32).reshape(-1, 3, 2)
    # their rot6d_to_axis_angle: Gram-Schmidt of the two columns
    a, b = six[..., 0], six[..., 1]
    x = torch.nn.functional.normalize(a, dim=-1)
    y = torch.nn.functional.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    rot = torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)
    return roma.rotmat_to_rotvec(rot).reshape(-1)[3:66]


def dposer_denoise(body_model, noisy: torch.Tensor, init: torch.Tensor, prior=None,
                   joints: Optional[Sequence[int]] = None, iterations: int = 3, steps_per_iter: int = 60,
                   t_max: float = 0.25, t_min: float = 0.05, lr: float = 0.03, sds_weight: float = 1.0) -> torch.Tensor:
    """Poses [S, T, 21, 3, 3] of noisy joints [S, T, 22, 3] by DPoser's optimization; init [S, T, 63] the start
    (axis-angle); prior: a DPoser (hp.models.dposer) of the body joints `joints` (None: all 21), an RSGM (the prior
    term on SO(3)^J as rsgm_complete: IGSO(3) noise of Brownian time s(t), RSGM's Tweedie step, squared angles; t in
    RSGM's times, so t_max, t_min and sds_weight need their own values), or None (no prior term); body_model: human_body_prior's BodyModel of SMPL-X neutral (vertices and joints, differentiable)."""
    s, t = noisy.shape[:2]
    device = noisy.device if prior is None else prior.device
    noisy = noisy.to(device, torch.float32)
    pose = init.reshape(s * t, 63).to(device, torch.float32).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pose], lr, betas=(0.9, 0.999))
    total = iterations * steps_per_iter
    keep = list(range(21)) if joints is None else list(joints)
    with torch.enable_grad():
        for it in range(iterations):
            for i in range(steps_per_iter):
                step = it * steps_per_iter + i
                losses = []
                if isinstance(prior, RSGM):  # the same on SO(3)^J (rsgm_complete): squared angles, RSGM's times
                    tt = t_min + (total - step - 1) / total * (t_max - t_min)
                    x = roma.rotvec_to_rotmat(pose.view(-1, 21, 3))[:, keep]
                    # IGSO(3) noise of the Brownian time of every joint (per joint with joint_scale), drawn on the
                    # device where it has float64 (cuda), else on the CPU (mps)
                    work = device if torch.device(device).type == "cuda" else torch.device("cpu")
                    t64 = torch.full((s * t,), tt, dtype=torch.float64, device=work)
                    angles = prior.angles.sample(prior.joint_s(t64))[..., None]
                    noise = (angles * igso3.random_axes((s * t, len(keep)), device=work)).to(device, x.dtype)
                    x0_hat = rsgm_denoise(prior, exp_frame(x.detach(), noise), t64.to(device, x.dtype))
                    sds = (0.5 * _angle2(x, x0_hat)).sum() / t
                    losses.append(0.1 * (1 + it) * sds_weight * sds)
                elif prior is not None:
                    tt = t_min + (total - step - 1) / total * (t_max - t_min)
                    vec_t = torch.full((s * t,), tt, device=device)
                    x0 = prior.normalize(roma.rotvec_to_rotmat(pose.view(-1, 21, 3))[:, keep])
                    alpha, std = prior.marginal(vec_t)
                    noised = alpha[:, None] * x0 + std[:, None] * torch.randn_like(x0)
                    with torch.no_grad():
                        x0_hat = (noised + std[:, None] ** 2 * prior.score(noised, vec_t)) / alpha[:, None]
                    sds = (0.5 * (x0 - x0_hat) ** 2).sum() / t
                    losses.append(0.1 * (1 + it) * sds_weight * sds)
                out = body_model(pose_body=pose)
                v = out.v.view(s, t, -1, 3)
                temporal = (v[:, :-1] - v[:, 1:]).norm(dim=-1).mean((1, 2)).sum()
                data = (out.Jtr[:, :BODY_JOINTS].view(s, t, BODY_JOINTS, 3) - noisy).norm(dim=-1).mean((1, 2)).sum()
                losses += [10.0 * (1 + it) * temporal, 100.0 / (1 + it * it) * data]
                optimizer.zero_grad()
                torch.stack(losses).sum().backward()
                optimizer.step()
    return roma.rotvec_to_rotmat(pose.detach().view(s, t, 21, 3))
