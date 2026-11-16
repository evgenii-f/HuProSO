"""Motion denoising as PoseD-Flow evaluates it (Tab. 3 of their paper, motion_denoising.py there; the protocol of
DPoser): 60-frame chunks of HumanEva (AMASS), Gaussian noise of std sigma (40 or 100 mm) on every coordinate of the 22
body joints of SMPL-X neutral (betas 0, root at the identity), the body poses recovered by a prior; MPJPE of the 22
joints (mm, the pelvis included, its error 0) averaged over all frames.

- Sequences (sequences()): the AMASS SMPL-X stageii files, the first and the last 10% of every file cut, resampled to
  30 Hz (np.linspace over the frames, as the commented-out code of their create_chunks.py), cut into chunks of 60
  frames; HumanEva gives the 190 sequences of their paper (supplementary D). Only pose_body is used.
  - test: all of HumanEva (part of the validation split of their AMASS, so no model trained on their training split
    has seen it);
  - validation: a fixed draw of chunks of the other datasets of that split (HDM05, SFU, MoSh), for choosing settings.
- Noise: drawn on the CPU by torch.Generator().manual_seed(index of the sequence), the same for every model.
- Metrics per frame: MPJPE of the 22 joints; MPVPE of all vertices with the toes at rest in the prediction and the
  ground truth (our 19-joint models have no toes; the joints do not depend on the toe rotations); MPJPE of the noisy
  joints (the input). In mm.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import roma
import torch

from hp.evaluation.completion import KEEP, TOES, at_rest, file_identity

LENGTH, FPS, CROP = 60, 30, 0.1
TEST_DATASETS = ("HumanEva",)
VALID_DATASETS = ("HDM05", "SFU", "MoSh")
SIGMAS = (0.04, 0.1)
# their README and Tab. D.1: lr 0.25, Euler 100, start blend 0.75, running cost 1e-5; 300 steps and temporal 1e-2 at
# 40 mm, 400 steps and 1e-1 at 100 mm
POSEDFLOW_SETTINGS = {
    0.04: dict(steps=300, lr=0.25, temp_weight=1e-2, running_weight=1e-5),
    0.1: dict(steps=400, lr=0.25, temp_weight=1e-1, running_weight=1e-5),
}
# DPoser's motion_denoising.py (time strategy 3, AMASS: sds_weight 1)
DPOSER_SETTINGS = {
    0.04: dict(iterations=3, steps_per_iter=60, t_max=0.25, t_min=0.05),
    0.1: dict(iterations=3, steps_per_iter=80, t_max=0.25, t_min=0.1),
}
PAPER = {  # their Tab. 3, MPJPE (mm): AMASS 40 / 100 mm, HPS 40 / 100 mm
    "w/o prior": (24.19, 51.48, 23.67, 50.87),
    "VPoser": (23.42, 49.10, 22.78, 46.69),
    "Pose-NDF": (22.13, 46.10, 21.60, 47.50),
    "DPoser": (19.87, 33.18, 20.54, 35.32),
    "PoseRFM": (18.88, 34.68, 19.79, 32.95),
}
METRICS_VERSION = 2  # 2: Body in chunks of 512 (MPS)


def chunks_of(file: Path) -> List[np.ndarray]:
    """The 60-frame chunks [60, 63] (pose_body) of one AMASS file."""
    data = np.load(file, allow_pickle=True)
    poses, rate = data["poses"], float(data["mocap_frame_rate"])
    n = len(poses)
    poses = poses[int(CROP * n):int((1 - CROP) * n)]
    frames = np.linspace(0, len(poses) - 1, num=int(FPS / rate * len(poses)), dtype=int)
    body = poses[frames, 3:66].astype(np.float32)
    return [body[i * LENGTH:(i + 1) * LENGTH] for i in range(len(body) // LENGTH)]


def build_sequences(amass: str, datasets) -> Tuple[torch.Tensor, List[str]]:
    """All chunks [N, 60, 21, 3, 3] of the stageii files of `datasets` (sorted by path) and their names."""
    poses, names = [], []
    for ds in datasets:
        for file in sorted((Path(amass).expanduser() / ds).glob("*/*_stageii.npz")):
            for i, chunk in enumerate(chunks_of(file)):
                poses.append(chunk)
                names.append(f"{ds}/{file.parent.name}/{file.stem}_batch{i:03d}")
    rot = roma.rotvec_to_rotmat(torch.from_numpy(np.stack(poses)).double().view(-1, LENGTH, 21, 3)).float()
    return rot, names


def sequences(split: str, file: str = "~/data/experiments/denoising/sequences.pt", amass: Optional[str] = None,
              n_valid: int = 100, seed: int = 0) -> Tuple[torch.Tensor, List[str]]:
    """The sequences [N, 60, 21, 3, 3] of "test" (HumanEva, 190) or "valid" (n_valid chunks drawn with `seed` from
    HDM05, SFU, MoSh) and their names. Read from `file`; made from the raw AMASS (`amass`, default the setting
    amass_raw) and saved there if it is missing (the servers get the file, not AMASS)."""
    path = Path(file).expanduser()
    if not path.is_file():
        from hp.utils.paths import get_path

        root = amass or get_path("amass_raw")
        test, test_names = build_sequences(root, TEST_DATASETS)
        valid, valid_names = build_sequences(root, VALID_DATASETS)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"test": test, "test_names": test_names, "valid": valid, "valid_names": valid_names}, path)
    saved = torch.load(path, weights_only=True)
    if split == "test":
        return saved["test"], list(saved["test_names"])
    if split != "valid":
        raise ValueError(f"split must be 'test' or 'valid', got {split}.")
    pick = torch.randperm(len(saved["valid"]), generator=torch.Generator().manual_seed(seed))[:n_valid]
    return saved["valid"][pick], [saved["valid_names"][i] for i in pick]


class Body:
    """SMPL-X neutral, betas 0: vertices and the 22 body joints of poses [B, 21, 3, 3], and the skeleton of the
    joints for the losses."""

    def __init__(self, model_file: Optional[str] = None, device: str = "cpu"):
        from human_body_prior.body_model.body_model import BodyModel

        from hp.models.motion_denoising import Skeleton
        from hp.utils.paths import get_path

        model_file = model_file or str(Path(get_path("body_models")).expanduser() / "smplx" / "SMPLX_NEUTRAL.npz")
        self.model = BodyModel(bm_fname=str(Path(model_file).expanduser())).to(device)
        self.skeleton = Skeleton.from_file(model_file).to(device)
        self.identity = file_identity(model_file)
        self.device = device

    @torch.no_grad()
    def __call__(self, poses: torch.Tensor, chunk: int = 512):
        # chunks of more than 1024 poses give wrong joints (up to 9 m) on MPS with torch 2.13
        verts, joints = [], []
        for i in range(0, len(poses), chunk):
            p = poses[i:i + chunk].float().to(self.device)
            out = self.model(pose_body=roma.rotmat_to_rotvec(p).reshape(len(p), 63))
            verts.append(out.v.cpu())
            joints.append(out.Jtr[:, :22].cpu())
        return torch.cat(verts).double(), torch.cat(joints).double()


def gt_joints(gt: torch.Tensor, body: Body) -> torch.Tensor:
    """The 22 joints [N, T, 22, 3] of the sequences gt [N, T, 21, 3, 3]."""
    return body(gt.flatten(0, 1))[1].view(*gt.shape[:2], 22, 3)


def noisy_joints(joints: torch.Tensor, sigma: float) -> torch.Tensor:
    """joints [N, T, 22, 3] + sigma N(0, I), the noise of sequence i from torch.Generator().manual_seed(i)."""
    out = joints.clone().float()
    for i in range(len(out)):
        g = torch.Generator().manual_seed(i)
        out[i] += sigma * torch.randn(out[i].shape, generator=g)
    return out


def metrics(pred: torch.Tensor, gt: torch.Tensor, noisy: torch.Tensor, body: Body) -> Dict[str, np.ndarray]:
    """Per frame [N, T] (mm): mpjpe of the 22 joints, mpvpe of all vertices (the toes at rest in both), and
    input_mpjpe of the noisy joints."""
    n, t = gt.shape[:2]
    v_gt, j_gt = body(at_rest(gt.flatten(0, 1), TOES))
    v, j = body(at_rest(pred.flatten(0, 1), TOES))
    _, j_raw = body(pred.flatten(0, 1))
    _, j_raw_gt = body(gt.flatten(0, 1))
    mpjpe = (j_raw - j_raw_gt).norm(dim=-1).mean(-1) * 1000
    mpvpe = (v - v_gt).norm(dim=-1).mean(-1) * 1000
    inp = (noisy.double().flatten(0, 1) - j_raw_gt).norm(dim=-1).mean(-1) * 1000
    shape = (n, t)
    return {"mpjpe": mpjpe.view(shape).numpy(), "mpvpe": mpvpe.view(shape).numpy(),
            "input_mpjpe": inp.view(shape).numpy()}


def summary(m: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: float(np.nanmean(v)) for k, v in m.items()}


def run(model, gt: torch.Tensor, noisy: torch.Tensor, sigma: float, body: Body, method: str = "dflow",
        batch: int = 10, seed: int = 0, progress: bool = True, prior: Optional[dict] = None,
        **settings) -> torch.Tensor:
    """Denoised sequences [N, T, 21, 3, 3] of noisy joints [N, T, 22, 3].

    method "dflow": D-Flow on `model` (hp.models.dflow.prior_of; prior: its options, e.g. dict(sampler="heun",
    schedule="edm", ode_steps=25)) with POSEDFLOW_SETTINGS[sigma] updated by `settings`; "dposer": DPoser's
    optimization with DPOSER_SETTINGS[sigma] updated by `settings`, model a DPoser or None (no prior). `batch`
    sequences at a time; the random draws of a batch from torch.manual_seed(seed + index of its first sequence)."""
    from hp.models.motion_denoising import dflow_denoise, dposer_denoise, mean_pose

    out = torch.empty(*gt.shape)
    if method == "dflow":
        from hp.models.dflow import prior_of

        options = dict(prior or {})
        p = prior_of(model, options.pop("ode_steps", 100), **options)
        joints = None if p.joints == 21 else KEEP
        conf = {**POSEDFLOW_SETTINGS[sigma], **settings}
    elif method == "dposer":
        joints = None if model is None or model.num_joints == 21 else KEEP  # DPoser or RSGM
        conf = {**DPOSER_SETTINGS[sigma], **settings}
        start = mean_pose(conf.pop("mean_pose", "~/data/experiments/denoising/smpl_mean_params.npz"))
    else:
        raise ValueError(f"Unknown denoising method '{method}'. Use 'dflow' or 'dposer'.")
    for first in range(0, len(gt), batch):
        y = noisy[first:first + batch]
        torch.manual_seed(seed + first)
        if method == "dflow":
            x = dflow_denoise(p, body.skeleton, y, joints, **conf)
        else:
            init = start.expand(*y.shape[:2], 63) + 0.01 * torch.randn(*y.shape[:2], 63)
            device = model.device if model is not None else body.device
            bm = body.model.to(device)
            x = dposer_denoise(bm, y.to(device), init, model, joints, **conf)
        out[first:first + len(y)] = x.detach().float().cpu()
        if progress:
            print(f"\r  sequences {first + len(y)} / {len(gt)}", end="", flush=True)
    if progress:
        print(flush=True)
    return out


def identity(spec: dict) -> dict:
    """What the results of a spec depend on: the spec and its weights file (none for the model-free row)."""
    if "posedflow" in spec:
        return {"spec": spec, **file_identity(spec["posedflow"])}
    if "run" in spec:
        return {"spec": spec, **file_identity(Path(spec["run"]).expanduser() / f"{spec.get('weights', 'weights_ema')}.pt")}
    return {"spec": spec}


def build(spec: dict, device: str):
    """The model of a spec (as hp.evaluation.completion.build), None for {"method": "dposer"} without a model."""
    if "posedflow" not in spec and "run" not in spec:
        return None
    from hp.evaluation.completion import build as build_model

    return build_model(spec, device)


def cache_path(cache, label: str, split: str, sigma: float) -> Path:
    return Path(cache).expanduser() / label.replace("/", "_").replace(" ", "_") / f"{split}_{round(sigma * 1000)}mm.pt"


def cached_run(cache, label: str, spec: dict, split: str, sigma: float, device: Optional[str], batch: int = 10,
               sequences_file: str = "~/data/experiments/denoising/sequences.pt",
               **overrides) -> Optional[Dict[str, np.ndarray]]:
    """The per-frame metrics of a spec (model, method "dflow"/"dposer", prior options, settings) on a split and a
    noise level, from cache/<label>/<split>_<sigma>mm.pt if made with the same settings; otherwise computed (on
    `device`; None: only read) and saved with the predictions. spec["settings"] is one dict for every noise level
    or a dict {sigma: settings}."""
    path = cache_path(cache, label, split, sigma)
    gt, names = sequences(split, sequences_file)
    settings = spec.get("settings", {})
    settings = {**settings.get(sigma, {} if any(isinstance(k, float) for k in settings) else settings), **overrides}
    key = json.loads(json.dumps({"model": identity(spec), "split": split, "sigma": sigma, "names": names,
                                 "settings": settings, "version": METRICS_VERSION}))
    if path.is_file():
        saved = torch.load(path, weights_only=True)
        if json.loads(saved["key"]) == key:
            return {k: v.numpy() for k, v in saved["metrics"].items()}
        print(f"  {label} / {split} / {sigma}: cached with other settings")
    if device is None:
        return None
    body = Body(device=device)
    noisy = noisy_joints(gt_joints(gt, body), sigma)
    model = build(spec, device)
    pred = run(model, gt, noisy, sigma, body, spec.get("method", "dflow"), batch, prior=spec.get("prior"), **settings)
    m = metrics(pred, gt, noisy, body)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"key": json.dumps(key), "pred": pred.half(), "metrics": {k: torch.from_numpy(v) for k, v in m.items()}},
               tmp)
    tmp.replace(path)
    return m
