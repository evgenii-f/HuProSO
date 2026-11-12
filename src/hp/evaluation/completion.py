"""Pose completion as PoseD-Flow evaluates it (Tab. 2 of their paper; pose_completion.py and experiment/inverse_utils.py
there): the rotations of a body part are hidden, a model fills them, K hypotheses per pose.

- Data: their AMASS test set every 10th pose, in batches of 500 with the last one dropped (5000 of 5420 poses).
- Occlusions (OCCLUSIONS): their BodyPartIndices, among SMPL joints 1 ... 21 (0-based here). "trunk" hides the three
  spine joints and the two shoulders.
- Metrics, on SMPL-X neutral with betas = 0 (their body model, also for their SMPL poses), per pose over the K
  hypotheses: MPVPE over the vertices of the hidden part (their smplx_vert_segmentation.json) and MPJPE over the
  hidden joints (mm), each as mean, std and min over K; APD of the hidden joints over the K hypotheses (cm). Their
  table reports the mean over the poses of the mean and of the std of MPVPE, and APD. Added here: MGEO of the hidden
  joints (rad).
- Models of 19 joints (ours, without the toes): a hidden toe is set to rest. The "toes at rest" metrics set the toes to
  rest in the ground truth and in the completions of every model, so that 19- and 21-joint models are compared alike.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import roma
import torch

from hp.data.amass_cfg import AmassJoints4Learning

INTERVAL, BATCH = 10, 500  # theirs
OCCLUSIONS = {  # SMPL joints 1 ... 21, 0-based (their BodyPartIndices)
    "left_leg": (0, 3, 6, 9),
    "legs": (0, 1, 3, 4, 6, 7, 9, 10),
    "arms": (12, 13, 15, 16, 17, 18, 19, 20),
    "trunk": (2, 5, 8, 15, 16),
}
SEGMENTS = {  # their BodySegIndices: the parts of smplx_vert_segmentation.json
    "left_leg": ("leftLeg", "leftUpLeg", "leftFoot", "leftToeBase"),
    "legs": ("leftLeg", "leftUpLeg", "leftFoot", "leftToeBase", "rightLeg", "rightUpLeg", "rightFoot", "rightToeBase"),
    "arms": ("leftArm", "leftForeArm", "rightArm", "rightForeArm"),
    "trunk": ("spine1", "spine2", "leftShoulder", "rightShoulder"),
}
STEPS = {"arms": 300}  # their README: 300 optimizer steps for the arms, 200 otherwise
METRICS_VERSION = 1  # of metrics() and at_rest(): change it when what they compute changes, so cached metrics are redone
METHOD = "dflow 1: their start (the sampling ODE run from the observed pose)"  # change it when the method changes
SPEED = ("batch", "together")  # options of run that change only the random draws and the speed, not the method
TOES = (9, 10)
KEEP = [j.value - 1 for j in AmassJoints4Learning]  # our 19 joints among the 21
PAPER = {  # their Tab. 2: MPVPE mean +- std (mm), APD (cm)
    "VPoser": {"left_leg": (200.51, 12.20, 2.41), "legs": (221.34, 16.40, 5.44), "arms": (206.72, 12.91, 4.08), "trunk": (58.71, 7.38, 1.56)},
    "Pose-NDF": {"left_leg": (168.45, 8.66, 1.95), "legs": (169.86, 6.12, 1.97), "arms": (260.94, 4.81, 1.20), "trunk": (114.97, 5.43, 0.93)},
    "CVPoser": {"left_leg": (128.04, 10.36, 1.91), "legs": (134.35, 10.17, 2.43), "arms": (162.82, 5.58, 1.08), "trunk": (51.23, 4.32, 0.57)},
    "DPoser": {"left_leg": (78.31, 27.13, 6.53), "legs": (102.46, 25.39, 7.75), "arms": (104.94, 26.44, 5.69), "trunk": (44.60, 14.65, 2.21)},
    "PoseFM": {"left_leg": (102.60, 34.82, 8.77), "legs": (129.04, 30.93, 9.94), "arms": (140.88, 35.63, 7.85), "trunk": (51.86, 17.25, 2.70)},
    "PoseRFM": {"left_leg": (83.81, 29.07, 6.02), "legs": (95.00, 26.08, 7.23), "arms": (107.36, 30.41, 5.72), "trunk": (39.75, 9.12, 1.21)},
}


def test_poses(rotations: str = "~/data/AMASS_rotations_posedflow_c10", n: Optional[int] = None) -> torch.Tensor:
    """Their test poses [n, 21, 3, 3] (SMPL joints 1 ... 21), every INTERVAL-th, whole batches of BATCH (default)."""
    q = torch.load(Path(rotations).expanduser() / "PoseDFlow_test.pt", weights_only=True)
    poses = roma.unitquat_to_rotmat(torch.as_tensor(q)[::INTERVAL, 1:22].double()).float()
    return poses[: (n if n is not None else len(poses) // BATCH * BATCH)]


def at_rest(poses: torch.Tensor, joints) -> torch.Tensor:
    out = poses.clone()
    out[..., list(joints), :, :] = torch.eye(3, dtype=poses.dtype)
    return out


class Body:
    """SMPL-X neutral, betas = 0: vertices and the 22 body joints of poses [B, 21, 3, 3]."""

    def __init__(self, segmentation: str = "~/Documents/PoseDFlow/examples/smplx_vert_segmentation.json"):
        from hp.utils.paths import get_path
        from hp.visualization.render_tools import load_body_model

        model_file = Path(get_path("body_models")).expanduser() / "smplx" / "SMPLX_NEUTRAL.npz"
        self.model = load_body_model(str(model_file))
        self.identity = {"model": file_identity(model_file), "segmentation": file_identity(segmentation)}  # of the metrics
        parts = json.loads(Path(segmentation).expanduser().read_text())
        self.vertices = {k: sorted(set(sum((parts[p] for p in v), []))) for k, v in SEGMENTS.items()}

    @torch.no_grad()
    def __call__(self, poses: torch.Tensor, chunk: int = 1000):
        verts, joints = [], []
        for i in range(0, len(poses), chunk):
            p = poses[i:i + chunk].float().cpu()
            out = self.model(pose_body=roma.rotmat_to_rotvec(p).reshape(len(p), 63))
            verts.append(out.v)
            joints.append(out.Jtr[:, :22])
        return torch.cat(verts).double(), torch.cat(joints).double()


def metrics(pred: torch.Tensor, gt: torch.Tensor, occlusion: str, body: Body) -> Dict[str, np.ndarray]:
    """Per pose: mpvpe, mpjpe, mgeo [N, K] of the hidden part and apd [N], of completions pred [N, K, 21, 3, 3]."""
    n, k = pred.shape[:2]
    hidden = list(OCCLUSIONS[occlusion])
    vidx, jidx = body.vertices[occlusion], [j + 1 for j in hidden]  # + 1: the pelvis comes first
    v_gt, j_gt = body(gt)
    v, j = body(pred.reshape(n * k, 21, 3, 3))
    v, j = v.view(n, k, *v.shape[1:])[:, :, vidx], j.view(n, k, 22, 3)[:, :, jidx]
    mpvpe = (v - v_gt[:, None, vidx]).norm(dim=-1).mean(-1) * 1000
    mpjpe = (j - j_gt[:, None, jidx]).norm(dim=-1).mean(-1) * 1000
    mgeo = roma.rotmat_geodesic_distance(pred[:, :, hidden].double(), gt[:, None, hidden].double()).mean(-1)
    pairs = (j[:, :, None] - j[:, None]).norm(dim=-1).mean(-1)  # [N, K, K]
    apd = pairs.sum((1, 2)) / (k * (k - 1)) * 100
    return {"mpvpe": mpvpe.numpy(), "mpjpe": mpjpe.numpy(), "mgeo": mgeo.numpy(), "apd": apd.numpy()}


def summary(m: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Their numbers (means over the poses of the mean, std and min over K) and APD."""
    out = {"apd": float(m["apd"].mean())}
    for key in ("mpvpe", "mpjpe", "mgeo"):
        out[f"{key}_mean"] = float(m[key].mean(1).mean())
        out[f"{key}_std"] = float(m[key].std(1).mean())
        out[f"{key}_min"] = float(m[key].min(1).mean())
    return out


def run(model, gt: torch.Tensor, occlusion: str, k: int = 10, steps: Optional[int] = None, ode_steps: int = 100,
        lr: float = 0.1, running_weight: float = 1e-3, batch: int = BATCH, together: bool = False,
        progress: bool = True, prior: Optional[dict] = None) -> torch.Tensor:
    """Completions [N, K, 21, 3, 3] of the poses gt [N, 21, 3, 3] by D-Flow on `model` (hp.models.dflow); hypothesis
    k starts from torch.manual_seed(k) in every batch, as theirs. together=True: the K hypotheses of `batch` poses in
    one batch of K * batch (fewer, larger GPU calls; the hypotheses are independent all the same, as Adam works per
    coordinate), from torch.manual_seed(first pose of the batch): other random draws than theirs. prior: options of
    the D-Flow prior (hp.models.dflow.prior_of), e.g. the ODE of RSGM: dict(sampler="heun", schedule="edm", ode_steps=25)."""
    from hp.models.dflow import complete, prior_of

    prior = dict(prior or {})
    prior = prior_of(model, prior.pop("ode_steps", ode_steps), **prior)
    joints = list(range(21)) if prior.joints == 21 else KEEP
    visible = torch.ones(21, dtype=torch.bool)
    visible[list(OCCLUSIONS[occlusion])] = False
    out = gt[:, None].repeat(1, k, 1, 1, 1)
    for start in range(0, len(gt), batch):
        observed = gt[start:start + batch][:, joints]
        if together:
            b = len(observed)
            torch.manual_seed(start)
            filled = complete(prior, observed.repeat(k, 1, 1, 1), visible[joints], steps or STEPS.get(occlusion, 200), lr,
                              running_weight)  # [k * b], hypothesis-major
            out[start:start + b][:, :, joints] = filled.view(k, b, *filled.shape[1:]).transpose(0, 1).float().cpu()
            if progress:
                print(f"\r  {occlusion}: poses {start + b} / {len(gt)}", end="", flush=True)
            continue
        for h in range(k):
            torch.manual_seed(h)
            filled = complete(prior, observed, visible[joints], steps or STEPS.get(occlusion, 200), lr, running_weight)
            out[start:start + batch, h][:, joints] = filled.float().cpu()
            if progress:
                print(f"\r  {occlusion}: poses {start + len(observed)} / {len(gt)}, hypothesis {h + 1} / {k}", end="", flush=True)
    if prior.joints != 21:  # hidden toes at rest; visible ones stay observed
        hidden_toes = [t for t in TOES if not visible[t]]
        out = at_rest(out, hidden_toes) if hidden_toes else out
    if progress:
        print(flush=True)
    return out


def build(spec: dict, device: str):
    """A model of a spec: posedflow=<checkpoint> (their pretrained PoseD-Flow) or run=<run or checkpoint folder>, with
    optional weights= (default weights_ema). An optional prior= (options of the D-Flow prior, see run) is part of
    the spec and so of the cache entry; a spec without it is the protocol of PoseD-Flow (100 Euler steps)."""
    from hp.models.io import load_model
    from hp.models.posedflow import PoseDFlowPretrained

    if "posedflow" in spec:
        return PoseDFlowPretrained(spec["posedflow"]).to(device).eval()
    return load_model(Path(spec["run"]).expanduser(), device=device, weights=spec.get("weights", "weights_ema")).eval()


def file_identity(path) -> dict:
    """The size and the modification time (whole seconds) of a file: what tells that it is the same file."""
    stat = Path(path).expanduser().stat()
    return {"size": stat.st_size, "mtime": int(stat.st_mtime)}


def identity(spec: dict) -> dict:
    """What the completions of a spec depend on: the spec and the size and time of its weights file."""
    if "posedflow" in spec:
        file = Path(spec["posedflow"]).expanduser()
    else:
        file = Path(spec["run"]).expanduser() / f"{spec.get('weights', 'weights_ema')}.pt"
    return {"spec": spec, **file_identity(file)}


def cache_path(cache: Path, label: str, occlusion: str) -> Path:
    """cache/<label>/<occlusion>.pt, the completions of a model and an occlusion."""
    return Path(cache).expanduser() / label.replace("/", "_").replace(" ", "_") / f"{occlusion}.pt"


def _results_settings(settings: dict) -> dict:
    """The settings without SPEED: those that only change how the work is split, not what is computed."""
    return {key: value for key, value in settings.items() if key not in SPEED}


def cached_run(cache: Path, label: str, spec: dict, gt: torch.Tensor, occlusion: str, device: str, k: int = 10,
               rotations: str = "", **options) -> Optional[torch.Tensor]:
    """The completions of cache/<label>/<occlusion>.pt if they were made with these settings (whatever the SPEED
    settings were); otherwise made and saved (device=None: only read, None if missing)."""
    path = cache_path(cache, label, occlusion)
    settings = json.loads(json.dumps({"model": identity(spec), "poses": len(gt), "rotations": rotations, "k": k,
                                      "method": METHOD, **options}))
    if path.is_file():
        saved = torch.load(path, weights_only=True)
        if _results_settings(json.loads(saved["settings"])) == _results_settings(settings):
            return saved["pred"].float()
        print(f"  {label} / {occlusion}: cached with other settings")
    if device is None:
        return None
    pred = run(build(spec, device), gt, occlusion, k, prior=spec.get("prior"), **options)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"pred": pred.half(), "settings": json.dumps(settings)}, temporary)
    temporary.replace(path)
    return pred


def cached_metrics(cache: Path, label: str, occlusion: str, pred: torch.Tensor, gt: torch.Tensor,
                   body: "Body") -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """The per-pose metrics (metrics()) of the completions pred of cache/<label>/<occlusion>.pt (cached_run), and the
    same with the toes at rest in the completions and the ground truth. Read from cache/<label>/<occlusion>_metrics.pt
    if they were computed from this completions file (its size and time), as many test poses, the same body model and
    segmentation and METRICS_VERSION; otherwise computed and saved there."""
    completions = cache_path(cache, label, occlusion)
    path = completions.with_name(f"{occlusion}_metrics.pt")
    settings = json.loads(json.dumps({"completions": file_identity(completions), "poses": len(gt),
                                      "body": body.identity, "version": METRICS_VERSION}))
    if path.is_file():
        saved = torch.load(path, weights_only=True)
        if json.loads(saved["settings"]) == settings:
            return ({k: v.numpy() for k, v in saved["plain"].items()}, {k: v.numpy() for k, v in saved["rest"].items()})
    plain = metrics(pred, gt, occlusion, body)
    rest = metrics(at_rest(pred, TOES), at_rest(gt, TOES), occlusion, body)
    temporary = path.with_suffix(".tmp")
    torch.save({"settings": json.dumps(settings), "plain": {k: torch.from_numpy(v) for k, v in plain.items()},
                "rest": {k: torch.from_numpy(v) for k, v in rest.items()}}, temporary)
    temporary.replace(path)
    return plain, rest
