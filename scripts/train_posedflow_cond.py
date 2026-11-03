"""Train PoseD-Flow (hp/models/posedflow.py, PoseDFlow) conditioned on 3D keypoints, for the comparison of the
conditional models in explore/evaluation/conditional_flow_occlusion.ipynb (which finds every run with condition "3D"
below ~/data/experiments):

    python scripts/train_posedflow_cond.py                    # mps, our AMASS, their settings
    python scripts/train_posedflow_cond.py --dry-run          # print the configuration only
    python scripts/train_posedflow_cond.py --device cuda:0 train.time_limit_minutes=230

The data are those of the conditional RSGM (explore/training/rsgm_cond_train.ipynb): our AMASS (SMPL-X G) with all
frames and the official split; the condition is the same (hp.models.conditioning: the 22 keypoints, joints hidden at
random in the loss, cond_mask "random"). The network, the loss and the optimizer are theirs (config/models/posedflow.yaml:
4 x 512, AdamW lr 1e-3, weight decay 1e-4, EMA 0.99, 50k steps x batch 4096).

The run is <run root>/<name>/run_000N (default name posedflow_cond3d_s<seed>):
- a finished run is skipped (--retrain trains a new one);
- a run that stopped early (time limit, SIGTERM) is resumed from its newest checkpoint.
Further key=value arguments go to scripts/train_model.py and win over the settings here.

Progress: the plan (the network, the epochs), the trainer's progress bar with the elapsed and remaining time, the time
when it ends, and a quick check: keypoint errors of samples for some occlusions and the conditional log-likelihood.

The exit status is 0 when the run is finished and 75 when it stopped early and can be resumed (as train_model.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import roma
import torch

# the code of this checkout, before an editable install of another one
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hp.data.amass_cfg import ORIGINAL_AMASS_SPLITS, SUBSET_AMASS_SPLITS, AmassJoints, AmassJoints4Learning  # noqa: E402
from hp.data.conditions import keypoints  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.training import best_checkpoint, cli  # noqa: E402
from hp.training.config import load_config  # noqa: E402

# ---- parameters ----------------------------------------------------------------------
ROTATIONS = "~/data/AMASS_rotations"  # all frames, as the conditional RSGM
SPLIT = "original"
RUN_ROOT = "~/data/experiments"
SEED = 10
DEVICE = "mps"  # "cpu", "mps" or "cuda:N"
CONDITIONING = ["model.args.condition=3D", "model.args.cond_mask=random", "model.args.cond_hidden=64",
                "model.args.cond_features=64"]
N_CHECK = 500  # valid poses of the quick check after training (0: no check)
OCCLUSIONS = {  # some occlusions of the HuProSO3 paper (Tab. 3), by the keypoints they hide
    "none": [],
    "L (left leg)": ["LeftUpLeg", "LeftLeg", "LeftFoot"],
    "A+H (left arm and hand)": ["LeftArm", "LeftElbow", "LeftHand"],
}
# ----------------------------------------------------------------------------------------


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{seconds / 3600:.2f} h"


def train_frames(rotations: str, split: str) -> int:
    names = (SUBSET_AMASS_SPLITS if split == "subset" else ORIGINAL_AMASS_SPLITS)["train"]
    total = 0
    for name in names:
        path = Path(rotations).expanduser() / f"{name}.pt"
        try:
            total += torch.load(path, mmap=True, weights_only=True).shape[0]
        except RuntimeError:  # an old file format without mmap
            total += torch.load(path, weights_only=True).shape[0]
    return total


def runs_of(root: Path):
    return sorted(p for p in root.glob("run_[0-9][0-9][0-9][0-9]") if p.is_dir())


def finished(run: Path) -> bool:
    status = run / "status.json"
    return status.is_file() and json.loads(status.read_text()).get("finished", False) and (run / "weights.pt").is_file()


def resumable(run: Path) -> bool:
    return not finished(run) and any((c / "train_state.pt").is_file() for c in run.glob("checkpoint_*"))


def quick_check(run: Path, device: str, rotations: str, split: str, n: int, seed: int) -> None:
    """The best checkpoint, EMA weights: one sample per valid pose and occlusion (keypoint errors of the hidden and the
    visible joints, the angle to the true pose), and log p(pose | keypoints) of up to 100 poses."""
    checkpoint = best_checkpoint(run) or run
    model = load_model(checkpoint, device=device, weights="weights_ema")
    print(f"  check of {checkpoint} (EMA weights), {n} valid poses:", flush=True)
    valid = load_reference("valid", rotations, split)
    gt = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(seed))[:n]]
    kp = keypoints(gt)
    joints = len(AmassJoints4Learning)
    for occlusion, names in OCCLUSIONS.items():
        hidden = [AmassJoints[name].value for name in names]
        visible = [j for j in range(22) if j not in hidden]
        cond = kp.clone()
        cond[:, hidden] = float("nan")
        torch.manual_seed(seed)
        s = model.sample(n, cond.reshape(n, -1)).cpu().reshape(n, joints, 3, 3)
        err = (keypoints(s) - kp).norm(dim=-1) * 1000  # [n, 22] mm
        hidden_err = f"{err[:, hidden].mean():6.1f}" if hidden else "     -"
        print(f"    {occlusion:26s} MPJPE {err.mean():6.1f} mm (visible {err[:, visible].mean():5.1f}, hidden "
              f"{hidden_err}), MGEO {roma.rotmat_geodesic_distance(s, gt).mean():.3f} rad", flush=True)
    m = min(100, n)
    with torch.no_grad():
        ll = model.log_prob(gt[:m].to(model.device), kp[:m].reshape(m, -1)).cpu()
    print(f"    log p(pose | all keypoints) of {m} poses: mean {ll.mean():.2f}, {(~torch.isfinite(ll)).sum()} non-finite",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--rotations", default=ROTATIONS)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--name", default=None, help="the run name (default posedflow_cond3d_s<seed>)")
    parser.add_argument("--retrain", action="store_true", help="train a new run even if a finished one exists")
    parser.add_argument("--check", type=int, default=N_CHECK, help="valid poses of the quick check (0: none)")
    parser.add_argument("--dry-run", action="store_true", help="print the configuration and the plan, train nothing")
    args, overrides = parser.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        parser.error(f"unknown arguments {bad}; settings of train_model.py are given as key=value")
    name = args.name or f"posedflow_cond3d_s{args.seed}"
    argv = ["model=posedflow", f"run_root={args.run_root}", f"run_name={name}", f"seed={args.seed}",
            f"data.rotations={args.rotations}", f"data.split={args.split}", "data.load_dtype=float32",
            f"train.device={args.device}", *CONDITIONING, *overrides]
    frames = train_frames(args.rotations, args.split)
    runs = runs_of(Path(args.run_root).expanduser() / name)
    done = [r for r in runs if finished(r)]
    stopped = [r for r in runs if resumable(r)]
    if done and not args.retrain:
        print(f"{name}: finished already, {done[-1]} (--retrain trains a new run)", flush=True)
        if args.check and not args.dry_run:
            quick_check(done[-1], args.device, args.rotations, args.split, args.check, args.seed)
        return 0
    cfg = load_config(argv)
    t = cfg.train
    a = cfg.model.args
    plan = (f"{name} on {args.rotations} (split {args.split}, {frames:,} training poses), device {args.device}\n"
            f"  tMLP {a.num_layers} x {a.d_model}, condition {a.condition} (mask {a.cond_mask}), lr {t.lr}, "
            f"{t.max_iterations:,} steps x batch {t.batch_size} = {t.max_iterations * t.batch_size / frames:.1f} epochs")
    if stopped and not args.retrain:
        # resuming takes the settings the run began with; only the resumable ones (e.g. the time limit) may change
        argv = [f"resume={stopped[-1]}", f"train.device={args.device}", *overrides]
        plan += f"\n  resuming {stopped[-1]}"
    if args.dry_run:
        print(f"{plan}\n  would run scripts/train_model.py {' '.join(argv)}", flush=True)
        return 0
    print(f"{plan}\n  started {time.strftime('%H:%M')}", flush=True)
    start = time.perf_counter()
    run = cli.main(argv)
    seconds = time.perf_counter() - start
    times = run / "train_time.json"
    previous = json.loads(times.read_text())["seconds"] if times.is_file() else 0.0  # resumed: add up the sessions
    times.write_text(json.dumps({"seconds": previous + seconds, "device": args.device}))
    code = cli.exit_code(run)
    print(f"{name}: {'finished' if code == 0 else 'stopped early (run the same command to resume)'} after "
          f"{minutes(seconds)}, {run}", flush=True)
    if code == 0 and args.check:
        quick_check(run, args.device, args.rotations, args.split, args.check, args.seed)
    return code


if __name__ == "__main__":
    sys.exit(main())
