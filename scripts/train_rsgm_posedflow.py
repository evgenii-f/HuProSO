"""Train RSGM with the rounded settings of the comparison with PoseD-Flow, on their AMASS or on ours (the script
version of explore/training/rsgm_posedflow_train.ipynb, e.g. for a cluster):

    python scripts/train_rsgm_posedflow.py --config large --seeds 10 11 --device cuda:0
    python scripts/train_rsgm_posedflow.py --config small --dry-run          # print the configuration only
    python scripts/train_rsgm_posedflow.py --config search train.time_limit_minutes=230
    python scripts/train_rsgm_posedflow.py --data amass --config small search    # our AMASS, two networks in a row

--data picks the data (DATASETS): posedflow, their AMASS converted by scripts/preprocess/convert_posedflow.py (train
every 10th pose, valid every 100th), or amass, our AMASS (SMPL-X G, the official split) compressed to every 10th frame,
the arena of the HuProSO3 comparison (explore/evaluation/rsgm_huproso_comparison.ipynb). --rotations overrides the
folder. The settings are rounded values from the plateau of the last Optuna search on our AMASS (not tuned for either
data); CONFIGS picks the network, and several configs are trained one after the other. Every config and seed is its own
run, <run root>/<name>/run_000N, with <name> rsgm_posedflow[_<config>]_s<seed> on their data and
rsgm_amass_<folder>[_<config>]_s<seed> on ours (<folder>: c10 for AMASS_rotations_c10, full for AMASS_rotations):
- a finished run is skipped (--retrain trains a new one);
- a run that stopped early (time limit, SIGTERM) is resumed from its newest checkpoint.
Further key=value arguments go to scripts/train_model.py and win over the settings here.

Progress: a header per run ([k/n], the network, the epochs), the trainer's progress bar with the elapsed and remaining
time of the run, its time when it ends, and a summary of all runs at the end.

The exit status is 0 when every run is finished and 75 when one stopped early and can be resumed (as train_model.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

# the code of this checkout, before an editable install of another one
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.training import cli  # noqa: E402
from hp.training.config import load_config  # noqa: E402

# ---- parameters ----------------------------------------------------------------------
DATASETS = {  # --data: the rotation folder and the split
    "posedflow": dict(rotations="~/data/AMASS_rotations_posedflow_c10", split="posedflow"),
    "amass": dict(rotations="~/data/AMASS_rotations_c10", split="original"),
}
RUN_ROOT = "~/data/experiments"
SEEDS = [10]
DEVICE = "mps"  # "cpu", "mps" or "cuda:N"
N_CHECK = 500  # valid poses for the quick likelihood check after training (0: no check)

COMMON = [  # rounded from the plateau of the search
    "model.args.activation=silu",
    "model.args.beta_f=7.5",
    "model.args.loss_t_min=0.002",
    "model.args.weighting=std",
    "train.lr=0.002",
    "train.batch_size=4096",
    "train.ema_decay=0.999",
    "train.scheduler=step",
    "train.optimizer=adamw",
    "train.weight_decay=1e-5",
    "train.max_iterations=30000",
    "train.valid_interval=1000",
    "train.checkpoint_every=5000",
]
CONFIGS = {  # the network, and what differs from COMMON
    "search": ["model.args.hidden=[1536, 1536, 1536, 1536, 1536, 1536]"],  # 12.2M parameters
    "small": ["model.args.hidden=[512, 512, 512, 512]"],  # the size of PoseD-Flow's network
    "large": ["model.args.hidden=[4096, 4096, 4096, 4096, 4096, 4096]", "train.lr=1e-3"],  # 84.8M parameters
}
# ----------------------------------------------------------------------------------------


def run_name(data: str, rotations: str, config: str, seed: int) -> str:
    suffix = "" if config == "search" else "_" + config
    if data == "posedflow":
        return f"rsgm_posedflow{suffix}_s{seed}"
    folder = Path(rotations).expanduser().name.replace("AMASS_rotations", "").strip("_") or "full"
    return f"rsgm_amass_{folder}{suffix}_s{seed}"


def train_frames(rotations: str, split: str) -> int:
    """The number of training poses of the split, from the headers of the files where possible."""
    from hp.data.amass_cfg import ORIGINAL_AMASS_SPLITS, POSEDFLOW_AMASS_SPLITS, SUBSET_AMASS_SPLITS

    names = {"posedflow": POSEDFLOW_AMASS_SPLITS, "subset": SUBSET_AMASS_SPLITS}.get(split, ORIGINAL_AMASS_SPLITS)["train"]
    total = 0
    for name in names:
        path = Path(rotations).expanduser() / f"{name}.pt"
        try:
            total += torch.load(path, mmap=True, weights_only=True).shape[0]
        except RuntimeError:  # an old file format without mmap
            total += torch.load(path, weights_only=True).shape[0]
    return total


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{seconds / 3600:.2f} h"


def runs_of(root: Path):
    return sorted(p for p in root.glob("run_[0-9][0-9][0-9][0-9]") if p.is_dir())


def finished(run: Path) -> bool:
    status = run / "status.json"
    return status.is_file() and json.loads(status.read_text()).get("finished", False) and (run / "weights.pt").is_file()


def resumable(run: Path) -> bool:
    return not finished(run) and any((c / "train_state.pt").is_file() for c in run.glob("checkpoint_*"))


def quick_check(run: Path, device: str, rotations: str, split: str, n: int, seed: int) -> None:
    """The EMA weights: valid rotations, and a finite log-likelihood of n valid poses."""
    model = load_model(run, device=device, weights="weights_ema")
    torch.manual_seed(seed)
    samples = model.sample(1000).cpu().double()
    orthogonality = (samples.transpose(-1, -2) @ samples - torch.eye(3, dtype=torch.float64)).abs().max()
    valid = load_reference("valid", rotations, split)
    poses = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(seed))[:n]]
    with torch.no_grad():
        ll = torch.cat([model.log_prob(poses[i:i + 250].to(model.device)).cpu() for i in range(0, len(poses), 250)])
    print(f"  check: max |R^T R - I| = {orthogonality:.1e}, valid log-likelihood {ll.mean():.2f} +- "
          f"{ll.std() / len(ll) ** 0.5:.2f} (s.e.m., {len(ll)} poses), {(~torch.isfinite(ll)).sum()} non-finite", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", choices=sorted(CONFIGS), default=["search"],
                        help="one or more networks, trained one after the other")
    parser.add_argument("--data", choices=sorted(DATASETS), default="posedflow")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--rotations", default=None, help="the rotation folder (default: that of --data)")
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--retrain", action="store_true", help="train a new run even if a finished one exists")
    parser.add_argument("--check", type=int, default=N_CHECK, help="valid poses of the quick check (0: none)")
    parser.add_argument("--dry-run", action="store_true", help="print the configuration and the plan, train nothing")
    args, overrides = parser.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        parser.error(f"unknown arguments {bad}; settings of train_model.py are given as key=value")
    rotations = args.rotations or DATASETS[args.data]["rotations"]
    split = DATASETS[args.data]["split"]
    frames = train_frames(rotations, split)
    jobs = [(config, seed) for config in args.config for seed in args.seeds]
    print(f"{len(jobs)} run(s) on {rotations} (split {split}, {frames:,} training poses), device {args.device}", flush=True)

    status, summary, started = 0, [], time.perf_counter()
    for k, (config, seed) in enumerate(jobs, 1):
        name = run_name(args.data, rotations, config, seed)
        argv = ["model=rsgm", f"run_root={args.run_root}", f"run_name={name}", f"seed={seed}",
                f"data.rotations={rotations}", f"data.split={split}", "data.load_dtype=float32",
                f"train.device={args.device}", *COMMON, *CONFIGS[config], *overrides]
        runs = runs_of(Path(args.run_root).expanduser() / name)
        done = [r for r in runs if finished(r)]
        stopped = [r for r in runs if resumable(r)]
        head = f"[{k}/{len(jobs)}] {name}"
        if done and not args.retrain:
            print(f"{head}: finished already, {done[-1]} (--retrain trains a new run)", flush=True)
            summary.append((name, "finished already", None))
            continue
        resuming = bool(stopped) and not args.retrain
        cfg = load_config(argv)
        t = cfg.train
        plan = (f"hidden {list(cfg.model.args.hidden)}, lr {t.lr}, {t.max_iterations:,} steps x batch {t.batch_size} = "
                f"{t.max_iterations * t.batch_size / frames:.1f} epochs")
        if resuming:
            # resuming takes the settings the run began with; only the resumable ones (e.g. the time limit) may change
            argv = [f"resume={stopped[-1]}", f"train.device={args.device}", *overrides]
            plan = f"resuming {stopped[-1]}; " + plan
        if args.dry_run:
            print(f"{head}: {plan}\n  would run scripts/train_model.py {' '.join(argv)}", flush=True)
            continue
        print(f"\n{head}: {plan}; started {time.strftime('%H:%M')}, {minutes(time.perf_counter() - started)} into the script",
              flush=True)
        t0 = time.perf_counter()
        run = cli.main(argv)
        seconds = time.perf_counter() - t0
        times = run / "train_time.json"
        previous = json.loads(times.read_text())["seconds"] if times.is_file() else 0.0  # resumed: add up the sessions
        times.write_text(json.dumps({"seconds": previous + seconds, "device": args.device}))
        code = cli.exit_code(run)
        state = "finished" if code == 0 else "stopped early"
        print(f"{head}: {state} after {minutes(seconds)}, {run}; {len(jobs) - k} run(s) left", flush=True)
        summary.append((name, state, seconds))
        if code != 0:
            status = code
            continue
        if args.check:
            quick_check(run, args.device, rotations, split, args.check, seed)

    if not args.dry_run:
        print(f"\nall runs ({minutes(time.perf_counter() - started)}):", flush=True)
        for name, state, seconds in summary:
            print(f"  {name:40s} {state}{'' if seconds is None else ', ' + minutes(seconds)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
