"""Train RSGM on the AMASS of PoseD-Flow, for the comparison with their pretrained prior (the script version of
explore/training/rsgm_posedflow_train.ipynb, e.g. for a cluster):

    python scripts/train_rsgm_posedflow.py --config large --seeds 10 11 --device cuda:0
    python scripts/train_rsgm_posedflow.py --config small --dry-run          # print the configuration only
    python scripts/train_rsgm_posedflow.py --config search train.time_limit_minutes=230

The data is their AMASS converted by scripts/preprocess/convert_posedflow.py (train every 10th pose, valid every
100th). The settings are rounded values from the plateau of the last Optuna search on our AMASS (not tuned for this
data); CONFIGS picks the network. Every seed is its own run, <run root>/rsgm_posedflow[_<config>]_s<seed>/run_000N:
- a finished run is skipped (--retrain trains a new one);
- a run that stopped early (time limit, SIGTERM) is resumed from its newest checkpoint.
Further key=value arguments go to scripts/train_model.py and win over the settings here.

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
ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"
SPLIT = "posedflow"
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


def run_name(config: str, seed: int) -> str:
    return f"rsgm_posedflow{'' if config == 'search' else '_' + config}_s{seed}"


def runs_of(root: Path):
    return sorted(p for p in root.glob("run_[0-9][0-9][0-9][0-9]") if p.is_dir())


def finished(run: Path) -> bool:
    status = run / "status.json"
    return status.is_file() and json.loads(status.read_text()).get("finished", False) and (run / "weights.pt").is_file()


def resumable(run: Path) -> bool:
    return not finished(run) and any((c / "train_state.pt").is_file() for c in run.glob("checkpoint_*"))


def quick_check(run: Path, device: str, rotations: str, n: int, seed: int) -> None:
    """The EMA weights: valid rotations, and a finite log-likelihood of n valid poses."""
    model = load_model(run, device=device, weights="weights_ema")
    torch.manual_seed(seed)
    samples = model.sample(1000).cpu().double()
    orthogonality = (samples.transpose(-1, -2) @ samples - torch.eye(3, dtype=torch.float64)).abs().max()
    valid = load_reference("valid", rotations, SPLIT)
    poses = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(seed))[:n]]
    with torch.no_grad():
        ll = torch.cat([model.log_prob(poses[i:i + 250].to(model.device)).cpu() for i in range(0, len(poses), 250)])
    print(f"  check: max |R^T R - I| = {orthogonality:.1e}, valid log-likelihood {ll.mean():.2f} +- "
          f"{ll.std() / len(ll) ** 0.5:.2f} (s.e.m., {len(ll)} poses), {(~torch.isfinite(ll)).sum()} non-finite", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="search")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--rotations", default=ROTATIONS)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--retrain", action="store_true", help="train a new run even if a finished one exists")
    parser.add_argument("--check", type=int, default=N_CHECK, help="valid poses of the quick check (0: none)")
    parser.add_argument("--dry-run", action="store_true", help="print the configuration and the plan, train nothing")
    args, overrides = parser.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        parser.error(f"unknown arguments {bad}; settings of train_model.py are given as key=value")

    status = 0
    for seed in args.seeds:
        name = run_name(args.config, seed)
        argv = ["model=rsgm", f"run_root={args.run_root}", f"run_name={name}", f"seed={seed}",
                f"data.rotations={args.rotations}", f"data.split={SPLIT}", "data.load_dtype=float32",
                f"train.device={args.device}", *COMMON, *CONFIGS[args.config], *overrides]
        runs = runs_of(Path(args.run_root).expanduser() / name)
        done = [r for r in runs if finished(r)]
        stopped = [r for r in runs if resumable(r)]
        if done and not args.retrain:
            print(f"seed {seed}: finished already, {done[-1]} (--retrain trains a new run)", flush=True)
            continue
        if stopped and not args.retrain:
            # resuming takes the settings the run began with; only the resumable ones (e.g. the time limit) may change
            argv = [f"resume={stopped[-1]}", f"train.device={args.device}", *overrides]
            print(f"seed {seed}: resuming {stopped[-1]}", flush=True)
        if args.dry_run:
            cfg = load_config([a for a in argv if not a.startswith("resume=")]) if not stopped or args.retrain else None
            print(f"seed {seed}: would run scripts/train_model.py {' '.join(argv)}")
            if cfg is not None:
                frames = torch.load(Path(args.rotations).expanduser() / "PoseDFlow_train.pt", mmap=True, weights_only=True).shape[0]
                t = cfg.train
                print(f"  {t.max_iterations:,} iterations x batch {t.batch_size} = "
                      f"{t.max_iterations * t.batch_size / frames:.1f} epochs over {frames:,} training poses; "
                      f"hidden {list(cfg.model.args.hidden)}, lr {t.lr}")
            continue
        start = time.perf_counter()
        run = cli.main(argv)
        seconds = time.perf_counter() - start
        times = run / "train_time.json"
        previous = json.loads(times.read_text())["seconds"] if times.is_file() else 0.0  # resumed: add up the sessions
        times.write_text(json.dumps({"seconds": previous + seconds, "device": args.device}))
        code = cli.exit_code(run)
        print(f"seed {seed}: {'finished' if code == 0 else 'stopped early'} after {seconds / 60:.1f} min, {run}", flush=True)
        if code != 0:
            status = code
            continue
        if args.check:
            quick_check(run, args.device, args.rotations, args.check, seed)
    return status


if __name__ == "__main__":
    sys.exit(main())
