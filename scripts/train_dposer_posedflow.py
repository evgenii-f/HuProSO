"""Train DPoser from scratch on the AMASS of PoseD-Flow, the data of the RSGM runs, for the three-way comparison (the
script version of explore/training/dposer_posedflow_train.ipynb, e.g. for a cluster):

    python scripts/train_dposer_posedflow.py --seeds 10 11 --device cuda:0
    python scripts/train_dposer_posedflow.py --config no_sigma_scale --dry-run      # print the configuration only
    python scripts/train_dposer_posedflow.py train.time_limit_minutes=230

The data is their AMASS converted by scripts/preprocess/convert_posedflow.py (train every 10th pose, valid every
100th), our 19 joints. The model and its training are those of DPoser-body (config/models/dposer.yaml); CONFIGS changes
them for the controls. The z-score normalization of the model is fitted on the training poses once and kept in
<run root>/dposer_posedflow_normalization.json (shared with the notebook). Every seed is its own run,
<run root>/dposer_posedflow[_<config>]_s<seed>/run_000N:
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
from hp.models.dposer import normalization  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.training import cli  # noqa: E402
from hp.training.config import load_config  # noqa: E402

# ---- parameters ----------------------------------------------------------------------
ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"
SPLIT = "posedflow"
RUN_ROOT = "~/data/experiments"
SEEDS = [10]
DEVICE = "mps"  # "cpu", "mps" or "cuda:N"
N_CHECK = 10_000  # samples of the quick check after training (0: no check)

COMMON = ["train.valid_interval=2000", "train.checkpoint_every=10000"]
CONFIGS = {  # what differs from config/models/dposer.yaml
    "paper": [],  # DPoser-body: 200k steps of 5120, about 4 h on an M-series GPU
    "no_sigma_scale": ["model.args.scale_by_sigma=false"],  # without their output scaling (a control)
    "rsgm_budget": ["train.max_iterations=30000", "train.batch_size=4096", "train.warmup_steps=1000",
                    "train.valid_interval=1000", "train.checkpoint_every=5000"],  # the budget of the RSGM runs
}
# ----------------------------------------------------------------------------------------


def run_name(config: str, seed: int) -> str:
    return f"dposer_posedflow{'' if config == 'paper' else '_' + config}_s{seed}"


def runs_of(root: Path):
    return sorted(p for p in root.glob("run_[0-9][0-9][0-9][0-9]") if p.is_dir())


def finished(run: Path) -> bool:
    status = run / "status.json"
    return status.is_file() and json.loads(status.read_text()).get("finished", False) and (run / "weights.pt").is_file()


def resumable(run: Path) -> bool:
    return not finished(run) and any((c / "train_state.pt").is_file() for c in run.glob("checkpoint_*"))


def fitted_normalization(rotations: str, run_root: str) -> dict:
    """The mean and std of the axis-angle coordinates of the training poses, from the shared file or fitted now."""
    path = Path(run_root).expanduser() / "dposer_posedflow_normalization.json"
    frames = torch.load(Path(rotations).expanduser() / "PoseDFlow_train.pt", mmap=True, weights_only=True).shape[0]
    norm = json.loads(path.read_text()) if path.is_file() else None
    if norm is None or norm["rotations"] != rotations or norm["poses"] != frames:
        train = load_reference("train", rotations, SPLIT)
        mean, std = normalization(train)
        norm = {"rotations": rotations, "split": SPLIT, "poses": len(train), "mean": mean, "std": std}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(norm, indent=1))
        print(f"normalization fitted on {len(train):,} training poses, written to {path}", flush=True)
    return norm


def quick_check(run: Path, device: str, n: int, seed: int) -> None:
    """The EMA weights: valid rotations, and samples with mean ~0 and std ~1 in the normalized space of the model."""
    model = load_model(run, device=device, weights="weights_ema")
    torch.manual_seed(seed)
    samples = model.sample(n)
    with torch.no_grad():
        x = model.normalize(samples).cpu().double()
    samples = samples.cpu().double()
    orthogonality = (samples.transpose(-1, -2) @ samples - torch.eye(3, dtype=torch.float64)).abs().max()
    print(f"  check: max |R^T R - I| = {orthogonality:.1e}; normalized samples: |mean| <= {x.mean(0).abs().max():.3f}, "
          f"std in [{x.std(0).min():.3f}, {x.std(0).max():.3f}] (should be ~1)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="paper")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--rotations", default=ROTATIONS)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--retrain", action="store_true", help="train a new run even if a finished one exists")
    parser.add_argument("--check", type=int, default=N_CHECK, help="samples of the quick check (0: none)")
    parser.add_argument("--dry-run", action="store_true", help="print the configuration and the plan, train nothing")
    args, overrides = parser.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        parser.error(f"unknown arguments {bad}; settings of train_model.py are given as key=value")

    norm = fitted_normalization(args.rotations, args.run_root)
    status = 0
    for seed in args.seeds:
        name = run_name(args.config, seed)
        argv = ["model=dposer", f"run_root={args.run_root}", f"run_name={name}", f"seed={seed}",
                f"data.rotations={args.rotations}", f"data.split={SPLIT}", "data.load_dtype=float32",
                f"train.device={args.device}", f"model.args.mean={json.dumps(norm['mean'])}",
                f"model.args.std={json.dumps(norm['std'])}", *COMMON, *CONFIGS[args.config], *overrides]
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
            resuming = stopped and not args.retrain
            shown = [a if not a.startswith(("model.args.mean=", "model.args.std=")) else a.split("=")[0] + "=[57 numbers]" for a in argv]
            print(f"seed {seed}: would run scripts/train_model.py {' '.join(shown)}")
            if not resuming:
                cfg = load_config(argv)
                frames = norm["poses"]
                t, m = cfg.train, cfg.model.args
                print(f"  {t.max_iterations:,} iterations x batch {t.batch_size} = {t.max_iterations * t.batch_size / frames:.1f} "
                      f"epochs over {frames:,} training poses; TimeFC {m.hidden} x {m.blocks} blocks, scale_by_sigma "
                      f"{m.scale_by_sigma}, {t.optimizer} lr {t.lr}, {t.scheduler} with {t.warmup_steps} warm-up steps")
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
            quick_check(run, args.device, args.check, seed)
    return status


if __name__ == "__main__":
    sys.exit(main())
