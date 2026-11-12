"""Train VPoser from scratch on the AMASS of PoseD-Flow, under the two protocols of the Pareto comparison
(hp.evaluation.pareto), as DPoser and RSGM were:

    python scripts/train_vposer_posedflow.py --protocol c10 --device mps
    python scripts/train_vposer_posedflow.py --protocol full --device cuda:0 --seeds 10 11
    python scripts/train_vposer_posedflow.py --protocol full --dry-run          # print the configuration only

Protocols (the data, where the runs go, what differs from config/models/vposer.yaml):
- full: all their training poses (AMASS_rotations_posedflow), the training of their V02_05 (batch 128, ~270k steps);
  runs in <run root>/full, as the "their full" DPoser and RSGM;
- c10: every 10th training pose (AMASS_rotations_posedflow_c10) and the budget of the RSGM runs (30k steps of 4096),
  their optimizer and schedule otherwise, as "DPoser: their c10 (RSGM budget)".

The model and loss are those of V02_05 (hp/models/vposer.py), on our 19 joints; the loss needs the SMPL-X neutral body
model (model.args.body_model). Every seed is its own run, <run root>/vposer_posedflow[_rsgm_budget]_s<seed>/run_000N:
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

from hp.models.io import load_model  # noqa: E402
from hp.training import cli  # noqa: E402
from hp.training.config import load_config  # noqa: E402

# ---- parameters ----------------------------------------------------------------------
SPLIT = "posedflow"
SEEDS = [10]
DEVICE = "mps"  # "cpu", "mps" or "cuda:N"
N_CHECK = 10_000  # samples of the quick check after training (0: no check)

PROTOCOLS = {
    "full": dict(rotations="~/data/AMASS_rotations_posedflow", run_root="~/data/experiments/full", name="vposer_posedflow",
                 settings=[]),  # their V02_05 training, config/models/vposer.yaml
    "c10": dict(rotations="~/data/AMASS_rotations_posedflow_c10", run_root="~/data/experiments",
                name="vposer_posedflow_rsgm_budget",
                settings=["train.max_iterations=30000", "train.batch_size=4096", "model.args.extra_terms_steps=24000",
                          "train.valid_interval=1000", "train.checkpoint_every=5000"]),  # the budget of the RSGM runs
}
# ----------------------------------------------------------------------------------------


def runs_of(root: Path):
    return sorted(p for p in root.glob("run_[0-9][0-9][0-9][0-9]") if p.is_dir())


def finished(run: Path) -> bool:
    status = run / "status.json"
    return status.is_file() and json.loads(status.read_text()).get("finished", False) and (run / "weights.pt").is_file()


def resumable(run: Path) -> bool:
    return not finished(run) and any((c / "train_state.pt").is_file() for c in run.glob("checkpoint_*"))


def quick_check(run: Path, device: str, n: int, seed: int) -> None:
    """Valid rotations, and how far the samples are from the identity (a collapsed prior samples near its mean)."""
    model = load_model(run, device=device)
    torch.manual_seed(seed)
    samples = model.sample(n).cpu().double()
    orthogonality = (samples.transpose(-1, -2) @ samples - torch.eye(3, dtype=torch.float64)).abs().max()
    angle = torch.acos(((samples.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)).rad2deg()
    print(f"  check: max |R^T R - I| = {orthogonality:.1e}; angle to the identity: mean {angle.mean():.1f} deg, "
          f"spread over the samples {angle.mean(-1).std():.1f} deg", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--rotations", default=None, help="the data folder (default: that of the protocol)")
    parser.add_argument("--run-root", default=None, help="where the runs go (default: that of the protocol)")
    parser.add_argument("--retrain", action="store_true", help="train a new run even if a finished one exists")
    parser.add_argument("--check", type=int, default=N_CHECK, help="samples of the quick check (0: none)")
    parser.add_argument("--dry-run", action="store_true", help="print the configuration and the plan, train nothing")
    args, overrides = parser.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        parser.error(f"unknown arguments {bad}; settings of train_model.py are given as key=value")

    protocol = PROTOCOLS[args.protocol]
    rotations = args.rotations or protocol["rotations"]
    run_root = args.run_root or protocol["run_root"]
    status = 0
    for seed in args.seeds:
        name = f"{protocol['name']}_s{seed}"
        argv = ["model=vposer", f"run_root={run_root}", f"run_name={name}", f"seed={seed}",
                f"data.rotations={rotations}", f"data.split={SPLIT}", "data.load_dtype=float32",
                f"train.device={args.device}", *protocol["settings"], *overrides]
        runs = runs_of(Path(run_root).expanduser() / name)
        done = [r for r in runs if finished(r)]
        stopped = [r for r in runs if resumable(r)]
        if done and not args.retrain:
            print(f"seed {seed}: finished already, {done[-1]} (--retrain trains a new run)", flush=True)
            continue
        resuming = bool(stopped) and not args.retrain
        if resuming:
            # resuming takes the settings the run began with; only the resumable ones (e.g. the time limit) may change
            argv = [f"resume={stopped[-1]}", f"train.device={args.device}", *overrides]
            print(f"seed {seed}: resuming {stopped[-1]}", flush=True)
        if args.dry_run:
            print(f"seed {seed}: would run scripts/train_model.py {' '.join(argv)}")
            if not resuming:
                cfg = load_config(argv)
                t, m = cfg.train, cfg.model.args
                frames = torch.load(Path(rotations).expanduser() / "PoseDFlow_train.pt", mmap=True, weights_only=True).shape[0] \
                    if (Path(rotations).expanduser() / "PoseDFlow_train.pt").is_file() else None
                epochs = f" = {t.max_iterations * t.batch_size / frames:.1f} epochs over {frames:,} training poses" if frames else ""
                print(f"  {t.max_iterations:,} iterations x batch {t.batch_size}{epochs}; {t.optimizer} lr {t.lr} weight decay "
                      f"{t.weight_decay}, x{t.lr_gamma} at {t.lr_step_fraction:.0%}; extra terms for {m.extra_terms_steps:,} "
                      f"steps; body model {m.body_model}")
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
