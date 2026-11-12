"""Train VPoser from scratch under the protocols of the two papers (as RSGM and DPoser were trained), on our 19 joints.
The three runs:

    python scripts/train_vposer.py --paper huproso --protocol full --device cuda:0
    python scripts/train_vposer.py --paper huproso --protocol c30 --device cuda:0
    python scripts/train_vposer.py --paper posedflow --protocol full --device cuda:0

    python scripts/train_vposer.py --paper huproso --protocol c30 --dry-run      # print the configuration only

--paper, the data and its split:
- huproso: our AMASS, as explore/rsgm_huproso_comparison.ipynb (amass_split original: train CMU, PosePrior,
  TotalCapture, EyesJapanDataset, KIT, BMLrub, EKUT, TCDHands, ACCAD; valid HumanEva, HDM05, SFU, MoSh; test
  Transitions, SSM);
- posedflow: the AMASS of PoseD-Flow and DPoser, as explore/evaluation/pareto_posedflow_vs_rsgm.ipynb (amass_split
  posedflow).
--protocol, how much of it and the training:
- full: all training poses, the training of their V02_05 (config/models/vposer.yaml: batch 128, ~270k steps);
- c30 (huproso only): every 30th pose and the budget of the RSGM c30 run (100k steps of 512), their optimizer and
  schedule otherwise.

Runs (<seed> e.g. 10):
    huproso   full  ~/data/experiments/vposer_amass_s<seed>             (as rsgm: RSGM full)
    huproso   c30   ~/data/experiments/vposer_amass_c30_s<seed>         (as rsgm_c30: RSGM c30)
    posedflow full  ~/data/experiments/full/vposer_posedflow_s<seed>    (as full/dposer_posedflow_s10)

The model and loss are those of V02_05 (hp/models/vposer.py); the loss needs the SMPL-X neutral body model
(model.args.body_model; both datasets are AMASS SMPL-X fits). Every seed is its own run, <run folder>/run_000N:
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
SEEDS = [10]
DEVICE = "mps"  # "cpu", "mps" or "cuda:N"
N_CHECK = 10_000  # samples of the quick check after training (0: no check)
RUN_ROOT = "~/data/experiments"

SPLITS = {"huproso": "original", "posedflow": "posedflow"}  # the amass_split of each paper's data
RSGM_C30_BUDGET = ["train.max_iterations=100000", "train.batch_size=512", "model.args.extra_terms_steps=80000",
                   "train.valid_interval=2000", "train.checkpoint_every=10000"]
RUNS = {  # (paper, protocol): (data, folder under the run root, run name prefix, what differs from config/models/vposer.yaml)
    ("huproso", "full"): ("~/data/AMASS_rotations", "", "vposer_amass", []),
    ("huproso", "c30"): ("~/data/AMASS_rotations_c30", "", "vposer_amass_c30", RSGM_C30_BUDGET),
    ("posedflow", "full"): ("~/data/AMASS_rotations_posedflow", "full", "vposer_posedflow", []),
}
# ----------------------------------------------------------------------------------------


def training_poses(rotations: str, split: str):
    """The number of training poses of the split in the folder, or None if a file is not here."""
    from hp.data.amass_cfg import ORIGINAL_AMASS_SPLITS, POSEDFLOW_AMASS_SPLITS

    names = {"original": ORIGINAL_AMASS_SPLITS, "posedflow": POSEDFLOW_AMASS_SPLITS}[split]["train"]
    files = [Path(rotations).expanduser() / f"{name}.pt" for name in names]
    if not all(f.is_file() for f in files):
        return None
    return sum(torch.load(f, mmap=True, weights_only=True).shape[0] for f in files)


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
    parser.add_argument("--paper", choices=sorted(SPLITS), required=True, help="whose data and split")
    parser.add_argument("--protocol", choices=sorted({p for _, p in RUNS}), required=True,
                        help="full: all poses; c30: every 30th (huproso only)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--rotations", default=None, help="the data folder (default: that of the protocol)")
    parser.add_argument("--run-root", default=RUN_ROOT, help="the runs go to <run root>[/full]/<name>_s<seed>")
    parser.add_argument("--retrain", action="store_true", help="train a new run even if a finished one exists")
    parser.add_argument("--check", type=int, default=N_CHECK, help="samples of the quick check (0: none)")
    parser.add_argument("--dry-run", action="store_true", help="print the configuration and the plan, train nothing")
    args, overrides = parser.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        parser.error(f"unknown arguments {bad}; settings of train_model.py are given as key=value")

    if (args.paper, args.protocol) not in RUNS:
        parser.error(f"--paper {args.paper} has the protocols {sorted(p for q, p in RUNS if q == args.paper)}")
    data, folder, prefix, settings = RUNS[(args.paper, args.protocol)]
    rotations, split = args.rotations or data, SPLITS[args.paper]
    run_root = str(Path(args.run_root) / folder) if folder else args.run_root
    status = 0
    for seed in args.seeds:
        name = f"{prefix}_s{seed}"
        argv = ["model=vposer", f"run_root={run_root}", f"run_name={name}", f"seed={seed}",
                f"data.rotations={rotations}", f"data.split={split}", "data.load_dtype=float32",
                f"train.device={args.device}", *settings, *overrides]
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
                frames = training_poses(rotations, split)
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
