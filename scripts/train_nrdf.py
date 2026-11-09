"""Train NRDF (hp/models/nrdf.py) from scratch, with the settings of their released model, e.g. on a server:

    python scripts/train_nrdf.py --device cuda:0                       # their AMASS, every 10th training pose
    python scripts/train_nrdf.py --data amass --device cuda:0          # our AMASS, every 10th frame
    python scripts/train_nrdf.py --rotations ~/data/AMASS_rotations_posedflow --device cuda:0 --name nrdf_posedflow_full_s10
    python scripts/train_nrdf.py --dry-run                             # print the plan only

Two stages, as their pipeline (lib/data/gen_data.py, lib/core/train_nrdf.py), without FAISS:
1. The training set, once, saved in --cache: --noisy poses (4M, as their ~10k poses for each of ~400 folders) around random
   training poses, dist ~ N(0, 0.785^2), each with its distance to the training poses (hp.models.nrdf.nearest_distances:
   1000 candidates and the pose it came from).
2. The training: batches of 40,000 noisy and 40,000 clean poses, their L1 loss, Adam, lr 1e-4, weight decay 1e-4, 425k
   steps (the released checkpoint: 4170 epochs x 102 steps), no schedule, no EMA, no validation.

What differs from theirs: the batches are uniform over the poses, theirs take 10,000 poses from each of 4 folders of
sequences (our rotation files have no sequences); the clean poses are all those of the rotation folder.

The run is <run root>/<name>/run_000N (default name nrdf_<data>_s<seed>), with
checkpoints/checkpoint_epoch_best.tar in their format, the file hp.models.nrdf.NRDFPretrained loads:
- a finished run is skipped (--retrain trains a new one);
- a run that stopped early (--time-limit, SIGTERM, SIGUSR1) is resumed from its checkpoint, by the same command.

The exit status is 0 when the run is finished and 75 when it stopped early and can be resumed (as train_model.py).
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

import torch

# the code of this checkout, before an editable install of another one
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hp.data.amass_cfg import ORIGINAL_AMASS_SPLITS, POSEDFLOW_AMASS_SPLITS  # noqa: E402
from hp.models.conditioning import move  # noqa: E402
from hp.models.nrdf import NRDFField, nearest_distances, perturb  # noqa: E402

# ---- parameters ----------------------------------------------------------------------
DATASETS = {  # --data: the rotation folder and the training files
    "posedflow": dict(rotations="~/data/AMASS_rotations_posedflow_c10", files=POSEDFLOW_AMASS_SPLITS["train"]),
    "amass": dict(rotations="~/data/AMASS_rotations_c10", files=ORIGINAL_AMASS_SPLITS["train"]),
}
RUN_ROOT = "~/data/experiments"
CACHE = "~/data/nrdf_data"
SEED = 10
DEVICE = "mps"  # "cpu", "mps" or "cuda:N"
SIGMA = 0.785  # of dist, as configs/data_gen.yaml
NOISY = 4_000_000
CANDIDATES = 1000
QUERY_BATCH = 256  # noisy poses per search
STEPS = 425_000
BATCH = 40_000
LR = 1e-4
WEIGHT_DECAY = 1e-4
LOG_EVERY = 1000
CHECKPOINT_EVERY = 5000
EXIT_STOPPED = 75
# ----------------------------------------------------------------------------------------


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{seconds / 3600:.2f} h"


def load_clean(rotations: str, files) -> torch.Tensor:
    """The training poses [N, 21, 4] (float32, real part first and positive): joints 1 ... 21 of the [x, y, z, w] files."""
    poses = []
    for name in files:
        q = torch.load(Path(rotations).expanduser() / f"{name}.pt", weights_only=True)
        q = torch.as_tensor(q)[:, 1:22][..., [3, 0, 1, 2]].to(torch.float32)
        poses.append(torch.where(q[..., :1] < 0, -q, q))
    return torch.cat(poses)


def training_set(path: Path, clean: torch.Tensor, device: str, args) -> tuple:
    """(noisy poses [N, 21, 4], their distances [N]) on `device`, from the file if it was made with these settings."""
    settings = {"clean": len(clean), "noisy": args.noisy, "sigma": SIGMA, "candidates": CANDIDATES, "seed": args.seed}
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["settings"] == settings:
            print(f"training set: {path}", flush=True)
            return saved["noisy"].to(device), saved["dist"].to(device)
        print(f"{path} has other settings, making it again", flush=True)
    torch.manual_seed(args.seed)
    noisy, dist, start = [], [], time.perf_counter()
    clean_device = clean.to(device)
    for done in range(0, args.noisy, QUERY_BATCH):
        origin = clean[torch.randint(len(clean), (min(QUERY_BATCH, args.noisy - done),))].double()
        q = move(perturb(origin, SIGMA), device, torch.float32)
        d = nearest_distances(q, origin.float().to(device), clean_device, CANDIDATES)
        noisy.append(q.cpu())
        dist.append(d.cpu())
        if done and done % (QUERY_BATCH * 400) == 0:
            spent = time.perf_counter() - start
            print(f"  distances of {done:,} / {args.noisy:,} poses, {minutes(spent)}, left {minutes(spent * (args.noisy / done - 1))}", flush=True)
    noisy, dist = torch.cat(noisy), torch.cat(dist)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"noisy": noisy, "dist": dist, "settings": settings}, temporary)
    temporary.replace(path)
    print(f"training set: {path}, {minutes(time.perf_counter() - start)}; distance mean {dist.mean():.4f}, max {dist.max():.4f}", flush=True)
    return noisy.to(device), dist.to(device)


def runs_of(root: Path):
    return sorted(p for p in root.glob("run_[0-9][0-9][0-9][0-9]") if p.is_dir())


def finished(run: Path) -> bool:
    status = run / "status.json"
    return status.is_file() and json.loads(status.read_text()).get("finished", False)


def save(path: Path, step: int, net, optimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"epoch": step, "model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, temporary)
    temporary.replace(path)


def train(run: Path, noisy, dist, clean, device: str, args) -> bool:
    """The training of `run`, resumed from its checkpoint if there is one; True if it reached the last step."""
    checkpoint = run / "checkpoints" / "checkpoint_epoch_best.tar"
    net = NRDFField().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    step = 0
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        net.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        step = state["epoch"]
        print(f"resuming from step {step:,}", flush=True)
    stop = []
    for name in ("SIGTERM", "SIGUSR1"):
        signal.signal(getattr(signal, name), lambda num, frame: stop.append(signal.Signals(num).name))
    torch.manual_seed(args.seed + step)
    start, first, recent = time.perf_counter(), step, 0.0
    deadline = start + args.time_limit * 60 if args.time_limit else float("inf")
    while step < args.steps and not stop and time.perf_counter() < deadline:
        i = torch.randint(len(noisy), (args.batch,), device=device)
        j = torch.randint(len(clean), (args.batch,), device=device)
        loss = net.loss(noisy[i], dist[i], clean[j])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"the loss is {loss.item()} at step {step}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        recent += loss.item() / LOG_EVERY
        if step % LOG_EVERY == 0:
            spent = time.perf_counter() - start
            print(f"step {step:,} / {args.steps:,}: loss {recent:.5f}, {minutes(spent)}, left {minutes(spent * (args.steps - step) / (step - first))}", flush=True)
            with (run / "history.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "loss": recent, "seconds": spent}) + "\n")
            recent = 0.0
        if step % CHECKPOINT_EVERY == 0:
            save(checkpoint, step, net, optimizer)
    save(checkpoint, step, net, optimizer)
    done = step >= args.steps
    (run / "status.json").write_text(json.dumps({"finished": done, "step": step, "stopped_by": stop or None}))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=sorted(DATASETS), default="posedflow")
    parser.add_argument("--rotations", default=None, help="the rotation folder (default: that of --data)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--cache", default=CACHE, help="where the training set is saved")
    parser.add_argument("--name", default=None, help="the run name (default nrdf_<data>_s<seed>)")
    parser.add_argument("--noisy", type=int, default=NOISY, help="noisy poses of the training set")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--time-limit", type=float, default=None, help="minutes; stop with a checkpoint after this long")
    parser.add_argument("--retrain", action="store_true", help="train a new run even if a finished one exists")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, train nothing")
    args = parser.parse_args()

    rotations = args.rotations or DATASETS[args.data]["rotations"]
    name = args.name or f"nrdf_{args.data}_s{args.seed}"
    root = Path(args.run_root).expanduser() / name
    runs = runs_of(root)
    if runs and finished(runs[-1]) and not args.retrain:
        print(f"{name}: finished already, {runs[-1]} (--retrain trains a new run)", flush=True)
        return 0
    run = runs[-1] if runs and not finished(runs[-1]) and not args.retrain else root / f"run_{len(runs) + 1:04d}"
    plan = (f"{name} on {rotations} ({args.data}), device {args.device}\n  {args.noisy:,} noisy poses, {args.steps:,} steps x batch "
            f"{args.batch:,}, lr {LR}, weight decay {WEIGHT_DECAY}\n  run {run}")
    if args.dry_run:
        print(plan, flush=True)
        return 0
    print(plan, flush=True)
    clean = load_clean(rotations, DATASETS[args.data]["files"])
    print(f"{len(clean):,} training poses", flush=True)
    folder = Path(rotations).expanduser().name
    noisy, dist = training_set(Path(args.cache).expanduser() / f"{folder}_noisy{args.noisy}_s{args.seed}.pt", clean, args.device, args)
    clean = clean.to(args.device)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({**vars(args), "rotations": rotations, "sigma": SIGMA, "lr": LR, "weight_decay": WEIGHT_DECAY}, indent=1))
    start = time.perf_counter()
    done = train(run, noisy, dist, clean, args.device, args)
    print(f"{name}: {'finished' if done else 'stopped early (run the same command to resume)'} after {minutes(time.perf_counter() - start)}, {run}", flush=True)
    return 0 if done else EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(main())
