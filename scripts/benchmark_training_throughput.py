"""Training throughput of RSGM networks on a GPU: steps and poses per second, memory and GPU utilization, for batch
sizes, the two ways to feed batches (a DataLoader on the CPU, or data_on_device) and TF32 on or off. Only the speed
of training steps on the real training poses is measured; no run is written and nothing is learned from it.

    python scripts/benchmark_training_throughput.py --device cuda:0
    python scripts/benchmark_training_throughput.py --configs abl_norm_aniso dposer_like --batches 5120 40960 \\
        --loaders device --tf32 off on --out ~/data/experiments/throughput.jsonl

The networks are the configs of scripts/train_rsgm_posedflow.py (with joint_scale = 1 for the aniso ones: the same
computation as with the fitted scales). Each row: `warmup` steps, then `steps` timed steps (synchronized). Run it on a
GPU that nothing else uses, or the numbers say little.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the code of this checkout, before an editable install of another one

from hp.data.amass import AMASS  # noqa: E402
from hp.data.amass_cfg import AmassJoints4Learning  # noqa: E402
from hp.models import build_model  # noqa: E402
from hp.training import TrainConfig, Trainer  # noqa: E402
from hp.training.config import load_config  # noqa: E402


def train_script():
    spec = importlib.util.spec_from_file_location("train_rsgm_posedflow", ROOT / "scripts" / "train_rsgm_posedflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu_utilization(device) -> float:
    try:
        return float(torch.cuda.utilization(device))
    except Exception:  # pynvml missing, or not a cuda device
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=["abl_norm_aniso", "dposer_like_aniso"])
    parser.add_argument("--batches", type=int, nargs="+", default=[5120, 10240, 20480, 40960, 81920])
    parser.add_argument("--loaders", nargs="+", choices=["dataloader", "device"], default=["dataloader", "device"])
    parser.add_argument("--tf32", nargs="+", choices=["off", "on"], default=["off"])
    parser.add_argument("--rotations", default="~/data/AMASS_rotations_posedflow_c10")
    parser.add_argument("--split", default="posedflow")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", default=None, help="a .jsonl file the rows are appended to")
    args = parser.parse_args()
    script = train_script()
    unknown = [c for c in args.configs if c not in script.CONFIGS]
    if unknown:
        parser.error(f"unknown configs {unknown}; see scripts/train_rsgm_posedflow.py")

    import os
    os.environ["HUPROSO_AMASS_SPLIT"] = args.split
    train = AMASS(mode="train", path_to_dataset=args.rotations, rotation_representation="matrix",
                  selected_joints=AmassJoints4Learning, load_dtype="float32")
    joints = len(AmassJoints4Learning)
    print(f"{len(train):,} training poses of {args.rotations}, device {args.device}", flush=True)
    cuda = torch.device(args.device).type == "cuda"
    out = Path(args.out).expanduser() if args.out else None
    print(f"{'config':20s} {'batch':>6s} {'loader':10s} {'tf32':4s} {'ms/step':>8s} {'steps/s':>8s} {'kposes/s':>9s} "
          f"{'mem GB':>7s} {'util %':>6s}", flush=True)
    for name in args.configs:
        argv = ["model=rsgm", *script.COMMON, *script.CONFIGS[name]]
        if name in script.ANISO:
            argv.append(f"model.args.joint_scale={[1.0] * joints}")
        cfg = load_config(argv)
        for loader in args.loaders:
            for tf32 in args.tf32:
                for batch in args.batches:
                    torch.manual_seed(0)
                    model = build_model("rsgm", num_joints=joints, **dict(cfg.model.args))
                    config = TrainConfig(batch_size=batch, lr=float(cfg.train.lr), optimizer=cfg.train.optimizer,
                                         weight_decay=cfg.train.weight_decay, scheduler="none",
                                         ema_decay=cfg.train.get("ema_decay"), device=args.device, progress=False,
                                         data_on_device=loader == "device", tf32=tf32 == "on")
                    try:
                        trainer = Trainer(model, train, None, config)
                        if cuda:
                            torch.cuda.reset_peak_memory_stats(args.device)
                        batches = iter(trainer.train_loader)

                        def step():
                            nonlocal batches
                            try:
                                b = next(batches)
                            except StopIteration:
                                batches = iter(trainer.train_loader)
                                b = next(batches)
                            trainer.training_step(b)

                        for _ in range(args.warmup):
                            step()
                        if cuda:
                            torch.cuda.synchronize(args.device)
                        start, util = time.perf_counter(), []
                        for i in range(args.steps):
                            step()
                            if cuda and i % 20 == 10:
                                util.append(gpu_utilization(args.device))
                        if cuda:
                            torch.cuda.synchronize(args.device)
                        seconds = (time.perf_counter() - start) / args.steps
                        memory = torch.cuda.max_memory_allocated(args.device) / 2**30 if cuda else float("nan")
                        row = dict(config=name, batch=batch, loader=loader, tf32=tf32, ms_per_step=1000 * seconds,
                                   steps_per_s=1 / seconds, poses_per_s=batch / seconds, memory_gb=memory,
                                   utilization=sum(util) / len(util) if util else float("nan"), device=args.device,
                                   poses=len(train))
                    except torch.OutOfMemoryError:
                        row = dict(config=name, batch=batch, loader=loader, tf32=tf32, error="out of memory")
                        print(f"{name:20s} {batch:6d} {loader:10s} {tf32:4s} out of memory", flush=True)
                    else:
                        print(f"{name:20s} {batch:6d} {loader:10s} {tf32:4s} {row['ms_per_step']:8.2f} "
                              f"{row['steps_per_s']:8.1f} {row['poses_per_s'] / 1000:9.0f} {memory:7.2f} "
                              f"{row['utilization']:6.0f}", flush=True)
                    finally:
                        trainer = model = None
                        if cuda:
                            torch.cuda.empty_cache()
                    if out:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with out.open("a") as f:
                            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
