"""Benchmark the normalizing flow of HuProSO3 (hp/models/so3_flow.py) on one or more devices, and check that they agree, e.g.

    python scripts/benchmark_norm_flow.py                          # cpu only
    python scripts/benchmark_norm_flow.py --devices mps cpu        # several: each is compared with cpu (speed-up, results)
    python scripts/benchmark_norm_flow.py --devices mps cpu --batch-sizes 500 2000 --json results.json

Per device and batch size it times a training step (loss, backward, gradient clipping, Adam, as in the training),
sampling and the log-likelihood, with the same weights and poses on every device. It also checks
  - that the training losses stay finite,
  - that the devices give the same log-likelihood, loss and samples on the same input,
  - that the samples are rotations (orthogonal, determinant 1),
  - which operations fall back from mps to the CPU (PyTorch warns about them),
and projects how long a training run takes on each device.
The weights are those of the paper's prior (--pretrained), the poses are uniform random rotations: the timings do not
depend on the data, and no dataset is needed. Run it when nothing else is using the machine.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported

import argparse
import json
import time
import warnings
from pathlib import Path

import roma
import torch

from hp.models import build_model
from hp.utils.paths import REPO_ROOT

PAPER_BATCH, PAPER_ITERATIONS, PAPER_LR, PAPER_CLIP = 2000, 20_000, 5e-3, 1.0  # config/train.yaml
DEFAULT_PRETRAINED = REPO_ROOT / "experiments" / "pretrained_models" / "id_0900"


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device: str, repeats: int) -> float:
    """Median seconds of fn() over `repeats` runs, after one warm-up run."""
    fn()
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def rotations(n: int, joints: int, seed: int = 0) -> torch.Tensor:
    return roma.random_rotmat(n * joints, generator=torch.Generator().manual_seed(seed)).reshape(n, joints, 3, 3)


def make_model(args, device: str):
    """The same weights on every device: the paper's prior, or (--pretrained none) a fresh flow with a fixed seed."""
    torch.manual_seed(args.seed)
    model = build_model("so3_flow", num_joints=args.joints, flow={"layers": args.layers})
    if args.pretrained:
        model.load_legacy_checkpoint(args.pretrained)
    return model.to(device)


def rotation_error(samples: torch.Tensor) -> float:
    """Largest deviation of the samples from SO(3): |R R^T - I| and |det R - 1| (in float64)."""
    r = samples.double().reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=torch.float64)
    return float(max((r @ r.transpose(1, 2) - eye).abs().max(), (torch.linalg.det(r) - 1).abs().max()))


def benchmark_device(args, device: str) -> dict:
    model = make_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PAPER_LR)
    out = {"timings": {}, "nonfinite_losses": 0, "losses_checked": 0}

    for batch in args.batch_sizes:
        x = rotations(batch, args.joints).to(device)
        row = {}

        def train_step():
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PAPER_CLIP)
            optimizer.step()

        model.train()
        row["train_step"] = timed(train_step, device, args.repeats)
        model.eval()
        row["sampling"] = timed(lambda: model.sample(batch), device, args.repeats)
        with torch.no_grad():
            row["log_likelihood"] = timed(lambda: model.log_prob(x), device, args.repeats)
        row["log_likelihood_per_pose"] = row["log_likelihood"] / batch
        out["timings"][batch] = row

    # finite training losses (no optimizer step, so that the weights stay as they are)
    model.train()
    x = rotations(args.batch_sizes[-1], args.joints).to(device)
    with torch.no_grad():
        for _ in range(args.finite_checks):
            out["nonfinite_losses"] += int(not torch.isfinite(model.loss(x)).item())
            out["losses_checked"] += 1
    model.eval()
    out["sample_rotation_error"] = rotation_error(model.sample(args.compare_poses).cpu())
    return out


def reference_values(args, device: str) -> dict:
    """Loss, log-likelihood and samples of the weights on fixed poses and fixed random numbers, for the comparison."""
    model = make_model(args, device).eval()
    x = rotations(args.compare_poses, args.joints, seed=1).to(device)
    with torch.no_grad():
        torch.manual_seed(args.seed)  # the same random rotations are pushed through the flow on every device
        return {
            "valid_loss": model.loss(x).cpu().reshape(1),
            "log_likelihood": model.log_prob(x).cpu(),
            "samples": model.sample(args.compare_poses).cpu(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", nargs="+", default=["cpu"], help="e.g. --devices mps cpu")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[500, PAPER_BATCH], help="2000: batch of config/train.yaml")
    parser.add_argument("--joints", type=int, default=19)
    parser.add_argument("--layers", type=int, default=12, help="flow layers (12: the paper)")
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED), help="legacy checkpoint folder; 'none': fresh weights")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--finite-checks", type=int, default=20, help="training losses checked for NaN per device")
    parser.add_argument("--compare-poses", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="write the results to this file")
    args = parser.parse_args()
    args.pretrained = None if args.pretrained.lower() == "none" else Path(args.pretrained).expanduser()
    if args.pretrained and not args.pretrained.is_dir():
        raise SystemExit(f"No pretrained weights at {args.pretrained} (use --pretrained none for fresh weights).")

    devices = [d for d in args.devices if d != "mps" or torch.backends.mps.is_available()]
    if devices != args.devices:
        print(f"mps is not available; running on {devices}")
    print(f"torch {torch.__version__}, {args.joints} joints, {args.layers} layers, weights: {args.pretrained or 'fresh'}, "
          f"CPU threads: {torch.get_num_threads()}")

    results, references, fallbacks = {}, {}, {}
    for device in devices:
        print(f"--- {device}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results[device] = benchmark_device(args, device)
            references[device] = reference_values(args, device)
        fallbacks[device] = sorted({str(w.message).split("\n")[0] for w in caught if "fall back" in str(w.message).lower()})
        r = results[device]
        print(f"non-finite training losses: {r['nonfinite_losses']} / {r['losses_checked']}; "
              f"samples off SO(3) by at most {r['sample_rotation_error']:.1e}")
        for message in fallbacks[device]:
            print(f"fallback to the CPU: {message}")

    # every other device is compared with the base: cpu if it is there, else the first device
    base = "cpu" if "cpu" in devices else devices[0]
    others = [d for d in devices if d != base]

    print()
    header = f"{'batch':>6s} {'operation':24s} " + " ".join(f"{d:>10s}" for d in devices)
    print(header + "".join(f" {d + ' speed-up':>14s}" for d in others))
    for batch in args.batch_sizes:
        for name in ("train_step", "sampling", "log_likelihood_per_pose"):
            times = {d: results[d]["timings"][batch][name] for d in devices}
            line = f"{batch:6d} {name:24s} " + " ".join(f"{1000 * times[d]:8.2f}ms" for d in devices)
            line += "".join(f" {times[base] / times[d]:13.2f}x" for d in others)
            print(line)
    if others:
        print(f"(speed-up: time on {base} / time on the device)")

    if PAPER_BATCH in args.batch_sizes:
        print(f"\nprojected training run ({PAPER_ITERATIONS:,} iterations at batch {PAPER_BATCH}, the training step only):")
        for d in devices:
            hours = results[d]["timings"][PAPER_BATCH]["train_step"] * PAPER_ITERATIONS / 3600
            print(f"  {d:>6s}: {hours:6.1f} h")

    comparison = {}
    for other in others:
        print(f"\n{other} vs {base} on the same weights and input (max abs difference):")
        comparison[other] = {}
        for key in references[base]:
            a, b = references[other][key], references[base][key]
            entry = {"max_abs_diff": float((a - b).abs().max()), "max_abs_value": float(b.abs().max())}
            comparison[other][key] = entry
            print(f"  {key:15s} {entry['max_abs_diff']:.2e}  (values up to {entry['max_abs_value']:.2e})")

    if args.json:
        report = {"args": {k: str(v) for k, v in vars(args).items()}, "torch": torch.__version__, "results": results,
                  "fallbacks": fallbacks, "comparison": comparison}
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwritten to {args.json}")

    bad = [d for d in devices if results[d]["nonfinite_losses"]]
    if bad:
        raise SystemExit(f"Non-finite training losses on {bad}.")


if __name__ == "__main__":
    main()
