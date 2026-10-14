"""Benchmark RSGM (hp/models/rsgm.py) on one or more devices, and check that they agree, e.g.

    python scripts/benchmark_rsgm.py                          # cpu only
    python scripts/benchmark_rsgm.py --devices mps cpu        # several: each is compared with cpu (speed-up, results)
    python scripts/benchmark_rsgm.py --devices mps --batch-sizes 512 2000 --json results.json

Per device and batch size it times a training step (loss, backward, Adam), sampling and the log-likelihood, with
the same weights and poses on every device. It also checks
  - that the training losses stay finite (on mps a copy that changes device and dtype in one .to() used to
    return garbage now and then, which made losses NaN),
  - that mps and cpu give the same validation loss, log-likelihood and score on the same input,
  - which operations fall back from mps to the CPU (PyTorch warns about them).
The poses are uniform random rotations: the timings do not depend on the data, and no dataset is needed.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported

import argparse
import json
import time
import warnings

import roma
import torch

from hp.models import build_model


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
    """The same weights on every device."""
    torch.manual_seed(args.seed)
    model = build_model("rsgm", num_joints=args.joints, sample_steps=args.sample_steps, likelihood_steps=args.likelihood_steps)
    return model.to(device)


def benchmark_device(args, device: str) -> dict:
    model = make_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    out = {"timings": {}, "nonfinite_losses": 0, "losses_checked": 0}

    for batch in args.batch_sizes:
        x = rotations(batch, args.joints).to(device)
        row = {}

        def train_step():
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

        model.train()
        row["train_step"] = timed(train_step, device, args.repeats)
        model.eval()
        row["sampling"] = timed(lambda: model.sample(batch), device, args.repeats)
        x_ll = x[: args.likelihood_batch]
        with torch.no_grad():
            row["log_likelihood"] = timed(lambda: model.log_prob(x_ll), device, max(1, args.repeats // 3))
        row["log_likelihood_per_pose"] = row["log_likelihood"] / len(x_ll)
        out["timings"][batch] = row

    # finite training losses (no optimizer step, so that the weights stay as they are)
    model.train()
    x = rotations(args.batch_sizes[-1], args.joints).to(device)
    with torch.no_grad():
        for _ in range(args.finite_checks):
            out["nonfinite_losses"] += int(not torch.isfinite(model.loss(x)).item())
            out["losses_checked"] += 1
    return out


def reference_values(args, device: str) -> dict:
    """Validation loss, log-likelihood and score of fresh weights on fixed poses, for the comparison of devices."""
    model = make_model(args, device).eval()
    x = rotations(args.compare_poses, args.joints, seed=1).to(device)
    t = torch.linspace(0.01, 1.0, args.compare_poses).to(device)
    with torch.no_grad():
        return {
            "valid_loss": model.loss(x).cpu(),
            "log_likelihood": model.log_prob(x).cpu(),
            "score": model.score(x, t).cpu(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", nargs="+", default=["cpu"], help="e.g. --devices mps cpu")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[512, 2000], help="512: batch of config/models/rsgm.yaml")
    parser.add_argument("--joints", type=int, default=19)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--likelihood-steps", type=int, default=100)
    parser.add_argument("--likelihood-batch", type=int, default=200, help="poses in a timed log-likelihood call")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--finite-checks", type=int, default=100, help="training losses checked for NaN per device")
    parser.add_argument("--compare-poses", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="write the results to this file")
    args = parser.parse_args()

    devices = [d for d in args.devices if d != "mps" or torch.backends.mps.is_available()]
    if devices != args.devices:
        print(f"mps is not available; running on {devices}")
    print(f"torch {torch.__version__}, {args.joints} joints, {args.sample_steps} sampling and {args.likelihood_steps} likelihood steps")

    results, references, fallbacks = {}, {}, {}
    for device in devices:
        print(f"--- {device}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results[device] = benchmark_device(args, device)
            references[device] = reference_values(args, device)
        fallbacks[device] = sorted({str(w.message).split("\n")[0] for w in caught if "fall back" in str(w.message).lower()})
        r = results[device]
        print(f"non-finite training losses: {r['nonfinite_losses']} / {r['losses_checked']}")
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

    comparison = {}
    for other in others:
        print(f"\n{other} vs {base} on the same weights and poses (max abs difference):")
        comparison[other] = {}
        for key in references[base]:
            a, b = references[other][key], references[base][key]
            entry = {"max_abs_diff": float((a - b).abs().max()), "max_abs_value": float(b.abs().max())}
            comparison[other][key] = entry
            print(f"  {key:15s} {entry['max_abs_diff']:.2e}  (values up to {entry['max_abs_value']:.2e})")

    if args.json:
        report = {"args": vars(args), "torch": torch.__version__, "results": results, "fallbacks": fallbacks, "comparison": comparison}
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwritten to {args.json}")

    bad = [d for d in devices if results[d]["nonfinite_losses"]]
    if bad:
        raise SystemExit(f"Non-finite training losses on {bad}.")


if __name__ == "__main__":
    main()
