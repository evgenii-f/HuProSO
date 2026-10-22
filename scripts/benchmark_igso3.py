"""Compare the float32 device heat kernel (hp/models/igso3_gpu.py) with the float64 CPU one (hp/models/igso3.py):
accuracy and speed, e.g.

    python scripts/benchmark_igso3.py                       # mps or cuda if there is one, and cpu
    python scripts/benchmark_igso3.py --batch-sizes 512 4096 --json results.json

Accuracy
  - on a grid of (s, omega) against a 50-digit mpmath reference (needs mpmath), evaluated at the inputs as each dtype
    rounds them. "current formula in fp32" is igso3.py's formula run in float32: it shows why igso3.py needs float64.
  - on the training distribution of RSGM (t from loss_times, omega from the sampler), against the new kernel in
    float64 (which matches mpmath to ~1e-11 on the grid). Errors of d log f are std(s) * error: the scale of the
    training target, which is O(1).
  - the expected training loss of one fixed network with the noise of either kernel (they must agree within the
    Monte Carlo error).
Speed (median ms; the device is synchronized around every call)
  - d log f alone, the angle sampler alone,
  - the noising of RSGM.loss: from poses on the device to x_t, target, t on the device,
  - a whole training step (loss, backward, Adam) of RSGM with noise="cpu" and noise="device".
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported

import argparse
import json
import math
import time

import roma
import torch

from hp.models import igso3, igso3_gpu
from hp.models.rsgm import RSGM, exp_frame, move

J = 19


# ---- the kernels under comparison ----

def current_formula_fp32_log(omega, s):
    """igso3.log_density, in the dtype of the input."""
    k = igso3.IMAGES.to(omega)
    shifted = omega[..., None] + 2 * math.pi * k
    weight = (-1) ** k.abs() * torch.exp(-(shifted**2 - omega[..., None] ** 2) / (2 * s[..., None]))
    bracket = (weight * shifted).sum(-1)
    return s / 8 + 0.5 * torch.log(2 * math.pi / s) - torch.log(s) - omega**2 / (2 * s) + torch.log(bracket) - torch.log(torch.sin(omega / 2))


def current_formula_fp32_dlog(omega, s):
    """igso3.d_log_density, in the dtype of the input."""
    k = igso3.IMAGES.to(omega)
    shifted = omega[..., None] + 2 * math.pi * k
    weight = (-1) ** k.abs() * torch.exp(-(shifted**2 - omega[..., None] ** 2) / (2 * s[..., None]))
    bracket = (weight * shifted).sum(-1)
    d_bracket = (weight * (1 - shifted * 2 * math.pi * k / s[..., None])).sum(-1)
    return -omega / s + d_bracket / bracket - 0.5 / torch.tan(omega / 2)


# ---- helpers ----

def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device: str, repeats: int) -> float:
    """Median milliseconds of fn() over `repeats` runs, after two warm-up runs."""
    fn(), fn()
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return 1000 * sorted(times)[len(times) // 2]


def std_of(s):
    return torch.sqrt(-torch.expm1(-s.double()))


# ---- accuracy ----

def mp_reference(omega: float, s: float, mp):
    with mp.workdps(50):
        w, s = mp.mpf(omega), mp.mpf(s)

        def log_f(x):
            br = mp.fsum((-1) ** abs(k) * (x + 2 * mp.pi * k) * mp.exp(-((x + 2 * mp.pi * k) ** 2 - x**2) / (2 * s)) for k in range(-12, 13))
            return s / 8 + mp.log(mp.sqrt(2 * mp.pi / s)) - mp.log(s) - x**2 / (2 * s) + mp.log(br) - mp.log(mp.sin(x / 2))

        return float(log_f(w)), float(mp.diff(log_f, w))


def grid_accuracy(devices):
    import mpmath as mp

    s_values = torch.logspace(math.log10(4e-7), math.log10(12.0), 14, dtype=torch.float64).tolist() + [0.99, 1.0]
    omega_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, math.pi - 1e-3]
    variants = [("current igso3, cpu fp64", "cpu", torch.float64, igso3.log_density, igso3.d_log_density),
                ("current formula, cpu fp32", "cpu", torch.float32, current_formula_fp32_log, current_formula_fp32_dlog),
                ("new, cpu fp64", "cpu", torch.float64, igso3_gpu.log_density, igso3_gpu.d_log_density),
                ("new, cpu fp32", "cpu", torch.float32, igso3_gpu.log_density, igso3_gpu.d_log_density)]
    variants += [(f"new, {d} fp32", d, torch.float32, igso3_gpu.log_density, igso3_gpu.d_log_density) for d in devices if d != "cpu"]
    refs = {}
    rows = {}
    for name, device, dtype, log_fn, dlog_fn in variants:
        omega = torch.tensor(omega_values, dtype=dtype).expand(len(s_values), -1)
        s = torch.tensor(s_values, dtype=dtype)[:, None].expand_as(omega)
        if dtype not in refs:
            refs[dtype] = torch.tensor([[mp_reference(float(w), float(t), mp) for w, t in zip(wr, sr)] for wr, sr in zip(omega, s)], dtype=torch.float64)
        ref = refs[dtype]
        log_f = log_fn(omega.to(device), s.to(device)).cpu().double()
        d_log_f = dlog_fn(omega.to(device), s.to(device)).cpu().double()
        scaled = (d_log_f - ref[..., 1]).abs() * std_of(s)
        typical = omega.double() <= 6 * s.double().sqrt()
        rows[name] = {
            "dlog_scaled_err_typical": float(scaled[typical].nan_to_num(math.inf).max()),
            "dlog_scaled_err_all": float(scaled.nan_to_num(math.inf).max()),
            "dlog_rel_err_all": float(((d_log_f - ref[..., 1]).abs() / ref[..., 1].abs().clamp_min(1e-3)).nan_to_num(math.inf).max()),
            "logf_rel_err_all": float(((log_f - ref[..., 0]).abs() / ref[..., 0].abs().clamp_min(1)).nan_to_num(math.inf).max()),
            "nonfinite": int((~torch.isfinite(d_log_f)).sum() + (~torch.isfinite(log_f)).sum()),
        }
    return rows


def distribution_accuracy(devices, n: int):
    model = RSGM(num_joints=1, beta_f=10.0, noise="cpu")
    gen = torch.Generator().manual_seed(0)
    t = model.loss_times(torch.rand(n, dtype=torch.float64, generator=gen))
    s = model.s(t)
    omega = model.angles.sample(s, gen)
    ref = igso3_gpu.d_log_density(omega, s)
    scale = std_of(s)
    variants = [("current igso3, cpu fp64", "cpu", torch.float64, igso3.d_log_density),
                ("current formula, cpu fp32", "cpu", torch.float32, current_formula_fp32_dlog),
                ("new, cpu fp32", "cpu", torch.float32, igso3_gpu.d_log_density)]
    variants += [(f"new, {d} fp32", d, torch.float32, igso3_gpu.d_log_density) for d in devices if d != "cpu"]
    rows = {}
    for name, device, dtype, fn in variants:
        # the reference at the rounded inputs: the new kernel in float64 at omega, s as the dtype rounds them
        w, ss = omega.to(dtype), s.to(dtype)
        r = igso3_gpu.d_log_density(w.double(), ss.double())
        err = (fn(w.to(device), ss.to(device)).cpu().double() - r).abs() * scale
        rows[name] = {"max": float(err.nan_to_num(math.inf).max()), "rms": float(err.pow(2).mean().sqrt()),
                      "share_above_1e-4": float((err > 1e-4).double().mean())}
    rows["(scale: |std * d log f|)"] = {"max": float((ref * scale).abs().max()), "rms": float((ref * scale).pow(2).mean().sqrt()), "share_above_1e-4": float("nan")}
    return rows


def ks_distance(angles: torch.Tensor, s: float) -> float:
    """Kolmogorov-Smirnov distance of a sample of angles to the IGSO(3) angle distribution (as tests/test_igso3.py)."""
    angles = angles.sort().values
    empirical = torch.arange(1, len(angles) + 1, dtype=torch.float64) / len(angles)
    every = max(1, len(angles) // 500)
    return float((igso3.angle_cdf(angles[::every], s) - empirical[::every]).abs().max())


def sampler_accuracy(devices):
    old, new = igso3.AngleSampler(1e-7, 6.5), igso3_gpu.AngleSampler(1e-7, 6.5)
    n, rows = 20_000, {}
    for s in (1e-6, 1e-3, 0.1, 1.0, 5.0):
        row = {"current, cpu fp64": ks_distance(old.sample(torch.full((n,), s), torch.Generator().manual_seed(0)), s)}
        for d in devices:
            sampler = new.to(d)
            angles = sampler.sample(torch.full((n,), s, device=d), torch.Generator(device=d).manual_seed(0))
            row[f"new, {d} fp32"] = ks_distance(angles.cpu().double(), s)
        rows[s] = row
    return rows, 1.63 / math.sqrt(n)


def expected_losses(devices, batches: int, batch: int):
    """Mean training loss of one fixed network under the noise of either kernel, on fixed poses."""
    out = {}
    poses = roma.random_rotmat(batch * J, generator=torch.Generator().manual_seed(1)).reshape(batch, J, 3, 3)
    for d in devices:
        torch.manual_seed(0)
        current = RSGM(num_joints=J, noise="cpu").to(d).train()
        torch.manual_seed(0)
        new = RSGM(num_joints=J, noise="device").to(d).train()
        x = poses.to(d)
        with torch.no_grad():
            for name, model in (("current", current), ("new", new)):
                losses = torch.stack([model.loss(x).cpu() for _ in range(batches)]).double()
                out[f"{name}, {d}"] = {"mean": float(losses.mean()), "sem": float(losses.std() / math.sqrt(batches))}
    return out


# ---- speed ----

def speed(devices, batch_sizes, repeats):
    out = {}
    old_sampler = igso3.AngleSampler(4e-7, 5.1)
    for batch in batch_sizes:
        n = batch * J
        gen = torch.Generator().manual_seed(0)
        s64 = torch.exp(torch.rand(n, dtype=torch.float64, generator=gen) * 16 - 14.7)
        w64 = old_sampler.sample(s64, gen)
        row = {"d_log_density, current (cpu fp64)": timed(lambda: igso3.d_log_density(w64, s64), "cpu", repeats),
               "angle sampler, current (cpu fp64)": timed(lambda: old_sampler.sample(s64), "cpu", repeats)}
        for d in devices:
            w, s = w64.float().to(d), s64.float().to(d)
            row[f"d_log_density, new ({d} fp32)"] = timed(lambda: igso3_gpu.d_log_density(w, s), d, repeats)
            new_sampler = igso3_gpu.AngleSampler(4e-7, 5.1).to(d)
            row[f"angle sampler, new ({d} fp32)"] = timed(lambda: new_sampler.sample(s), d, repeats)

            torch.manual_seed(0)
            current = RSGM(num_joints=J, noise="cpu").to(d).train()
            torch.manual_seed(0)
            new = RSGM(num_joints=J, noise="device").to(d).train()
            pose = roma.random_rotmat(n, generator=torch.Generator().manual_seed(2)).reshape(batch, J, 3, 3).to(d)

            def noise_current():  # the first half of RSGM.loss, up to the inputs of the network on the device
                t = current.loss_times(torch.rand(batch, dtype=torch.float64))
                s_ = current.s(t)[:, None].expand(batch, J)
                omega = current.angles.sample(s_)
                axis = igso3.random_axes((batch, J))
                x0 = move(pose.reshape(batch, J, 3, 3), "cpu", torch.float64)
                xt = exp_frame(x0, omega[..., None] * axis)
                target = current.std(t)[:, None, None] * igso3.d_log_density(omega, s_)[..., None] * axis
                return [move(v, d, torch.float32) for v in (xt, target, t)]

            def noise_new():
                t = new.loss_times(torch.rand(batch, device=d))
                s_ = new.s(t)[:, None].expand(batch, J)
                omega = new.device_angles.sample(s_)
                axis = igso3_gpu.random_axes((batch, J), device=d)
                xt = exp_frame(pose, omega[..., None] * axis)
                return xt, new.std(t)[:, None, None] * igso3_gpu.d_log_density(omega, s_)[..., None] * axis, t

            row[f"noising of the loss, current ({d})"] = timed(noise_current, d, repeats)
            row[f"noising of the loss, new ({d})"] = timed(noise_new, d, repeats)
            for name, model in (("current", current), ("new", new)):
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

                def step():
                    optimizer.zero_grad()
                    model.loss(pose).backward()
                    optimizer.step()

                row[f"train step, {name} ({d})"] = timed(step, d, repeats)
        out[batch] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    accelerators = (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    parser.add_argument("--devices", nargs="+", default=accelerators + ["cpu"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[512, 4096, 16384], help="poses; x 19 joints")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--distribution-samples", type=int, default=1_000_000)
    parser.add_argument("--loss-batches", type=int, default=200, help="batches of 512 per expected loss")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    print(f"torch {torch.__version__}, devices {args.devices}, {J} joints")
    report = {"args": vars(args), "torch": torch.__version__}

    print("\n== accuracy on a grid of (s, omega) against mpmath (50 digits) ==")
    rows = report["grid"] = grid_accuracy(args.devices)
    print(f"{'kernel':28s} {'std*|dlog err| typical':>23s} {'std*|dlog err| all':>19s} {'rel dlog err':>13s} {'rel log f err':>14s} {'non-finite':>11s}")
    for name, r in rows.items():
        print(f"{name:28s} {r['dlog_scaled_err_typical']:23.2e} {r['dlog_scaled_err_all']:19.2e} {r['dlog_rel_err_all']:13.2e} {r['logf_rel_err_all']:14.2e} {r['nonfinite']:11d}")
    print("(typical: omega <= 6 sqrt(s); all: the whole grid, up to omega = pi for s = 4e-7)")

    print(f"\n== accuracy on the training distribution ({args.distribution_samples:,} draws of t and omega, beta_f = 10) ==")
    rows = report["distribution"] = distribution_accuracy(args.devices, args.distribution_samples)
    print(f"{'kernel':28s} {'max std*|err|':>14s} {'rms':>10s} {'share > 1e-4':>13s}")
    for name, r in rows.items():
        print(f"{name:28s} {r['max']:14.2e} {r['rms']:10.2e} {r['share_above_1e-4']:13.2e}")

    print("\n== the angle samplers: Kolmogorov-Smirnov distance to IGSO(3), 20,000 draws ==")
    rows, level = sampler_accuracy(args.devices)
    report["sampler_ks"] = {str(k): v for k, v in rows.items()}
    names = list(next(iter(rows.values())))
    print(f"{'s':>8s} " + " ".join(f"{n:>20s}" for n in names))
    for s, r in rows.items():
        print(f"{s:8.0e} " + " ".join(f"{r[n]:20.4f}" for n in names))
    print(f"(the 1% level of the test is {level:.4f})")

    print(f"\n== expected training loss of one fixed network, {args.loss_batches} noise draws on 512 fixed poses ==")
    rows = report["expected_loss"] = expected_losses(args.devices, args.loss_batches, 512)
    for name, r in rows.items():
        print(f"{name:20s} {r['mean']:.4f} +- {r['sem']:.4f}")

    print("\n== speed (median ms) ==")
    rows = report["speed"] = speed(args.devices, args.batch_sizes, args.repeats)
    for batch, r in rows.items():
        print(f"-- batch {batch} poses ({batch * J:,} rotations)")
        for name, ms in r.items():
            print(f"  {name:45s} {ms:9.2f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwritten to {args.json}")


if __name__ == "__main__":
    main()
