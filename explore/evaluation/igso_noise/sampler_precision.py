"""The IGSO(3) angle sampler in float32 on a device against float64 on the CPU: the same uniforms through both
tables (error of the angle), and the Kolmogorov-Smirnov distance of each to the IGSO(3) angle distribution. Uses the
angle table of an RSGM run. From the root of the repository on the Mac:

    PYTHONPATH=src python explore/evaluation/igso_noise/sampler_precision.py --device mps
"""
import argparse
import math
from pathlib import Path

import torch

from hp.models import igso3
from hp.models.io import load_model


def ks(angles: torch.Tensor, s: float) -> float:
    a = angles.double().sort().values
    empirical = torch.arange(1, len(a) + 1, dtype=torch.float64) / len(a)
    every = max(1, len(a) // 2000)
    return float((igso3.angle_cdf(a[::every], s) - empirical[::every]).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="~/data/experiments/full/rsgm_posedflow_small_s10/run_0001")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n", type=int, default=400_000)
    args = parser.parse_args()
    sampler = load_model(Path(args.run).expanduser(), device="cpu").angles
    log_s0, step = sampler.log_s[0].item(), (sampler.log_s[1] - sampler.log_s[0]).item()
    print(f"table: s in [{sampler.s_min:.2e}, {sampler.s_max:.2f}], {args.n:,} draws per s, device {args.device}")
    print("| s | median angle (deg) | max abs. error (deg) | max rel. error | mean rel. error | KS float64 (cpu) "
          "| KS float32 (device) |")
    print("|---|---|---|---|---|---|---|")
    for s in [sampler.s_min * 1.01, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, sampler.s_max * 0.99]:
        position = (math.log(s) - log_s0) / step
        lo = min(max(math.floor(position), 0), len(sampler.log_s) - 2)
        a = min(max(position - lo, 0.0), 1.0)

        def draw(table, u):  # the interpolation of AngleSampler._draw with given uniforms
            u = u * (len(sampler.u) - 1)
            j = u.floor().long().clamp(0, len(sampler.u) - 2)
            b = u - j
            def at(level):
                return table[level, j] * (1 - b) + table[level, j + 1] * b
            return at(lo) * (1 - a) + at(lo + 1) * a

        torch.manual_seed(0)
        u = torch.rand(args.n, dtype=torch.float64)
        ref = draw(sampler.table, u)
        f32 = draw(sampler.table_on(args.device, torch.float32), u.float().to(args.device)).cpu().double()
        rel = (f32 - ref).abs() / ref.clamp_min(1e-30)
        torch.manual_seed(1)
        ks64 = ks(sampler.sample_at(s, (args.n,), "cpu", torch.float64), s)
        torch.manual_seed(1)
        ks32 = ks(sampler.sample_at(s, (args.n,), args.device, torch.float32).cpu(), s)
        print(f"| {s:.2e} | {math.degrees(ref.median()):.4f} | {math.degrees((f32 - ref).abs().max()):.1e} "
              f"| {rel.max():.1e} | {rel.mean():.1e} | {ks64:.4f} | {ks32:.4f} |")
    print(f"KS 1% critical value for n = {args.n:,}: {1.63 / math.sqrt(args.n):.4f}")


if __name__ == "__main__":
    main()
