"""Generation of two cells of the factorial (section 13 of docs/experiments/2026-11-14_score_network_architectures.md)
with other samplers: is the worse generation of per-joint noise the sampler? Validation only; from the root:

    PYTHONPATH=src python explore/evaluation/factorial_sampler_check.py [--cells fact_N fact_NA] [--device mps]

10 sets of 500 samples per sampler, FID against the joint statistics of the validation poses (FID_val) and APD.
"""
import argparse
import time

import numpy as np
import torch

from hp.evaluation.data import load_reference
from hp.evaluation.posedflow_protocol import SmplSkeleton, apd, confidence, fid
from hp.models.io import load_model

RUN = "~/data/experiments/factorial/rsgm_posedflow_{}_s10/run_0001"
VARIANTS = [("euler_maruyama", "uniform", 100), ("euler_maruyama", "uniform", 1000),
            ("euler_maruyama", "edm", 200), ("heun", "edm", 50)]
SETS, SIZE = 10, 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", nargs="+", default=["fact_N", "fact_NA"])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    valid = load_reference("valid", "~/data/AMASS_rotations_posedflow_c10", "posedflow")
    skeleton = SmplSkeleton("~/data/body_models/smpl/smpl_neutral_skeleton.npz")
    joints = skeleton.joints(valid).reshape(len(valid), -1).numpy().astype(np.float64)
    mu, cov = joints.mean(0), np.cov(joints, rowvar=False)
    for cell in args.cells:
        model = load_model(RUN.format(cell), device=args.device, weights="weights_ema").eval()
        for sampler, schedule, steps in VARIANTS:
            model.sampler, model.schedule, model.sample_steps = sampler, schedule, steps
            torch.manual_seed(0)
            start, fids, apds = time.time(), [], []
            for _ in range(SETS):
                j = skeleton.joints(model.sample(SIZE).float().cpu())
                fids.append(fid(j, mu, cov))
                apds.append(apd(j))
            f, a = confidence(fids), confidence(apds)
            print(f"{cell:8s} {sampler:15s} {schedule:8s} {steps:5d} (NFE {model.sample_nfe:5d}): FID_val "
                  f"{f['mean']:.4f} +- {f['ci95']:.4f}, APD {a['mean']:.2f} +- {a['ci95']:.2f}  "
                  f"({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
