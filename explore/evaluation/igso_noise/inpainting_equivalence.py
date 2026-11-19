"""Inpainting metrics with the IGSO(3) noise drawn on the device against the CPU (RSGM.noise_on_cpu), over seeds.

Validation poses, K = 10, the four occlusions; RSGM small std (c10); steps 100, resample 2, guidance 0.1, replace,
EDM grid. Writes the per-seed summaries to a JSON file and prints a table with Welch's t-test per metric. From the root
of the repository on the Mac:

    PYTHONPATH=src python explore/evaluation/igso_noise/inpainting_equivalence.py \
        --out docs/experiments/2026-11-16_igso3_noise_on_gpu_inpainting.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import hp.evaluation.completion as ec
from hp.models.io import load_model

METRICS = (("mpvpe_mean", "MPVPE mean (mm)"), ("apd", "APD (cm)"), ("mpvpe_min", "best of 10 (mm)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="~/data/experiments/full/rsgm_posedflow_small_s10/run_0001")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--poses", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = load_model(Path(args.run).expanduser(), device=args.device, weights="weights_ema").eval()
    gt, body = ec.valid_poses("~/data/AMASS_rotations_posedflow_c10", args.poses), ec.Body()
    prior = dict(steps=100, resample=2, guidance=0.1, replace=True, schedule="edm")
    seed_of_batch = torch.manual_seed
    results = {}
    for way in ("cpu", "device"):
        model.noise_on_cpu = way == "cpu"
        results[way] = []
        for seed in range(args.seeds):
            torch.manual_seed = lambda s, seed=seed: seed_of_batch(s + 1000 * seed)  # run() seeds every batch
            start, row = time.perf_counter(), {}
            for occlusion in ec.OCCLUSIONS:
                pred = ec.run(model, gt, occlusion, 10, batch=args.poses, together=True, progress=False,
                              method="inpaint", prior=prior)
                s = ec.summary(ec.metrics(pred, gt, occlusion, body))
                row[occlusion] = {key: s[key] for key, _ in METRICS}
            torch.manual_seed = seed_of_batch
            row["seconds"] = time.perf_counter() - start
            results[way].append(row)
            print(f"{way} seed {seed}: {row['seconds']:.0f} s", flush=True)
    Path(args.out).write_text(json.dumps({"run": args.run, "poses": args.poses, "k": 10, "prior": prior,
                                          "device": args.device, "results": results}, indent=1))
    occlusions = list(ec.OCCLUSIONS)
    print("| metric | occlusion | noise on CPU | noise on device | difference | p (Welch) |")
    print("|---|---|---|---|---|---|")
    for key, name in METRICS:
        for occ in occlusions + ["mean"]:
            def values(way):
                return np.array([np.mean([r[o][key] for o in occlusions]) if occ == "mean" else r[occ][key]
                                 for r in results[way]])
            a, b = values("cpu"), values("device")
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            print(f"| {name} | {occ} | {a.mean():.2f} +- {a.std(ddof=1):.2f} | {b.mean():.2f} +- {b.std(ddof=1):.2f} "
                  f"| {b.mean() - a.mean():+.2f} | {p:.2f} |")


if __name__ == "__main__":
    main()
