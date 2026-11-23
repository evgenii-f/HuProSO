"""The validation loss of RSGM runs in bins of the time t of the noise, paired (the same 8192 validation poses and
noise draws for every run, from the fixed seed of the loss in eval mode); section 19 of
docs/experiments/2026-11-14_score_network_architectures.md. Short (minutes on the Mac):

    PYTHONPATH=src python explore/evaluation/loss_by_noise_level.py
"""
from pathlib import Path

import torch
from hp.evaluation.data import load_reference
from hp.models.io import load_model

E = str(Path("~/data/experiments/full").expanduser()) + "/"
RUNS = {"local": "rsgm_posedflow_wide_RAD_x_s10", "local+pairs": "rsgm_posedflow_wide_RAD_x_pairs_l1_s10",
        "mace_g_x": "rsgm_posedflow_mace_g_x_s10", "mace_x": "rsgm_posedflow_mace_x_s10"}
BINS = [(0.002, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]
N = 8192
valid = load_reference("valid", "~/data/AMASS_rotations_posedflow_c10", "posedflow")
poses = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(0))[:N]].float()
rows = {}
for name, run in RUNS.items():
    model = load_model(E + run + "/run_0001", device="mps", weights="weights_ema").eval()
    out = []
    for lo, hi in BINS:
        model.loss_times = lambda u, lo=lo, hi=hi: lo + (hi - lo) * u
        with torch.no_grad():
            vals = [float(model.loss(poses[i:i + 2048].reshape(-1, 19 * 9).to("mps"))) for i in range(0, N, 2048)]
        out.append(sum(vals) / len(vals))
    rows[name] = out
    print(name, " ".join(f"{v:7.3f}" for v in out), flush=True)
print("bins", BINS)
