"""The denoising loss of RSGM runs on training and on validation poses in bins of the time t of the noise: is the
input or the generalization the limit (section 21 of docs/experiments/2026-11-14_score_network_architectures.md)?

8192 random poses of each split of c10 (the same for every run), eval mode with the fixed noise of the loss, the median
over 16 chunks of 512 poses (robust to a rare pose). "zero score": the loss of a network that predicts nothing.
The runs were trained on the full data. Two minutes on the CPU, from the root of the repository:

    HP_DEVICE=cpu PYTHONPATH=src python explore/evaluation/train_valid_gap.py
"""
from pathlib import Path

import torch

from hp.evaluation.data import load_reference
from hp.models.io import load_model

E = Path("~/data/experiments/full").expanduser()
RUNS = {"local s10": "wide_RAD_x_s10", "local s11": "wide_RAD_x_s11", "local+pairs": "wide_RAD_x_pairs_l1_s10",
        "local+wigner": "wide_RAD_x_wigner_s10", "wigner+pairs": "wide_RAD_x_wigner_pairs_s10",
        "mace_g_x": "mace_g_x_s10"}
BINS = [(0.002, 0.004), (0.004, 0.007), (0.007, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 0.1), (0.1, 0.2), (0.2, 0.5)]
N, CHUNK = 8192, 512


def losses(model, poses):
    out = []
    for lo, hi in BINS:
        model.loss_times = lambda u, lo=lo, hi=hi: lo + (hi - lo) * u
        with torch.no_grad():
            out.append([float(model.loss(poses[i:i + CHUNK].reshape(-1, 171))) for i in range(0, N, CHUNK)])
    return torch.tensor(out).median(1).values


def line(name, values):
    print(f"{name:18s} " + " ".join(f"{v:7.3f}" for v in values), flush=True)


poses = {}
for split in ("train", "valid"):
    ref = load_reference(split, "~/data/AMASS_rotations_posedflow_c10", "posedflow")
    poses[split] = ref[torch.randperm(len(ref), generator=torch.Generator().manual_seed(0))[:N]].float().clone()
    del ref
print("bins", BINS)
for k, (name, run) in enumerate(RUNS.items()):
    model = load_model(str(E / f"rsgm_posedflow_{run}" / "run_0001"), device="cpu", weights="weights_ema").eval()
    if k == 0:
        score, model.score = model.score, lambda x, t, f=None: torch.zeros(x.shape[0], 19, 3)
        line("zero score valid", losses(model, poses["valid"]))
        model.score = score
    train, valid = losses(model, poses["train"]), losses(model, poses["valid"])
    line(name + " train", train)
    line(name + " valid", valid)
    line(name + " gap", valid - train)
