"""Validation log-likelihood of RSGM runs (EMA weights) on the 500 poses of the quick check of
scripts/train_rsgm_posedflow.py (c10 validation, seed 10), with quantiles and the number of poses below 0.
Run from the root of the repository on the Mac:

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/loglik_valid.py ~/data/experiments/<run>/run_0001 ...
"""
import sys

import torch

from hp.evaluation.data import load_reference
from hp.models.io import load_model

valid = load_reference("valid", "~/data/AMASS_rotations_posedflow_c10", "posedflow")
poses = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(10))[:500]]
for run in sys.argv[1:]:
    model = load_model(run, device="mps", weights="weights_ema")
    torch.manual_seed(10)
    with torch.no_grad():
        ll = torch.cat([model.log_prob(poses[i:i + 100].to(model.device)).cpu() for i in range(0, 500, 100)])
    q = torch.quantile(ll.double(), torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64))
    print(run.rstrip("/").split("/")[-2], f"mean {ll.mean():.2f} +- {ll.std() / 500 ** 0.5:.2f}, quantiles "
          "0/1/5/50/95/100%:", " ".join(f"{v:.1f}" for v in q), f"| below 0: {(ll < 0).sum()}", flush=True)
