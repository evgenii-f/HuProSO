"""SDS with the settings of validation trial 5 (fixed before the test) on the 5000 test poses of PoseD-Flow, K=10,
for the RSGMs with the likelihood weighting; cached as the rows of evaluate_completion.py."""
import sys, time
from pathlib import Path
sys.path.insert(0, "scripts")
from evaluate_completion import CACHE, ROTATIONS
from hp.evaluation.completion import Body, OCCLUSIONS, cached_metrics, cached_run, summary, test_poses
E = "~/data/experiments/"
t_max = 0.03187115040476566
SDS = dict(t_max=t_max, t_min=t_max * 0.5923495200996912, lr=0.04777483122927568, start="noise",
           prior_weight=0.09072109154659835, scale="std", stats=ROTATIONS)
MODELS = {
    "RSGM 512x4: their c10, SDS (validation trial 5)": E + "rsgm_posedflow_small_s10/run_0001",
    "RSGM 512x4 lik: their c10, SDS (validation trial 5)": E + "rsgm_posedflow_small_lik_s10/run_0001",
    "RSGM 512x4 lik: their c10, 10k, SDS (validation trial 5)": E + "rsgm_posedflow_small_lik_s10/run_0001/checkpoint_010000",
    "RSGM 512x4 lik: their full, SDS (validation trial 5)": E + "full/rsgm_posedflow_small_lik_s10/run_0001",
    "RSGM 1536x6: their c10, SDS (validation trial 5)": E + "rsgm_posedflow_s10/run_0001",
    "RSGM 1536x6 lik: their c10, SDS (validation trial 5)": E + "rsgm_posedflow_search_lik_s10/run_0001",
    "RSGM 1536x6 lik: their full, SDS (validation trial 5)": E + "full/rsgm_posedflow_search_lik_s10/run_0001",
}
gt, body = test_poses(ROTATIONS), Body()
for label, path in MODELS.items():
    spec = dict(run=path, method="dposer", prior=SDS)
    row, t0 = [], time.time()
    for occ in OCCLUSIONS:
        pred = cached_run(Path(CACHE), label, spec, gt, occ, "mps", rotations=ROTATIONS, batch=500, together=True)
        s = summary(cached_metrics(Path(CACHE), label, occ, pred, gt, body)[0])
        row.append((s["mpvpe_mean"], s["apd"], s["mpvpe_min"]))
    mean = sum(r[0] for r in row) / 4
    print(f"{label:58s} | " + " | ".join(f"{a:5.1f}/{b:5.2f}/{c:5.1f}" for a, b, c in row) + f" | mean {mean:.2f} | {time.time()-t0:.0f}s", flush=True)
