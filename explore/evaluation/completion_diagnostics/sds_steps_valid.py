"""SDS (rsgm_complete through run(method="dposer")) with the settings of validation trial 5 and several steps per
iteration (comma-separated), on the 1000 validation poses of the Optuna studies, K=10:
    PYTHONPATH=src python explore/evaluation/completion_diagnostics/sds_steps_valid.py 100,200,400,800 "search lik full"
"""
import sys, time, torch, numpy as np
sys.path.insert(0, "scripts")
from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run, summary, valid_poses
ROT = "~/data/AMASS_rotations_posedflow_c10"
E = "~/data/experiments/"
SETTINGS = {
    "trial5": dict(t_max=0.0319, t_min=0.0319 * 0.592, lr=0.0478, prior_weight=0.0907, start="noise", scale="std"),
    "wide":   dict(t_max=0.08, t_min=0.016, lr=0.08, prior_weight=0.05, start="noise", scale="std"),
}
MODELS = {
    "small std c10":      E + "rsgm_posedflow_small_s10/run_0001",
    "small lik c10":      E + "rsgm_posedflow_small_lik_s10/run_0001",
    "small lik c10 @10k": E + "rsgm_posedflow_small_lik_s10/run_0001/checkpoint_010000",
    "search std c10":     E + "rsgm_posedflow_s10/run_0001",
    "search lik c10":     E + "rsgm_posedflow_search_lik_s10/run_0001",
    "small std full":     E + "full/rsgm_posedflow_small_s10/run_0001",
    "small lik full":     E + "full/rsgm_posedflow_small_lik_s10/run_0001",
    "search lik full":    E + "full/rsgm_posedflow_search_lik_s10/run_0001",
}
steps = [int(x) for x in sys.argv[1].split(",")]
only = sys.argv[2:]
setting = "trial5"
torch.manual_seed(10)
gt, body = valid_poses(ROT, 1000), Body()
for name in only:
  m = build({"run": MODELS[name]}, "mps")
  for n_steps in steps:
      t0, row, means, bests = time.time(), [], [], []
      for occ in OCCLUSIONS:
          torch.manual_seed(10)
          opts = dict(SETTINGS[setting], stats=ROT, steps_per_iter=n_steps)
          pred = run(m, gt, occ, 10, batch=500, together=True, progress=False, method="dposer", prior=opts)
          s = summary(metrics(pred, gt, occ, body))
          means.append(s["mpvpe_mean"]); bests.append(s["mpvpe_min"])
          row.append(f"{s['mpvpe_mean']:5.1f}/{s['apd']:5.2f}/{s['mpvpe_min']:5.1f}")
      print(f"steps {n_steps:4d} | {name:19s} | " + " | ".join(row) + f" | mean {np.mean(means):.2f} best {np.mean(bests):.2f} | {time.time()-t0:.0f}s", flush=True)
