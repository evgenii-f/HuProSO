"""Configs of the pairs x data weighting gate (section 20 of the score networks report).

The settings are ready points of the validation fronts of `fact_RDW` (seed 10, inputs `local`) of the same beta: the
lowest mean at APD >= 4 / 5 / 6 / 7 cm. Every study has as many trials as points, so nothing is searched. Run from the
root of the repository:

    python explore/evaluation/pairs_gate/make_configs.py ~/data/experiments/pairs_gate
"""
import os
import sys

import optuna
import yaml

optuna.logging.set_verbosity(optuna.logging.ERROR)
FRONTS = os.path.expanduser("~/data/experiments/completion/optuna")
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/data/experiments/pairs_gate")
POINTS = {"b0": {"inpaint": [99, 111], "sds": [28, 34, 85, 107], "dflow": [53, 47]},
          "b3p3": {"inpaint": [55, 16], "sds": [23, 30, 71, 114], "dflow": [0, 34]}}
E = "~/data/experiments"
MODELS = {("pairs_s10", "b0"): f"{E}/pairs/rsgm_posedflow_fact_RDW_s10",
          ("pairs_s11", "b0"): f"{E}/pairs/rsgm_posedflow_fact_RDW_s11",
          ("ctrl_s11", "b0"): f"{E}/factorial/rsgm_posedflow_fact_RDW_s11",
          ("pairs_s10", "b3p3"): f"{E}/pairs/rsgm_posedflow_rdw_b3.3_s10",
          ("pairs_s11", "b3p3"): f"{E}/pairs/rsgm_posedflow_rdw_b3.3_s11",
          ("ctrl_s11", "b3p3"): f"{E}/pairs_control/rsgm_posedflow_rdw_b3.3_s11",
          ("repro_s10", "b0"): f"{E}/factorial/rsgm_posedflow_fact_RDW_s10"}

os.makedirs(OUT, exist_ok=True)
for (name, beta), run in MODELS.items():
    for method, numbers in POINTS[beta].items():
        study_name = f"rsgm_{method}_mean_apd_rdw_{beta}"
        with open(f"config/completion_tuning/{study_name}.yaml") as f:
            config = yaml.safe_load(f)
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{FRONTS}/{study_name}.db")
        if name == "repro_s10":  # one point per method: the code reproduces the stored front
            numbers = [111] if method == "inpaint" else numbers[:1]
        config.update(model=run + "/run_0001", n_trials=len(numbers),
                      first=[dict(study.trials[n].params) for n in numbers])
        with open(f"{OUT}/gate_{beta}_{name}_{method}.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
