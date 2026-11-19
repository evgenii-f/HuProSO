"""Motion denoising (validation) with SDS for the isotropic / per-joint-noise pairs of the score network ablation, at
the settings tuned for RSGM 512x4 c10 (scripts/evaluate_denoising.py, SDS), and for the aniso models also with t mapped
so that s(t') = s(t) / k: "t geo" (k = 12.69, the geometric mean of c_j: a joint with that c_j gets the noise of the
isotropic model) and "t max" (k = 42.79, the largest c_j: no joint gets more noise than in the isotropic model).
Separate cache, so that no row of scripts/evaluate_denoising.py is overwritten. From the root of the repository on
the Mac:

    PYTHONPATH=src python explore/evaluation/denoising_aniso/rows.py --split valid \
        --cache ~/data/experiments/denoising/cache_aniso --device mps
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import evaluate_denoising as ed  # noqa: E402

BETA_0, BETA_F = 0.001, 7.5  # the schedule of the runs (model.json)


def s_of(t):
    return BETA_0 * t + 0.5 * (BETA_F - BETA_0) * t * t


def t_of(s):
    return (-BETA_0 + math.sqrt(BETA_0 ** 2 + 2 * (BETA_F - BETA_0) * s)) / (BETA_F - BETA_0)


E = "~/data/experiments/"
PAIRS = {"abl_norm": E + "rsgm_posedflow_abl_norm_s10/run_0001",
         "abl_norm_aniso": E + "igso_speed/rsgm_posedflow_abl_norm_aniso_s10/run_0001",
         "abl_residual": E + "rsgm_posedflow_abl_residual_s10/run_0001",
         "abl_residual_aniso": E + "igso_speed/rsgm_posedflow_abl_residual_aniso_s10/run_0001"}
for name, run in PAIRS.items():
    ed.MODELS[f"RSGM {name}: their c10, DPoser method"] = dict(run=run, method="dposer", settings=ed.SDS)
    if name.endswith("aniso"):
        for k, tag in ((12.69, "geo"), (42.79, "max")):
            mapped = {sigma: {**v, "t_max": t_of(s_of(v["t_max"]) / k), "t_min": t_of(s_of(v["t_min"]) / k)}
                      for sigma, v in ed.SDS.items()}
            ed.MODELS[f"RSGM {name}: their c10, DPoser method, t {tag}"] = dict(run=run, method="dposer",
                                                                               settings=mapped)

if __name__ == "__main__":
    sys.exit(ed.main())
