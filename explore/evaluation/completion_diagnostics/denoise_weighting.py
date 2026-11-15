"""One-step denoising of 2000 validation poses at matched IGSO(3) noise levels: ratio error after / noise before
(< 1: the prior helps), for RSGM std vs likelihood weighting, final EMA and the checkpoint at the valid-loss minimum."""
import math, sys, torch, roma
from hp.evaluation.completion import KEEP, build, valid_poses
from hp.models import igso3
from hp.models.rsgm import exp_frame
from hp.models.score_completion import rsgm_denoise
E = "~/data/experiments/"
MODELS = {
    "small std c10":        dict(run=E + "rsgm_posedflow_small_s10/run_0001"),
    "small lik c10":        dict(run=E + "rsgm_posedflow_small_lik_s10/run_0001"),
    "small lik c10 @10k":   dict(run=E + "rsgm_posedflow_small_lik_s10/run_0001/checkpoint_010000"),
    "search std c10":       dict(run=E + "rsgm_posedflow_s10/run_0001"),
    "search lik c10":       dict(run=E + "rsgm_posedflow_search_lik_s10/run_0001"),
    "small std full":       dict(run=E + "full/rsgm_posedflow_small_s10/run_0001"),
    "small lik full":       dict(run=E + "full/rsgm_posedflow_small_lik_s10/run_0001"),
    "search lik full":      dict(run=E + "full/rsgm_posedflow_search_lik_s10/run_0001"),
}
torch.manual_seed(0)
x0 = valid_poses(n=2000)[:, KEEP].double()
n, deg = len(x0), 180 / math.pi
sigmas = [0.01, 0.02, 0.05, 0.1, 0.2]
noisy = {}
for sg in sigmas:
    s = sg ** 2
    v = igso3.AngleSampler(s * 0.99, s * 1.01).sample(torch.full((n, 19), s))[..., None] * igso3.random_axes((n, 19))
    noisy[sg] = exp_frame(x0, v)
print("ratio error/noise per sigma (rad per coordinate):", sigmas)
for name, spec in MODELS.items():
    m = build(spec, "mps")
    row = []
    for sg in sigmas:
        xt = noisy[sg]
        t = m.t_of_s(torch.tensor(sg ** 2, dtype=torch.float64)).float()
        with torch.no_grad():
            x0h = rsgm_denoise(m, xt.float().to(m.device), t.expand(n).to(m.device)).cpu().double()
        r = roma.rotmat_geodesic_distance(x0h, x0).mean() / roma.rotmat_geodesic_distance(xt, x0).mean()
        row.append(f"{r:.3f}")
    print(f"{name:20s} " + "  ".join(row), flush=True)
