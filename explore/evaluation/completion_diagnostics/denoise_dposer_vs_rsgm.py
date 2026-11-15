"""One-step denoising of validation poses: RSGM (IGSO(3) noise, x0_hat = x_t exp(s score)) against DPoser (noise in its
z-scored axis-angle space, Tweedie). Noise levels are matched by the mean angle of the noise over all joints; per joint
the noise of DPoser is anisotropic (proportional to the spread of the joint), so the per-joint comparison is the ratio
error after / noise before (1: no gain; < 1: the prior helps; > 1: it hurts)."""
import math, torch, roma
from hp.data.amass_cfg import AmassJoints4Learning
from hp.evaluation.completion import KEEP, build, valid_poses
from hp.models import igso3
from hp.models.score_completion import one_step_denoise, rsgm_denoise

torch.manual_seed(0)
x0 = valid_poses(n=2000)[:, KEEP]                                    # [N, 19, 3, 3]
names = [j.name for j in AmassJoints4Learning]
groups = {"legs": [i for i, n in enumerate(names) if any(k in n for k in ("Leg", "Foot", "Hip", "Knee", "Ankle"))],
          "spine": [i for i, n in enumerate(names) if any(k in n for k in ("Spine", "Neck", "Head"))],
          "arms": [i for i, n in enumerate(names) if any(k in n for k in ("Shoulder", "Arm", "Elbow", "Hand", "Collar", "Wrist"))]}
deg = 180 / math.pi
rsgm = build(dict(run="~/data/experiments/rsgm_posedflow_small_s10/run_0001"), "mps")
dposers = {"DPoser full": build(dict(run="~/data/experiments/full/dposer_posedflow_s10/run_0001"), "mps"),
           "DPoser c10 30k": build(dict(run="~/data/experiments/dposer_posedflow_rsgm_budget_s10/run_0001"), "mps")}
n = len(x0)

def angles(a, b):
    return roma.rotmat_geodesic_distance(a.cpu().double(), b.cpu().double())   # [N, 19]

def row(label, noise, err):
    r = err.mean(0) / noise.mean(0)
    parts = " ".join(f"{g} {r[idx].mean():.2f}" for g, idx in groups.items())
    print(f"  {label:15s} noise {noise.mean()*deg:6.2f} deg  error {err.mean()*deg:6.2f} deg  ratio {err.mean()/noise.mean():.2f} | {parts}")

print("joints:", {g: [names[i] for i in idx] for g, idx in groups.items()})
for sigma in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:          # RMS angle per coordinate (rad)
    print(f"sigma {sigma}  (mean noise angle ~ {sigma*1.596*deg:.2f} deg)")
    # RSGM: s = sigma^2
    s = torch.tensor(sigma ** 2, dtype=torch.float64)
    t = rsgm.t_of_s(s).float()
    v = igso3.AngleSampler(float(s) * 0.99, float(s) * 1.01).sample(torch.full((n, 19), float(s)))[..., None] * igso3.random_axes((n, 19))
    from hp.models.rsgm import exp_frame
    xt = exp_frame(x0.double(), v).float()
    with torch.no_grad():
        x0h = rsgm_denoise(rsgm, xt.to(rsgm.device), t.expand(n).to(rsgm.device)).cpu()
    row("RSGM", angles(xt, x0), angles(x0h, x0))
    # DPoser: t such that the mean noise angle (in radians, per coordinate std(t)/alpha * data_std) matches sigma
    for name, d in dposers.items():
        target = sigma
        ts = torch.linspace(1e-4, 0.6, 6000)
        a, sd = d.marginal(ts)
        rad = (sd / a)[:, None] * d.data_std.cpu()[None]                       # [T, 57]
        tt = ts[(rad.pow(2).mean(1).sqrt() - target).abs().argmin()]
        a, sd = d.marginal(tt.expand(n).to(d.device))
        xn = d.normalize(x0.to(d.device))
        xtn = a[:, None] * xn + sd[:, None] * torch.randn_like(xn)
        with torch.no_grad():
            x0hn = one_step_denoise(d, xtn, tt.expand(n).to(d.device))
        noisy = d.denormalize(xtn / a[:, None])
        row(f"{name} t={tt:.3f}", angles(noisy, x0), angles(d.denormalize(x0hn), x0))
