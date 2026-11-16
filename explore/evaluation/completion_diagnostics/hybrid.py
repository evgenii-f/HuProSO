"""DPoser's completion optimization one to one (its variable, losses, weights, lr, schedule) with the denoiser swapped:
  control: DPoser's own one-step denoising (must reproduce DPoser's numbers);
  hybrid:  the pose of the variable as rotations, IGSO(3) noise of the same mean angle (s = sigma_rad^2), RSGM's
           one-step denoising, back to the normalized axis-angle of the variable.
Validation poses, K hypotheses, all occlusions:
    RS=<RSGM run folder> PYTHONPATH=src python explore/evaluation/completion_diagnostics/hybrid.py 500 [dposer] [rsgm]
RS defaults to RSGM 512x4 c10 (rsgm_posedflow_small_s10)."""
import os, sys, time, torch, roma
from hp.evaluation.completion import KEEP, OCCLUSIONS, TOES, Body, at_rest, build, metrics, summary, valid_poses
from hp.models import igso3
from hp.models.rsgm import exp_frame
from hp.models.score_completion import one_step_denoise, rsgm_denoise

N, K = int(sys.argv[1]) if len(sys.argv) > 1 else 500, 10
dp = build(dict(run="~/data/experiments/full/dposer_posedflow_s10/run_0001"), "mps")
rs = build(dict(run=os.environ.get("RS", "~/data/experiments/rsgm_posedflow_small_s10/run_0001")), "mps")
dev = dp.device
data_std = dp.data_std
sampler = igso3.AngleSampler(1e-7, 1.0)

def rsgm_hat_aniso(x, t_d):
    """As rsgm_hat, but every joint gets the noise angle DPoser gives it (proportional to its spread): IGSO(3) of
    time s_j = (std/alpha * sigma_j)^2 per joint, with RSGM's time t of the RMS noise as the network's input."""
    a, sd = dp.marginal(t_d)
    per_joint = ((sd / a)[:, None, None] * data_std.view(1, 19, 3)).pow(2).mean(-1).sqrt()   # [B, 19] radians
    s_j = per_joint.cpu().double() ** 2
    s = s_j.mean(1)
    R = dp.denormalize(x)
    b = len(x)
    noise = sampler.sample(s_j)[..., None] * igso3.random_axes((b, 19))
    Rt = exp_frame(R.cpu().double(), noise).float().to(dev)
    t_r = rs.t_of_s(s).float().to(dev)
    # one-step denoising with the per-joint variance: x0_hat = x_t exp(s_j score)
    R0 = exp_frame(Rt, s_j.float().to(dev)[..., None] * rs.score(Rt, t_r))
    return dp.normalize(R0)


def rsgm_hat(x, t_d):
    """RSGM's denoised estimate of the normalized vectors x [B, 57] at DPoser's time t_d (matched mean noise).
    RSGM's time t_r: s(t_r) = mean over the joints of s_j^DPoser / c_j, s_j^DPoser = (std/alpha)^2 * the mean square
    of DPoser's data_std of joint j, and the noise is IGSO(3) of c_j s(t_r) per joint. For the isotropic model (c = 1)
    that is the mean square noise of DPoser, the same for every joint; with joint_scale (c_j ~ spread^2) it is close
    to DPoser's own noise of every joint."""
    a, sd = dp.marginal(t_d)
    s_dp = ((sd / a)[:, None, None] * data_std.view(1, 19, 3)).pow(2).mean(-1).cpu().double()   # [B, 19]
    c = torch.ones(19, dtype=torch.float64) if rs.joint_scale is None else rs.joint_scale.cpu().double()
    s = (s_dp / c).mean(1)
    R = dp.denormalize(x)                                                        # [B, 19, 3, 3]
    b = len(x)
    noise = sampler.sample(s[:, None] * c)[..., None] * igso3.random_axes((b, 19))
    Rt = exp_frame(R.cpu().double(), noise).float().to(dev)
    t_r = rs.t_of_s(s).float().to(dev)
    R0 = rsgm_denoise(rs, Rt, t_r)
    return dp.normalize(R0)

def optimize(observed, visible, init, denoiser):
    """dposer_complete with a choice of denoiser."""
    n = observed.shape[0]
    y = dp.normalize(observed.to(dev))
    mask = visible.to(dev).repeat_interleave(3)[None].expand_as(y)
    x = torch.where(mask, y, init.to(dev)).detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.1, betas=(0.9, 0.99))
    total = 200
    with torch.enable_grad():
        for it in range(2):
            for i in range(100):
                step = it * 100 + i
                t = torch.full((n,), 0.05 + (total - step - 1) / total * 0.10, device=dev)
                a, sd = dp.marginal(t)
                with torch.no_grad():
                    if denoiser == "dposer":
                        x0h = one_step_denoise(dp, a[:, None] * x + sd[:, None] * torch.randn_like(x), t)
                    elif denoiser == "rsgm_aniso":
                        x0h = rsgm_hat_aniso(x.detach(), t)
                    else:
                        x0h = rsgm_hat(x.detach(), t)
                w = 0.5 * torch.sqrt(1 + a / sd)[:, None]
                loss = 100 / (1 + it) * ((x * mask - y * mask) ** 2).mean() + 0.1 * (it + 1) * (w * (x - x0h) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
    return dp.denormalize(torch.where(mask, y, x.detach()))

gt = valid_poses(n=N); body = Body()
for denoiser in sys.argv[2:] or ("dposer", "rsgm"):
    row, t0 = [], time.time()
    for occ in OCCLUSIONS:
        vis = torch.ones(21, dtype=torch.bool); vis[list(OCCLUSIONS[occ])] = False
        torch.manual_seed(0)
        init = torch.randn(N, 57).repeat(K, 1)
        filled = optimize(gt[:, KEEP].repeat(K, 1, 1, 1), vis[KEEP], init, denoiser).float().cpu()
        pred = gt[:, None].repeat(1, K, 1, 1, 1)
        pred[:, :, KEEP] = filled.view(K, N, 19, 3, 3).transpose(0, 1)
        pred = at_rest(pred, [t for t in TOES if not vis[t]])
        s = summary(metrics(pred, gt, occ, body))
        row.append(f"{occ} {s['mpvpe_mean']:.1f}/{s['apd']:.2f}/{s['mpvpe_min']:.1f}")
    print(f"{denoiser:7s} ({time.time()-t0:.0f}s): " + " | ".join(row), flush=True)
