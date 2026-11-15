"""The error of DPoser's completion optimization along its 200 steps (every 10th step): the hidden-joint error (deg),
the spread of the K hypotheses (deg) and the mean |x - x0_hat|, with DPoser's own denoiser and with RSGM's (the hybrid
of hybrid.py), on 300 validation poses, left_leg and arms:
    PYTHONPATH=src python explore/evaluation/completion_diagnostics/trajectory.py <RSGM run folder> [rsgm]
With a second argument only RSGM's denoiser runs."""
import sys, time, torch, roma
from hp.evaluation.completion import KEEP, OCCLUSIONS, TOES, Body, at_rest, build, metrics, summary, valid_poses
from hp.models import igso3
from hp.models.rsgm import exp_frame
from hp.models.score_completion import one_step_denoise, rsgm_denoise

K = 10
dp = build(dict(run="~/data/experiments/full/dposer_posedflow_s10/run_0001"), "mps")
rs = build(dict(run=sys.argv[1]), "mps")
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
    """RSGM's denoised estimate of the normalized vectors x [B, 57] at DPoser's time t_d (matched mean noise)."""
    a, sd = dp.marginal(t_d)
    sigma = ((sd / a)[:, None] * data_std[None]).pow(2).mean(1).sqrt()          # RMS radians per coordinate
    s = sigma.cpu().double() ** 2
    R = dp.denormalize(x)                                                        # [B, 19, 3, 3]
    b = len(x)
    noise = sampler.sample(s[:, None].expand(b, 19))[..., None] * igso3.random_axes((b, 19))
    Rt = exp_frame(R.cpu().double(), noise).float().to(dev)
    t_r = rs.t_of_s(s).float().to(dev)
    R0 = rsgm_denoise(rs, Rt, t_r)
    return dp.normalize(R0)


def optimize_tracked(observed, visible, init, denoiser, gt19, every=10):
    """dposer_complete with a choice of denoiser; every `every` steps the hidden-joint error (deg) of the iterate, its
    spread over the K hypotheses (deg), and the mean |x - x0_hat| of the hidden coordinates (normalized)."""
    n = observed.shape[0]
    y = dp.normalize(observed.to(dev))
    mask = visible.to(dev).repeat_interleave(3)[None].expand_as(y)
    x = torch.where(mask, y, init.to(dev)).detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.1, betas=(0.9, 0.99))
    hidden = ~visible
    log = []
    with torch.enable_grad():
        for it in range(2):
            for i in range(100):
                step = it * 100 + i
                t = torch.full((n,), 0.05 + (200 - step - 1) / 200 * 0.10, device=dev)
                a, sd = dp.marginal(t)
                with torch.no_grad():
                    if denoiser == "dposer":
                        x0h = one_step_denoise(dp, a[:, None] * x + sd[:, None] * torch.randn_like(x), t)
                    else:
                        x0h = rsgm_hat(x.detach(), t)
                if step % every == 0 or step == 199:
                    with torch.no_grad():
                        R = dp.denormalize(x.detach()).cpu()
                        err = roma.rotmat_geodesic_distance(R[:, hidden], gt19[:, hidden]) * 57.3      # [n, H]
                        hyp = R[:, hidden].view(K, -1, int(hidden.sum()), 3, 3)
                        spread = roma.rotmat_geodesic_distance(hyp, hyp[:1]).mean() * 57.3 * K / (K - 1)
                        pull = (x - x0h).abs()[:, ~mask[0]].mean()
                    log.append((step, float(t[0]), float(err.mean()), float(spread), float(pull)))
                w = 0.5 * torch.sqrt(1 + a / sd)[:, None]
                loss = 100 / (1 + it) * ((x * mask - y * mask) ** 2).mean() + 0.1 * (it + 1) * (w * (x - x0h) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
    return log

N = 300
gt = valid_poses(n=N)
for occ in ("left_leg", "arms"):
    vis = torch.ones(21, dtype=torch.bool); vis[list(OCCLUSIONS[occ])] = False
    logs = {}
    for den in (["dposer", "rsgm"] if len(sys.argv) < 3 else ["rsgm"]):
        torch.manual_seed(0)
        init = torch.randn(N, 57).repeat(K, 1)
        g = gt[:, KEEP].repeat(K, 1, 1, 1)
        logs[den] = optimize_tracked(g, vis[KEEP], init, den, g)
    if "dposer" not in logs:
        print(occ, "step t | RSGM err spread |x-x0h|")
        for st, t, e, sp, p in logs["rsgm"]:
            print(f"  {st:4d} {t:.3f} | {e:7.2f} {sp:6.2f} {p:6.3f}", flush=True)
        continue
    print(f"{occ}: step  t      | DPoser err  spread  |x-x0h| | RSGM err  spread  |x-x0h|")
    for (st, t, e1, s1, p1), (_, _, e2, s2, p2) in zip(logs["dposer"], logs["rsgm"]):
        print(f"        {st:4d}  {t:.3f} | {e1:7.2f}   {s1:6.2f}  {p1:6.3f}  | {e2:7.2f}  {s2:6.2f}  {p2:6.3f}", flush=True)
