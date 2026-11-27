"""Gibbs sampling of the pairwise matrix-Fisher energy of pseudo_lik.py: every joint in turn is drawn exactly from its
conditional, a matrix Fisher distribution exp <F_j, x_j>, so no score and no noise level is used. A matrix Fisher
distribution of rotations is a Bingham distribution of unit quaternions, <F, R(q)> = q^T K(F) q, drawn by rejection
from an angular central Gaussian envelope (Kent, Ganeiber and Mardia 2013, about 50-75% accepted in 4 dimensions).
Validation only. Modes:

- --check: the sampler against the mean of the distribution (d log c / dF of pseudo_lik.py) and <F, R(q)> = q^T K q;
- --generate: unconditional samples (chains from uniform rotations, --sweeps each, the last state kept) and the
  harness of generate.py (hp.evaluation.evaluate: Wasserstein, recall, precision against the validation poses; FID_val,
  APD, dNN), with the same settings as its runs of 1000 samples in batches of 500;
- --inpaint: completion of the first 200 validation poses of complete.py, K = 10 chains per pose, the visible joints
  kept at the truth and the hidden ones started uniform; MPVPE mean, best of 10 and APD per occlusion as complete.py
  (hidden toes at rest), for every number of sweeps given. --out saves them as complete.py --out (with the poses).

From the root of the repository:

    python explore/ace_energy/gibbs.py --check
    python explore/ace_energy/gibbs.py --generate --sweeps 200 1000
    python explore/ace_energy/gibbs.py --inpaint --sweeps 50 200 1000 --out ~/data/experiments/ace/pl/inpaint_gibbs.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseudo_lik import ACE, PairwiseFisher, log_c  # noqa: E402

J = 19


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Rotations [..., 3, 3] of unit quaternions q [..., 4] = (w, x, y, z)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z], -1)], -2)


def bingham_of_fisher(f: torch.Tensor) -> torch.Tensor:
    """K [..., 4, 4] with <F, R(q)> = q^T K q for unit q = (w, x, y, z)."""
    f11, f12, f13 = f[..., 0, 0], f[..., 0, 1], f[..., 0, 2]
    f21, f22, f23 = f[..., 1, 0], f[..., 1, 1], f[..., 1, 2]
    f31, f32, f33 = f[..., 2, 0], f[..., 2, 1], f[..., 2, 2]
    rows = [[f11 + f22 + f33, f32 - f23, f13 - f31, f21 - f12],
            [f32 - f23, f11 - f22 - f33, f12 + f21, f13 + f31],
            [f13 - f31, f12 + f21, -f11 + f22 - f33, f23 + f32],
            [f21 - f12, f13 + f31, f23 + f32, -f11 - f22 + f33]]
    return torch.stack([torch.stack(r, -1) for r in rows], -2)


def sample_fisher(f: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """One rotation [B, 3, 3] from each matrix Fisher distribution exp <F, R> of f [B, 3, 3] (float64)."""
    a = -bingham_of_fisher(f)  # the density exp(-q^T A q) of q
    lam, u = torch.linalg.eigh(a)
    lam = lam - lam[:, :1]  # the smallest eigenvalue 0
    lo, hi = torch.ones(len(f), dtype=f.dtype, device=f.device), torch.full((len(f),), 4.0, dtype=f.dtype, device=f.device)
    for _ in range(60):  # b with sum 1 / (b + 2 lam) = 1, in [1, 4]
        b = (lo + hi) / 2
        big = (1 / (b[:, None] + 2 * lam)).sum(1) > 1
        lo, hi = torch.where(big, b, lo), torch.where(big, hi, b)
    b = ((lo + hi) / 2)[:, None]
    omega = 1 + 2 * lam / b  # the envelope's precision in the eigenbasis
    out = torch.empty(len(f), 4, dtype=f.dtype, device=f.device)
    todo = torch.arange(len(f), device=f.device)
    while len(todo):
        y = torch.randn(len(todo), 4, dtype=f.dtype, device=f.device, generator=gen) / omega[todo].sqrt()
        x = y / y.norm(dim=1, keepdim=True)
        bb = b[todo, 0]
        log_ratio = (-(lam[todo] * x ** 2).sum(1) + 2 * torch.log((omega[todo] * x ** 2).sum(1))
                     + (4 - bb) / 2 - 2 * torch.log(4 / bb))
        ok = torch.log(torch.rand(len(todo), dtype=f.dtype, device=f.device, generator=gen)) < log_ratio
        out[todo[ok]] = (u[todo[ok]] @ x[ok, :, None])[..., 0]
        todo = todo[~ok]
    return quat_to_rotmat(out)


def uniform(n: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(n, 4, dtype=torch.float64, device=gen.device, generator=gen)
    return quat_to_rotmat(q / q.norm(dim=1, keepdim=True))


def gibbs(model: PairwiseFisher, x: torch.Tensor, free, sweeps: int, gen: torch.Generator, trace=(), mh: int = 3,
          stats=None) -> torch.Tensor:
    """sweeps of Gibbs updates of the joints `free` (a random order per sweep) of poses x [B, 19, 3, 3]; the mean
    energy is printed after the sweeps in `trace`. A model with one-body terms g_j (boltzmann.Boltzmann with degrees)
    has the conditional exp(<F_j, x_j> + g_j(x_j)): then every update is `mh` Metropolis steps with the matrix Fisher
    exp <F_j, x_j> as the independent proposal (accepted with min(1, exp(g_j(new) - g_j(old)))), and stats (a list)
    collects the share accepted."""
    with torch.no_grad():
        w, a = model.coupling().detach(), model.a.detach()
        extra = bool(getattr(model, "degrees", ()))
        x = x.double().clone()
        v = x.reshape(len(x), -1)
        free = torch.as_tensor(list(free))
        for sweep in range(1, sweeps + 1):
            for j in free[torch.randperm(len(free), generator=gen, device=gen.device).cpu()].tolist():
                f = (a[9 * j:9 * j + 9] + v @ w[:, 9 * j:9 * j + 9]).reshape(-1, 3, 3)
                if not extra:
                    x[:, j] = sample_fisher(f, gen)
                else:
                    old = model.one_body(x[:, j], j)
                    for _ in range(mh):
                        new = sample_fisher(f, gen)
                        g_new = model.one_body(new, j)
                        ok = torch.log(torch.rand(len(x), dtype=x.dtype, device=x.device, generator=gen)) < g_new - old
                        x[ok, j] = new[ok]
                        old = torch.where(ok, g_new, old)
                        if stats is not None:
                            stats.append(float(ok.double().mean()))
                v = x.reshape(len(x), -1)
            if sweep in trace:
                print(f"    sweep {sweep}: mean energy {float(model.energy(x).mean()):.2f}", flush=True)
    return x


def check() -> None:
    gen = torch.Generator().manual_seed(0)
    f = torch.randn(4, 3, 3, dtype=torch.float64, generator=gen) * torch.tensor([0.5, 3.0, 20.0, 200.0],
                                                                                dtype=torch.float64)[:, None, None]
    q = torch.randn(1000, 4, dtype=torch.float64, generator=gen)
    q = q / q.norm(dim=1, keepdim=True)
    r = quat_to_rotmat(q)
    k = bingham_of_fisher(f[0])
    print(f"max |<F, R(q)> - q^T K q| = {float(((r * f[0]).sum((1, 2)) - ((q @ k) * q).sum(1)).abs().max()):.2e}, "
          f"max |R^T R - I| = {float((r.transpose(1, 2) @ r - torch.eye(3, dtype=r.dtype)).abs().max()):.2e}")
    for i in range(len(f)):
        fi = f[i].clone().requires_grad_(True)
        mean, = torch.autograd.grad(log_c(fi[None])[0], fi)
        t0 = time.time()
        s = sample_fisher(fi.detach().expand(100_000, 3, 3).contiguous(), gen)
        err = float((s.mean(0) - mean).abs().max())
        print(f"F scale {float(f[i].norm()):.1f}: max |sample mean - E R| = {err:.4f} (100,000 draws, "
              f"{time.time() - t0:.2f} s; Monte Carlo error about {1 / np.sqrt(1e5):.4f})")


def load(path: str) -> PairwiseFisher:
    """A fit of pseudo_lik.py or boltzmann.py."""
    from boltzmann import Boltzmann

    saved = torch.load(Path(path).expanduser(), weights_only=True)
    model = Boltzmann(saved.get("degrees", ()))
    missing, _ = model.load_state_dict(saved["state"], strict=False)
    assert set(missing) <= {"c"}, missing
    print(f"fit {path}: ridge {saved['ridge']}, one-body degrees {list(model.degrees)}", flush=True)
    return model.requires_grad_(False)


def generate(model, sweeps_list, n: int, batch: int, out) -> None:
    from generate import N_CONTRAST, joint_metrics  # noqa: E402
    from fit import REFERENCE, ROTATIONS, SPLIT  # noqa: E402
    from hp.evaluation.completion import build  # noqa: E402
    from hp.evaluation.data import load_reference  # noqa: E402
    from hp.evaluation.evaluate import EvalConfig, evaluate  # noqa: E402
    from hp.evaluation.posedflow_protocol import SmplSkeleton  # noqa: E402

    valid, train = load_reference("valid", ROTATIONS, SPLIT), load_reference("train", ROTATIONS, SPLIT)
    skeleton = SmplSkeleton()
    joints = skeleton.joints(valid).reshape(len(valid), -1).numpy()
    reference = joints.mean(0), np.cov(joints, rowvar=False)
    contrast = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:N_CONTRAST]]
    cfg = EvalConfig(n_generated=n, n_wasserstein=min(2000, n), n_queries=min(2000, n), n_likelihood=0, repeats=1,
                     sample_batch=batch)
    holder = build(dict(run=REFERENCE), "cpu")  # only its interface: its sample() is replaced by the Gibbs samples
    holder.has_log_prob = False
    print("| model | Wasserstein | recall | precision | FID_val | APD (cm) | dNN | minutes |\n"
          "|---|---|---|---|---|---|---|---|")
    m = joint_metrics(contrast[:n], skeleton, reference, train)
    print(f"| train poses | – | – | – | {m['fid']:.4f} | {m['apd']:.2f} | {m['dnn']:.4f} | – |", flush=True)
    for sweeps in sweeps_list:
        t0 = time.perf_counter()
        gen = torch.Generator().manual_seed(0)
        poses = gibbs(model, uniform(n * J, gen).reshape(n, J, 3, 3), range(J), sweeps, gen,
                      trace={1, 10, sweeps // 2, sweeps}).float()
        pieces = iter(poses.split(batch))
        holder.sample = lambda k, cond=None: next(pieces)
        report = evaluate(holder, valid, cfg)
        m = joint_metrics(poses, skeleton, reference, train)
        print(f"| Gibbs, {sweeps} sweeps | {report['wasserstein']['generated']['joint_mean_deg']['mean']:.2f} | "
              f"{report['recall']['mean_deg']['mean']:.2f} | {report['precision']['mean_deg']['mean']:.2f} | "
              f"{m['fid']:.4f} | {m['apd']:.2f} | {m['dnn']:.4f} | {(time.perf_counter() - t0) / 60:.1f} |", flush=True)
        if out:
            Path(out).expanduser().parent.mkdir(parents=True, exist_ok=True)
            torch.save(poses, Path(out).expanduser().with_name(f"{Path(out).stem}_{sweeps}.pt"))


def inpaint(model, sweeps_list, poses: int, out) -> None:
    from hp.evaluation.completion import KEEP, OCCLUSIONS, TOES, Body, at_rest, metrics, summary, valid_poses

    gt, body = valid_poses(n=1000)[:poses], Body()
    k = 10
    saved = {}
    for sweeps in sweeps_list:
        rows = {}
        for occ, hidden21 in OCCLUSIONS.items():
            t0 = time.perf_counter()
            free = [i for i, j in enumerate(KEEP) if j in hidden21]
            gen = torch.Generator().manual_seed(0)
            x = gt[:, KEEP].double().repeat_interleave(k, 0)  # [N * K, 19, 3, 3], pose by pose
            x[:, free] = uniform(len(x) * len(free), gen).reshape(len(x), len(free), 3, 3)
            x = gibbs(model, x, free, sweeps, gen)
            pred = gt[:, None].repeat(1, k, 1, 1, 1)
            pred[:, :, KEEP] = x.float().reshape(poses, k, J, 3, 3)
            hidden_toes = [t for t in TOES if t in hidden21]
            pred = at_rest(pred, hidden_toes) if hidden_toes else pred
            m = metrics(pred, gt, occ, body)
            s = summary(m)
            rows[occ] = dict(mpvpe=torch.as_tensor(m["mpvpe"]), apd=torch.as_tensor(m["apd"]), apd_mean=s["apd"],
                             mpvpe_mean=s["mpvpe_mean"], mpvpe_min=s["mpvpe_min"], poses=pred[:, :, KEEP].float(),
                             minutes=(time.perf_counter() - t0) / 60)
            print(f"  {sweeps} sweeps, {occ}: MPVPE mean {s['mpvpe_mean']:.2f}, best of 10 {s['mpvpe_min']:.2f}, "
                  f"APD {s['apd']:.2f} ({rows[occ]['minutes']:.1f} min)", flush=True)
        mean = np.mean([r["mpvpe_mean"] for r in rows.values()])
        best = np.mean([r["mpvpe_min"] for r in rows.values()])
        apd = np.mean([r["apd_mean"] for r in rows.values()])
        print(f"Gibbs, {sweeps} sweeps, {poses} poses: mean {mean:.2f}, best of 10 {best:.2f}, APD {apd:.2f}", flush=True)
        saved[sweeps] = rows
        if out:
            p = Path(out).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            for s_, r in saved.items():
                torch.save(dict(settings=dict(sweeps=s_, k=k, poses=poses), occlusions=r),
                           p.with_name(f"{p.stem}_{s_}{p.suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", default=str(ACE / "pl" / "energy_pl.pt"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--inpaint", action="store_true")
    parser.add_argument("--sweeps", type=int, nargs="+", default=[200])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--poses", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.check:
        check()
        return
    model = load(args.fit)
    if args.generate:
        generate(model, args.sweeps, args.samples, args.batch, args.out)
    if args.inpaint:
        inpaint(model, args.sweeps, args.poses, args.out)


if __name__ == "__main__":
    main()
