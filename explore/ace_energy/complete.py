"""Pose completion with an ACE energy as the prior: the reverse process of RSGM (inpainting) with the score of the energy
in place of the network (docs/experiments/2026-11-23_ace_descriptors.md). A quick check on the first poses of the
validation draw of the completion studies, without tuning: the inpainting point t81 of abl_residual. From the root:

    python explore/ace_energy/complete.py --fit ~/data/experiments/ace/energy_b_smooth_w123_p123.pt --poses 200
    python explore/ace_energy/complete.py --fit ... --network          # the reference network on the same poses
    python explore/ace_energy/complete.py --fit ... --band 0.04 0.4    # the energy at 0.04 <= s < 0.4, else the network
    python explore/ace_energy/complete.py --fit ... --no-dps           # replacement of the visible joints, no guidance

The energy is a linear fit of fit.py or an embedding of embedding.py (inspect_energy.load_energy). The model is the
reference RSGM run (its noising, sampler and time grid) with its score replaced by -grad E~_t / std(t). Metrics as
scripts/evaluate.py completion (MPVPE mean, best of K, APD), K = 10, and the spread of the errors over the hypotheses:
quantiles, and the share of the excess over the median that the worst 10% of the hypotheses carry.

With --out FILE the errors of every hypothesis are saved after each occlusion. The same command resumes (finished
occlusions are skipped); the first Ctrl+C or SIGTERM stops after the current occlusion.
"""
import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import REFERENCE  # noqa: E402
from inspect_energy import load_energy  # noqa: E402
from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run, summary, valid_poses  # noqa: E402

T81 = dict(steps=400, schedule="edm", guidance=0.04934967328387599, resample=5, replace=False)
NO_DPS = dict(T81, guidance=0.0, replace=True)


def energy_model(fit: str, reference: str, device: str, band=(0.0, float("inf"))):
    """The reference RSGM with the score of the energy at the noise levels band[0] <= s(t) < band[1], the network's own
    score elsewhere. One noise level per call, as the samplers and the likelihood ODE do."""
    model = build(dict(run=reference), device)
    energy, _, _, info = load_energy(fit, device)
    network = model.score

    def score(x, t, f=None):
        if not band[0] <= float(model.s(t[0])) < band[1]:
            return network(x, t, f)
        create = torch.is_grad_enabled() and x.requires_grad  # DPS differentiates through the score
        scaled = energy.scaled_score(x, t, create_graph=create)
        return scaled / model.joint_std(t.to(x.dtype))[..., None]

    model.score = score
    return model, energy, info


def spread(mpvpe: np.ndarray) -> str:
    """Where the error sits: quantiles over all hypotheses [N, K], and the share of the total excess over the median
    carried by the worst 10% of the hypotheses."""
    e = np.sort(mpvpe.ravel())
    excess = np.clip(e - np.median(e), 0, None)
    top = excess[-len(e) // 10:].sum() / max(excess.sum(), 1e-9)
    q = np.quantile(e, [0.1, 0.5, 0.9, 0.99])
    return f"q10/50/90/99 {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f}, worst 10% carry {100 * top:.0f}% of the excess"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", default="~/data/experiments/ace/energy_b_smooth_w123_p123.pt")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--network", action="store_true", help="the reference network itself instead of the energy")
    parser.add_argument("--band", type=float, nargs=2, default=(0.0, float("inf")), metavar=("S_LO", "S_HI"),
                        help="noise levels s(t) at which the energy is used, the network elsewhere")
    parser.add_argument("--no-dps", action="store_true", help="replacement of the visible joints instead of guidance")
    parser.add_argument("--poses", type=int, default=200)
    parser.add_argument("--occlusions", nargs="+", default=list(OCCLUSIONS))
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="a .pt file for the errors per hypothesis (resumable)")
    parser.add_argument("--time-only", action="store_true", help="time one score call and stop")
    args = parser.parse_args()

    if args.network:
        model, info = build(dict(run=args.reference), args.device), "network"
    else:
        model, _, info = energy_model(args.fit, args.reference, args.device, tuple(args.band))
        if tuple(args.band) != (0.0, float("inf")):
            info += f", at {args.band[0]:g} <= s < {args.band[1]:g}"
    options = dict(NO_DPS if args.no_dps else T81)
    info += ", no DPS" if args.no_dps else ""
    if args.time_only:
        from hp.models.ace_features import _random_rotations
        x = _random_rotations(2000, 19, torch.Generator().manual_seed(0)).float().to(args.device).requires_grad_()
        t = torch.full((2000,), 0.1, device=args.device)
        t0 = time.perf_counter()
        s = model.score(x, t)
        s.sum().backward()
        float(x.grad.sum())  # waits for the device
        print(f"one score call with its backward, 2000 poses: {time.perf_counter() - t0:.2f} s "
              f"(inpainting t81: {T81['steps'] * T81['resample']} calls per batch)")
        return

    settings = dict(fit=None if args.network else Path(args.fit).name, info=info, options=options, poses=args.poses,
                    batch=args.batch)
    out = Path(args.out).expanduser() if args.out else None
    done = {}
    if out and out.exists():
        saved = torch.load(out, weights_only=True)
        if saved["settings"] != settings:
            raise SystemExit(f"{out} was made with other settings: {saved['settings']}")
        done = saved["occlusions"]
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):  # the first one: finish the occlusion; the second: as usual
        signal.signal(sig, lambda number, _: (stop.append(number), signal.signal(number, signal.SIG_DFL),
                                              print("\nstopping after this occlusion", flush=True)))
    gt, body = valid_poses(n=1000)[:args.poses], Body()
    print(f"{info}, {args.poses} poses", flush=True)
    for occ in args.occlusions:
        if occ not in done:
            t0 = time.perf_counter()
            pred = run(model, gt, occ, k=10, batch=args.batch, together=True, progress=True, prior=dict(options),
                       method="inpaint")
            m = metrics(pred, gt, occ, body)
            total = summary(m)
            done[occ] = dict(mpvpe=torch.as_tensor(m["mpvpe"]), apd=torch.as_tensor(m["apd"]),
                             apd_mean=total["apd"], mpvpe_mean=total["mpvpe_mean"], mpvpe_min=total["mpvpe_min"],
                             minutes=(time.perf_counter() - t0) / 60)
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(settings=settings, occlusions=done), out)
        m = done[occ]
        print(f"{occ}: MPVPE mean {m['mpvpe_mean']:.2f}, best of 10 {m['mpvpe_min']:.2f}, APD {m['apd_mean']:.2f} "
              f"({m['minutes']:.1f} min); {spread(m['mpvpe'].numpy())}", flush=True)
        if stop:
            print("stopped; the same command resumes" if out else "stopped (no --out, nothing saved)")
            return
    rows = [done[occ] for occ in args.occlusions]
    mean = {k: sum(m[k] for m in rows) / len(rows) for k in ("mpvpe_mean", "mpvpe_min", "apd_mean")}
    print(f"{info}, {args.poses} poses: mean {mean['mpvpe_mean']:.2f}, best of 10 {mean['mpvpe_min']:.2f}, "
          f"APD {mean['apd_mean']:.2f}")


if __name__ == "__main__":
    main()
