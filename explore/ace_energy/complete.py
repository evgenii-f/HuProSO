"""Pose completion with an ACE energy as the prior: the reverse process of RSGM (inpainting) with the score of the energy
in place of the network (docs/experiments/2026-11-23_ace_descriptors.md). A quick check on the first poses of the
validation draw of the completion studies, without tuning: the inpainting point t81 of abl_residual. From the root:

    python explore/ace_energy/complete.py --fit ~/data/experiments/ace/energy_b_smooth_w123_p123.pt --poses 200
    python explore/ace_energy/complete.py --fit ... --network          # the reference network on the same poses

The energy is a linear fit of fit.py (its basis from the fit's settings), as an embedding.py model with the second
density zero. The model is the reference RSGM run (its noising, sampler and time grid) with its score replaced by
-grad E~_t / std(t). Metrics as scripts/evaluate.py completion (MPVPE mean, best of K, APD), K = 10.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding import Embedded, owners  # noqa: E402
from fit import REFERENCE, Bins, WithHarmonics  # noqa: E402
from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run, summary, valid_poses  # noqa: E402
from hp.models.ace_features import AceFeatures  # noqa: E402

T81 = dict(steps=400, schedule="edm", guidance=0.04934967328387599, resample=5, replace=False)


def energy_model(fit: str, reference: str, device: str):
    """The reference RSGM with the score of the linear energy of a fit."""
    model = build(dict(run=reference), device)
    saved = torch.load(Path(fit).expanduser(), weights_only=False)
    s = saved["settings"]
    if s.get("one_body"):
        raise ValueError("fits with --one-body are not supported here; use --wigner fits")
    basis = WithHarmonics(AceFeatures(smooth=not s.get("plain", False)), s.get("wigner", []), s.get("pairs", []))
    bins = Bins(model, model.loss_t_min, s["bins"])
    energy = Embedded(basis, bins, owners(basis)).to(device)
    energy.init_linear(saved["w"], scale2=0.0)
    energy.requires_grad_(False)

    def score(x, t, f=None):
        create = torch.is_grad_enabled() and x.requires_grad  # DPS differentiates through the score
        scaled = energy.scaled_score(x, t, create_graph=create)
        return scaled / model.joint_std(t.to(x.dtype))[..., None]

    model.score = score
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", default="~/data/experiments/ace/energy_b_smooth_w123_p123.pt")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--network", action="store_true", help="the reference network itself instead of the energy")
    parser.add_argument("--poses", type=int, default=200)
    parser.add_argument("--occlusions", nargs="+", default=list(OCCLUSIONS))
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-only", action="store_true", help="time one score call and stop")
    args = parser.parse_args()

    model = build(dict(run=args.reference), args.device) if args.network else energy_model(args.fit, args.reference,
                                                                                          args.device)
    if args.time_only:
        from hp.models.ace_features import _random_rotations
        x = _random_rotations(2000, 19, torch.Generator().manual_seed(0)).float().to(args.device).requires_grad_()
        t = torch.full((2000,), 0.1, device=args.device)
        t0 = time.perf_counter()
        s = model.score(x, t)
        s.sum().backward()
        print(f"one score call with its backward, 2000 poses: {time.perf_counter() - t0:.2f} s "
              f"(inpainting t81: {T81['steps'] * T81['resample']} calls per batch)")
        return
    gt, body = valid_poses(n=1000)[:args.poses], Body()
    rows = []
    for occ in args.occlusions:
        t0 = time.perf_counter()
        pred = run(model, gt, occ, k=10, batch=args.batch, together=True, progress=False, prior=dict(T81),
                   method="inpaint")
        m = summary(metrics(pred, gt, occ, body))
        rows.append((occ, m))
        print(f"{occ}: MPVPE mean {m['mpvpe_mean']:.2f}, best of 10 {m['mpvpe_min']:.2f}, APD {m['apd']:.2f} "
              f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    keys = ("mpvpe_mean", "mpvpe_min", "apd")
    mean = {k: sum(m[k] for _, m in rows) / len(rows) for k in keys}
    print(f"{'network' if args.network else 'energy'}, {args.poses} poses: mean {mean['mpvpe_mean']:.2f}, "
          f"best of 10 {mean['mpvpe_min']:.2f}, APD {mean['apd']:.2f}")


if __name__ == "__main__":
    main()
