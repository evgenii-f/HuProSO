"""Motion denoising (PoseD-Flow's Tab. 3, hp/evaluation/denoising.py) with the ACE energies: DPoser's optimization
(score distillation on SO(3)^J) with the reference RSGM's sampler noise and the score of an energy
(complete.energy_model), at the settings tuned for RSGM 512x4 c10 (scripts/evaluate_denoising.py, SDS), on the 100
validation sequences. The reference network is the first row; the per-sequence MPJPE differences against it are
paired (same noise, same seeds). From the root:

    python explore/ace_energy/denoise.py ~/data/experiments/ace/head/energy_two_head.pt \
        --head energy score band --device cuda:0 --out ~/data/experiments/ace/head/denoise_two_head.pt

Validation only (no --split test here).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from complete import energy_model  # noqa: E402
from diagnose import bootstrap  # noqa: E402
from evaluate_denoising import SDS  # noqa: E402
from fit import REFERENCE  # noqa: E402
from generate import two_heads  # noqa: E402
from hp.evaluation.completion import build  # noqa: E402
from hp.evaluation.denoising import Body, gt_joints, metrics, noisy_joints, run, split_sequences  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fits", nargs="*", help="energy_*.pt of embedding.py")
    parser.add_argument("--head", choices=("energy", "score", "band"), nargs="+", default=["band"])
    parser.add_argument("--split-s", type=float, default=0.04)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.04, 0.1])
    parser.add_argument("--first", type=int, default=None, help="only the first n validation sequences")
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="a .pt file of the per-frame metrics (resumable per row)")
    args = parser.parse_args()
    gt, _ = split_sequences("valid", first=args.first)
    body = Body(device=args.device)
    rows = [("network", None, None)] + [(Path(f).stem + (f", {h}" if two_heads(f) else ""), f, h)
                                         for f in args.fits for h in (args.head if two_heads(f) else [None])]
    out = Path(args.out).expanduser() if args.out else None
    done = torch.load(out, weights_only=True) if out is not None and out.is_file() else {}
    print("| model | sigma (mm) | MPJPE (mm) | MPJPE - network (mm) | minutes |\n|---|---|---|---|---|", flush=True)
    for sigma in args.sigmas:
        noisy = noisy_joints(gt_joints(gt, body), sigma)
        base = None
        for name, path, head in rows:
            key = f"{name} / {sigma:g}"
            t0 = time.perf_counter()
            if key not in done:
                if path is None:
                    model = build(dict(run=REFERENCE), args.device)
                else:
                    model = energy_model(path, REFERENCE, args.device, head=head or "band", split_s=args.split_s)[0]
                pred = run(model, gt, noisy, sigma, body, "dposer", args.batch, progress=False, **SDS[sigma])
                m = metrics(pred, gt, noisy, body)
                done[key] = {k: torch.as_tensor(v) for k, v in m.items()}
                if out is not None:
                    torch.save(done, out)
            seq = np.nanmean(done[key]["mpjpe"].double().numpy(), 1)  # per sequence (mm)
            base = seq if path is None else base
            diff = bootstrap(seq - base) if path is not None else "–"
            print(f"| {name} | {1000 * sigma:.0f} | {seq.mean():.2f} | {diff} | "
                  f"{(time.perf_counter() - t0) / 60:.1f} |", flush=True)


if __name__ == "__main__":
    main()
