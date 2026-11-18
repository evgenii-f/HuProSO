"""Inpainting with the reverse process of RSGM (run(method="inpaint"), hp.models.inpainting) on the first N of the
1000 validation poses of the Optuna studies, K=10, all four occlusions, for named variants. Run from the root:

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/inpaint_valid.py 100 --out ~/data/experiments/completion/inpaint_valid.jsonl
    PYTHONPATH=src python explore/evaluation/completion_diagnostics/inpaint_valid.py 1000 replace-edm200 dps0.05-edm200 \\
        --model "small std c10" --device cuda:0 --out ...

Variants (VARIANTS): the options of rsgm_inpaint; NFE = steps * resample. --out appends one JSON line per model and
variant (per occlusion MPVPE mean, APD, best of K, their means, seconds); a row already in the file with the same
poses is skipped. Compare with dflow_valid.py on the same poses (D-Flow, Heun EDM 50).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflow_valid import MODELS, ROT  # noqa: E402

from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run, summary, valid_poses  # noqa: E402

VARIANTS = {
    "replace-uniform200": dict(steps=200),
    "replace-edm200": dict(steps=200, schedule="edm"),
    "replace-edm200-r5": dict(steps=200, schedule="edm", resample=5),
    "dps0.02-edm200": dict(steps=200, schedule="edm", replace=False, guidance=0.02),
    "dps0.1-edm200": dict(steps=200, schedule="edm", replace=False, guidance=0.1),
    "replace-dps0.05-edm200": dict(steps=200, schedule="edm", guidance=0.05),
    "dps0.2-edm200": dict(steps=200, schedule="edm", replace=False, guidance=0.2),
    "dps0.5-edm200": dict(steps=200, schedule="edm", replace=False, guidance=0.5),
    "dps0.2-edm500": dict(steps=500, schedule="edm", replace=False, guidance=0.2),
    "dps0.2-edm200-r5": dict(steps=200, schedule="edm", replace=False, guidance=0.2, resample=5),
    "replace-dps0.2-edm200-r5": dict(steps=200, schedule="edm", guidance=0.2, resample=5),
    # the best region of the Optuna studies of RSGM (400 steps, guidance ~0.055, r = 5, no replacement)
    "dps0.055-edm400-r5": dict(steps=400, schedule="edm", replace=False, guidance=0.055, resample=5),
    # for DPoser (dposer_inpaint): its guidance acts on z-scored axis-angle, another scale than the radians of RSGM
    "dps0.05-edm200": dict(steps=200, schedule="edm", replace=False, guidance=0.05),
    "dps0.05-edm200-r5": dict(steps=200, schedule="edm", replace=False, guidance=0.05, resample=5),
    "dps0.1-edm200-r5": dict(steps=200, schedule="edm", replace=False, guidance=0.1, resample=5),
    "dps1-edm200": dict(steps=200, schedule="edm", replace=False, guidance=1.0),
    "dps2-edm200": dict(steps=200, schedule="edm", replace=False, guidance=2.0),
    "dps5-edm200": dict(steps=200, schedule="edm", replace=False, guidance=5.0),
    "dps2-edm200-r5": dict(steps=200, schedule="edm", replace=False, guidance=2.0, resample=5),
    "replace-dps0.5-edm200-r5": dict(steps=200, schedule="edm", guidance=0.5, resample=5),
    "replace-edm400-r5": dict(steps=400, schedule="edm", resample=5),
    "replace-edm200-r10": dict(steps=200, schedule="edm", resample=10),
    # two points of the Optuna study rsgm_inpaint_mean_apd_residual (the lowest mean, and the lowest mean with APD
    # above D-Flow's after ~60 trials), to see them per occlusion next to the baselines (baselines_valid.py)
    "residual-trial59": dict(steps=400, schedule="edm", replace=False, guidance=0.05457452885996979, resample=5),
    "residual-trial16": dict(steps=200, schedule="edm", replace=False, guidance=0.06107655015655633, resample=5),
    # the test point of abl residual (trial 81, "RSGM abl residual: their c10, inpaint t81" of evaluate_completion.py)
    "residual-t81": dict(steps=400, schedule="edm", replace=False, guidance=0.04934967328387599, resample=5),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("poses", type=int, help="the first n of the 1000 validation poses")
    parser.add_argument("variants", nargs="*", help=f"names of VARIANTS (default: all): {list(VARIANTS)}")
    parser.add_argument("--model", nargs="+", default=["small std c10"], help="names of dflow_valid.MODELS")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=500, help="poses per batch (times K in one GPU batch)")
    parser.add_argument("--out", default=None, help="a .jsonl file the rows are appended to")
    args = parser.parse_args()
    names = args.variants or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS] + [m for m in args.model if m not in MODELS]
    if unknown:
        parser.error(f"unknown {unknown}; variants {list(VARIANTS)}, models {list(MODELS)}")
    out = Path(args.out).expanduser() if args.out else None
    done = set()
    if out and out.exists():
        done = {(r["model"], r["variant"], r["poses"]) for r in map(json.loads, out.read_text().splitlines()) if r}
    gt, body = valid_poses(ROT, 1000)[:args.poses], Body()
    for model_name in args.model:
        model = build(MODELS[model_name], args.device)
        for name in names:
            if (model_name, name, args.poses) in done:
                print(f"{model_name} / {name}: already in {out}, skipped", flush=True)
                continue
            start, cells, means, bests = time.time(), {}, [], []
            for occ in OCCLUSIONS:
                torch.manual_seed(10)
                pred = run(model, gt, occ, 10, batch=args.batch, together=True, progress=False, method="inpaint",
                           prior=VARIANTS[name])
                s = summary(metrics(pred, gt, occ, body))
                cells[occ] = {key: float(s[key]) for key in ("mpvpe_mean", "apd", "mpvpe_min")}
                means.append(s["mpvpe_mean"])
                bests.append(s["mpvpe_min"])
            seconds = time.time() - start
            row = " | ".join(f"{c['mpvpe_mean']:5.1f}/{c['apd']:5.2f}/{c['mpvpe_min']:5.1f}" for c in cells.values())
            print(f"{model_name} | {name:22s} | {row} | mean {np.mean(means):.2f} best {np.mean(bests):.2f} | "
                  f"{seconds:.0f}s", flush=True)
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("a") as f:
                    f.write(json.dumps(dict(model=model_name, spec=MODELS[model_name], variant=name,
                                            options=VARIANTS[name], poses=args.poses, occlusions=cells,
                                            mean=float(np.mean(means)), best=float(np.mean(bests)),
                                            seconds=seconds, device=args.device)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
