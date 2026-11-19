"""PoseD-Flow (their checkpoint, D-Flow as run() by default) and DPoser (trained on their full data, with its own
optimization) on the first N of the 1000 validation poses of the Optuna studies, K=10, all four occlusions: the
baselines next to inpaint_valid.py and dflow_valid.py on the same poses and seeds. Run from the root:

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/baselines_valid.py 100 \\
        --out ~/data/experiments/completion/baselines_valid_mac.jsonl

--out appends one JSON line per row (per occlusion MPVPE mean, APD, best of K, the means, seconds); a row already in
the file with the same poses is skipped.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run, summary, valid_poses

ROT = "~/data/AMASS_rotations_posedflow_c10"
E = "~/data/experiments/"
ROWS = {  # name: (model spec of build(), method of run())
    "PoseD-Flow (their checkpoint, D-Flow)": (
        dict(posedflow="~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt"), "dflow"),
    "DPoser full (DPoser method)": (dict(run=E + "full/dposer_posedflow_s10/run_0001"), "dposer"),
    "DPoser full (D-Flow)": (dict(run=E + "full/dposer_posedflow_s10/run_0001"), "dflow"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("poses", type=int, help="the first n of the 1000 validation poses")
    parser.add_argument("rows", nargs="*", help=f"names of ROWS (default: all): {list(ROWS)}")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default=None, help="a .jsonl file the rows are appended to")
    args = parser.parse_args()
    names = args.rows or list(ROWS)
    out = Path(args.out).expanduser() if args.out else None
    done = set()
    if out and out.exists():
        done = {(r["model"], r["poses"]) for r in map(json.loads, out.read_text().splitlines()) if r}
    gt, body = valid_poses(ROT, 1000)[:args.poses], Body()
    for name in names:
        if (name, args.poses) in done:
            print(f"{name}: already in {out}, skipped", flush=True)
            continue
        spec, method = ROWS[name]
        model = build(spec, args.device)
        start, cells = time.time(), {}
        for occ in OCCLUSIONS:
            torch.manual_seed(10)
            pred = run(model, gt, occ, 10, batch=500, together=True, progress=False, method=method)
            s = summary(metrics(pred, gt, occ, body))
            cells[occ] = {key: float(s[key]) for key in ("mpvpe_mean", "apd", "mpvpe_min")}
        seconds = time.time() - start
        mean = float(np.mean([c["mpvpe_mean"] for c in cells.values()]))
        best = float(np.mean([c["mpvpe_min"] for c in cells.values()]))
        print(f"{name}: mean {mean:.2f} best {best:.2f} | {seconds:.0f}s", flush=True)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a") as f:
                f.write(json.dumps(dict(model=name, method=method, options=None, poses=args.poses, occlusions=cells,
                                        mean=mean, best=best, seconds=seconds, device=args.device)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
