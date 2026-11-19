"""The test of the points of the completion fronts (config/completion_test_points.yaml, scripts/select_test_points.py):
per front the points with their validation and test numbers, and the hypervolume of the points on both splits (the
measure of the rule: hp.evaluation.fronts.hypervolume, APD 4.0-8.5 cm, against mean 115 mm and best of 10 65 mm).
Computes the metrics of the test rows that have all four occlusions in the cache (cached_metrics, the body model on the
Mac); a front is rated only when all its points are there.

The cache ties a row to the weights it was made with by their size and modification time; a copy between machines
changes the time. A row whose spec and size of the weights match but whose time does not is accepted when the
sha256 of the weights here equals that checked on the servers (VERIFIED; the first 16 hex digits, the same on
workstation1, workstation2 and the Mac on 2026-11-19). Run from the root:

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/test_points_report.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_completion import CACHE, ROTATIONS  # noqa: E402

import torch  # noqa: E402

from hp.evaluation.completion import (OCCLUSIONS, Body, cache_path, cached_metrics, cached_run, identity,  # noqa: E402
                                      summary, test_poses)

VERIFIED = {  # run folder: sha256 of weights_ema.pt (first 16 hex digits), the same on every machine (2026-11-19)
    "~/data/experiments/rsgm_posedflow_abl_norm_s10/run_0001": "12d3375227740e78",
    "~/data/experiments/rsgm_posedflow_abl_residual_s10/run_0001": "d8ca0900c8cdd305",
    "~/data/experiments/rsgm_posedflow_deep_s10/run_0001": "3b8aeadf615b348e",
    "~/data/experiments/full/rsgm_posedflow_dposer_like_aniso_s10/run_0001": "3b9af16007959df7",
    "~/data/experiments/full/rsgm_posedflow_dposer_like_s10/run_0001": "4246190f5c5fd71f",
}


def sha16(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def completions(cache: Path, label: str, spec: dict, gt, occlusion: str):
    """The completions of a row: by cached_run if made with these weights as they are here; else, if only the time of
    the weights differs and their sha256 is VERIFIED, read from the cache file; None otherwise."""
    pred = cached_run(cache, label, spec, gt, occlusion, None, rotations=ROTATIONS)
    if pred is not None:
        return pred
    saved = torch.load(cache_path(cache, label, occlusion), weights_only=True)
    made, here = json.loads(saved["settings"])["model"], identity(spec)
    run = spec.get("run")
    if (made["spec"] == here["spec"] and made["size"] == here["size"] and run in VERIFIED
            and sha16(Path(run).expanduser() / "weights_ema.pt") == VERIFIED[run]
            and json.loads(saved["settings"])["poses"] == len(gt)):
        return saved["pred"].float()
    return None
from hp.evaluation.fronts import hypervolume  # noqa: E402


def test_numbers(cache: Path, label: str, spec: dict, gt, body) -> dict:
    """mean, best_of_k, apd of a test row averaged over the occlusions, or None if an occlusion is missing."""
    cells = []
    for occlusion in OCCLUSIONS:
        if not cache_path(cache, label, occlusion).is_file():
            return None
        pred = completions(cache, label, spec, gt, occlusion)
        if pred is None:
            return None
        cells.append(summary(cached_metrics(cache, label, occlusion, pred, gt, body)[0]))
    mean = lambda key: sum(c[key] for c in cells) / len(cells)
    return dict(mean=mean("mpvpe_mean"), best_of_k=mean("mpvpe_min"), apd=mean("apd"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", default=str(ROOT / "config" / "completion_test_points.yaml"))
    parser.add_argument("--cache", default=CACHE)
    args = parser.parse_args()
    cache = Path(args.cache).expanduser()
    fronts = yaml.safe_load(Path(args.points).read_text())
    gt, body = test_poses(ROTATIONS), Body()
    rated = {}
    for name, points in fronts.items():
        print(f"\n{name}")
        print("| APD >= | point | val mean | val best10 | val APD | test mean | test best10 | test APD |")
        print("|---|---|---|---|---|---|---|---|")
        rows = []
        for p in points:
            t = test_numbers(cache, p["label"], p["spec"], gt, body)
            v = p["validation"]
            rows.append((v, t))
            cells = " | ".join(f"{t[k]:.2f}" for k in ("mean", "best_of_k", "apd")) if t else "not yet | | "
            print(f"| {', '.join(f'{a:g}' for a in p['thresholds'])} | {p['study']} t{p['trial']} | {v['mean']:.2f} | "
                  f"{v['best_of_k']:.2f} | {v['apd']:.2f} | {cells} |", flush=True)
        if all(t for _, t in rows):
            rated[name] = {split: {metric: hypervolume([r[i] for r in rows], metric) for metric in ("mean", "best_of_k")}
                           for i, split in ((0, "validation"), (1, "test"))}
    print("\nhypervolume of the points (mm x cm, APD 4.0-8.5; mean against 115 mm, best of 10 against 65 mm); "
          "band mean = ref - HV / 4.5")
    print("| front | val HV mean | test HV mean | val band mean | test band mean | val HV best10 | test HV best10 |")
    print("|---|---|---|---|---|---|---|")
    for name, hv in sorted(rated.items(), key=lambda kv: -kv[1]["test"]["mean"]):
        v, t = hv["validation"], hv["test"]
        print(f"| {name} | {v['mean']:.2f} | {t['mean']:.2f} | {115 - v['mean'] / 4.5:.2f} | {115 - t['mean'] / 4.5:.2f} "
              f"| {v['best_of_k']:.2f} | {t['best_of_k']:.2f} |")
    missing = sorted(set(fronts) - set(rated))
    if missing:
        print(f"\nnot rated yet (points missing on the test): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
