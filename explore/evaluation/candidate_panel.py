"""The cheap stages of the evaluation of a candidate score network (docs/mace_lite_plan.md, "Rule"), on validation
poses, one JSON line per run in --out (a run already there is skipped, so an interrupted panel goes on):

    PYTHONPATH=src python explore/evaluation/candidate_panel.py --run LABEL=RUN_FOLDER ... \\
        --out ~/data/experiments/candidate_panel.jsonl --device mps

  stage 0  training: the last validation loss of history.json, finished or not
  stage 1  completion: explore/evaluation/completion_diagnostics/hybrid.py 200 rsgm (DPoser's procedure with the
           RSGM denoiser, 200 validation poses, K = 10, four occlusions): mean, best of 10 and APD per occlusion and
           their means
  stage 2  generation: the harness (hp.evaluation.evaluate, the model's own sampler, EMA weights) against the
           validation poses of the PoseD-Flow split, contrast 100k training poses: Wasserstein, recall, precision
           (deg; mean and std over its repeats), no likelihood (the per-axis noise has none)

The stages 3 (phase diagnostics, quick protocol) are run by hand, for the candidates that pass. Short: seconds to
minutes per run; no stop and resume beyond the per-run lines.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hp.evaluation.data import load_reference  # noqa: E402
from hp.evaluation.evaluate import EvalConfig, evaluate  # noqa: E402
from hp.models.io import load_model  # noqa: E402

ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"  # its valid split and training poses, as evaluate_factorial.py
HARNESS = EvalConfig(n_likelihood=0)  # the defaults otherwise: 20k samples, 2000 queries, 3 repeats
N_CONTRAST = 100_000
POSES = 200
GROUP = r"(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)"
OCCLUSIONS = ("left_leg", "legs", "arms", "trunk")


def run_folder(path: str) -> Path:
    p = Path(path).expanduser()
    return p if (p / "model.json").is_file() else sorted(p.glob("run_[0-9][0-9][0-9][0-9]"))[-1]


def training(run: Path) -> dict:
    h = json.loads((run / "history.json").read_text())
    status = json.loads((run / "status.json").read_text()) if (run / "status.json").is_file() else {}
    return {"valid_loss": h["valid_loss"][-1], "finished": bool(status.get("finished"))}


def completion(run: Path, device: str) -> dict:
    env = {**os.environ, "RS": str(run), "PYTHONPATH": str(ROOT / "src"), "DEVICE": device, "HP_DEVICE": device}
    out = subprocess.run([sys.executable, str(ROOT / "explore/evaluation/completion_diagnostics/hybrid.py"),
                          str(POSES), "rsgm"], env=env, capture_output=True, text=True, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("rsgm"))
    cells = {}
    for occ in OCCLUSIONS:
        mean, apd, best = (float(v) for v in re.search(occ + r" " + GROUP, line).groups())
        cells[occ] = {"mean": mean, "apd": apd, "best": best}
    return {"occlusions": cells, **{k: float(np.mean([c[k] for c in cells.values()])) for k in ("mean", "apd", "best")}}


def generation(run: Path, device: str, valid, contrast) -> dict:
    model = load_model(run, device=device, weights="weights_ema")
    model.has_log_prob = False  # this instance only: the harness then skips the likelihood
    torch.manual_seed(0)
    report = evaluate(model, valid, HARNESS, contrast=contrast)
    out = {}
    for name, part in (("wasserstein", report["wasserstein"]["generated"]["joint_mean_deg"]),
                       ("recall", report["recall"]["mean_deg"]), ("precision", report["precision"]["mean_deg"])):
        out[name], out[name + "_std"] = part["mean"], part.get("std")
    for name, part in (("wasserstein", report["wasserstein"]["contrast"]["joint_mean_deg"]),
                       ("recall", report["contrast_recall"]["mean_deg"]),
                       ("precision", report["contrast_precision"]["mean_deg"])):
        out[name + "_contrast"] = part["mean"]  # real training poses of other subjects in place of the generator
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, metavar="LABEL=FOLDER")
    parser.add_argument("--out", default="~/data/experiments/candidate_panel.jsonl")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--stages", nargs="+", default=["0", "1", "2"])
    args = parser.parse_args()
    out = Path(args.out).expanduser()
    done = {json.loads(l)["label"] for l in out.read_text().splitlines()} if out.is_file() else set()
    valid = contrast = None
    for spec in args.run:
        label, path = spec.split("=", 1)
        if label in done:
            print(f"{label}: in {out} already, skipped", flush=True)
            continue
        run, row, t0 = run_folder(path), {"label": label}, time.perf_counter()
        row["run"] = str(run)
        if "0" in args.stages:
            row["training"] = training(run)
        if "1" in args.stages:
            t = time.perf_counter()
            row["completion"] = completion(run, args.device)
            row["completion"]["seconds"] = time.perf_counter() - t
        if "2" in args.stages:
            if valid is None:
                valid = load_reference("valid", ROTATIONS, "posedflow")
                train = load_reference("train", ROTATIONS, "posedflow")
                pick = np.random.default_rng(0).choice(len(train), N_CONTRAST, replace=False)
                contrast = train[torch.from_numpy(pick)]
            t = time.perf_counter()
            row["generation"] = generation(run, args.device, valid, contrast)
            row["generation"]["seconds"] = time.perf_counter() - t
        row["seconds"] = time.perf_counter() - t0
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        nan = float("nan")
        c, g, tr = row.get("completion", {}), row.get("generation", {}), row.get("training", {})
        print(f"{label}: valid loss {tr.get('valid_loss', nan):.4f} | completion {c.get('mean', nan):.1f} / "
              f"{c.get('apd', nan):.2f} / {c.get('best', nan):.1f} | W {g.get('wasserstein', nan):.2f} "
              f"R {g.get('recall', nan):.2f} P {g.get('precision', nan):.2f} | {row['seconds'] / 60:.1f} min",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
