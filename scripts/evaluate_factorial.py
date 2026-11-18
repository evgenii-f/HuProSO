"""All responses of the runs of the factorial study (section 13 of docs/experiments/2026-11-14_score_network_architectures.md),
on validation poses only, one JSON line per run:

    python scripts/evaluate_factorial.py --root ~/data/experiments/factorial --device cuda:0 \\
        --out ~/data/experiments/factorial/responses.jsonl
    python scripts/evaluate_factorial.py --run ~/data/experiments/factorial/rsgm_posedflow_fact_NA_s10 --parts inpaint

Parts (--parts, default all):
  generation  the PoseD-Flow protocol on validation: 20 sets of 500 samples, FID against the joint statistics of the
              validation poses (FID_val) and of the training poses (FID own), APD, dNN against the training poses; and
              the harness (hp.evaluation.evaluate) against the validation poses: Wasserstein, recall, precision and the
              log-likelihood of 1000 validation poses
  inpaint     inpainting at the fixed point INPAINT on the 1000 validation poses of the completion studies, K = 10,
              the four occlusions: MPVPE mean, best of K and APD per occlusion and their means
  hybrid      explore/evaluation/completion_diagnostics/hybrid.py (500 validation poses) and trajectory.py (300), as
              eval_runs.sh, in subprocesses with HP_DEVICE

--smoke: tiny sizes, to test the script (do not mix its rows with real ones). A run already in --out with the same
part is skipped, so an interrupted evaluation goes on. Weights: weights_ema.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the code of this checkout, before an editable install of another one

from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run as complete, summary, valid_poses  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.evaluation.evaluate import EvalConfig, evaluate  # noqa: E402
from hp.evaluation.posedflow_protocol import SmplSkeleton, apd, confidence, fid, nearest_neighbour_distance  # noqa: E402
from hp.models.io import load_model  # noqa: E402

ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"
SKELETON = "~/data/body_models/smpl/smpl_neutral_skeleton.npz"
SETS, SET_SIZE = 20, 500
HARNESS = EvalConfig(n_likelihood=1000)
N_CONTRAST = 100_000
# the rule point of the isotropic abl_norm inpainting study (trial 118 of rsgm_inpaint_mean_apd_norm)
INPAINT = dict(steps=200, schedule="edm", guidance=0.09515197497012039, resample=5, replace=False)
PARTS = ("generation", "inpaint", "hybrid")
DIAG = ROOT / "explore" / "evaluation" / "completion_diagnostics"
CELL = re.compile(r"rsgm_posedflow_(fact_[A-Z0-9]+)_s(\d+)")
GROUP = r"(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)"


def run_folder(path: Path) -> Path:
    return path if (path / "model.json").is_file() else sorted(path.glob("run_[0-9][0-9][0-9][0-9]"))[-1]


def generation(model, valid, train, train_dev, skeleton, stats) -> dict:
    out = {"FID_val": [], "FID_own": [], "APD": [], "dNN": []}
    torch.manual_seed(0)
    for _ in range(SETS):
        poses = model.sample(SET_SIZE).float().cpu()
        joints = skeleton.joints(poses)
        out["FID_val"].append(fid(joints, *stats["valid"]))
        out["FID_own"].append(fid(joints, *stats["train"]))
        out["APD"].append(apd(joints))
        out["dNN"].append(nearest_neighbour_distance(poses, train_dev).mean().item())
    result = {name: confidence(values) for name, values in out.items()}
    contrast = train[torch.from_numpy(np.random.default_rng(0).choice(len(train), N_CONTRAST, replace=False))]
    report = evaluate(model, valid, HARNESS, contrast=contrast)
    ll = report["likelihood"]
    result["harness"] = {"wasserstein_deg": report["wasserstein"]["generated"]["joint_mean_deg"]["mean"],
                         "recall_deg": report["recall"]["mean_deg"]["mean"],
                         "precision_deg": report["precision"]["mean_deg"]["mean"],
                         "log_likelihood": ll["mean"] if ll else None, "report": report}
    return result


def inpaint(run: Path, device: str, gt, body) -> dict:
    model = build(dict(run=str(run)), device)
    cells = {}
    for occ in OCCLUSIONS:
        torch.manual_seed(10)
        pred = complete(model, gt, occ, 10, batch=500, together=True, progress=False, method="inpaint", prior=INPAINT)
        s = summary(metrics(pred, gt, occ, body))
        cells[occ] = {key: float(s[key]) for key in ("mpvpe_mean", "apd", "mpvpe_min")}
    return {"occlusions": cells, **{name: float(np.mean([c[key] for c in cells.values()]))
                                    for name, key in (("mean", "mpvpe_mean"), ("best", "mpvpe_min"), ("apd", "apd"))}}


class PartFailed(RuntimeError):
    pass


def _call(command, env) -> str:
    """The stdout of a diagnostic script; its stderr in the error if it fails."""
    done = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True)
    if done.returncode:
        raise PartFailed(f"{Path(command[1]).name} exited with {done.returncode}:\n" + "\n".join(done.stderr.splitlines()[-15:]))
    return done.stdout


def hybrid(run: Path, device: str, poses: int = 500) -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HP_DEVICE": device, "RS": str(run)}
    out = _call([sys.executable, str(DIAG / "hybrid.py"), str(poses), "rsgm"], env)
    line = next(l for l in out.splitlines() if l.startswith("rsgm"))
    groups = {occ: dict(zip(("mpvpe_mean", "apd", "mpvpe_min"), map(float, m)))
              for occ, *m in re.findall(r"(\w+) " + GROUP, line)}
    out = _call([sys.executable, str(DIAG / "trajectory.py"), str(run)], env)
    err = [float(l.split()[7]) for l in out.splitlines() if re.match(r"^ +(100|199) ", l)]  # as eval_runs.sh
    names = ("left_leg_100", "left_leg_199", "arms_100", "arms_199")
    return {"hybrid": groups, "trajectory": dict(zip(names, err)), "raw": line}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="a folder with rsgm_posedflow_fact_*_s* runs")
    parser.add_argument("--run", action="append", default=[], help="a run (folder of a config or its run_000N)")
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True, help="the .jsonl file the rows are appended to")
    parser.add_argument("--skeleton", default=SKELETON)
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, to test the script")
    args = parser.parse_args()
    global SETS, SET_SIZE, HARNESS, N_CONTRAST
    n_inpaint, n_hybrid = 1000, 500
    if args.smoke:
        SETS, SET_SIZE, N_CONTRAST, n_inpaint, n_hybrid = 2, 100, 2_000, 20, 20
        HARNESS = EvalConfig(n_wasserstein=100, n_queries=100, n_generated=500, n_likelihood=20, repeats=2)
    runs = [Path(r).expanduser() for r in args.run]
    if args.root:
        runs += sorted(p for p in Path(args.root).expanduser().glob("rsgm_posedflow_fact_*_s*") if p.is_dir())
    runs = [run_folder(r) for r in runs]
    if not runs:
        parser.error("no runs: give --root or --run")
    out = Path(args.out).expanduser()
    done = set()
    if out.exists():
        done = {(r["run"], r["part"]) for r in map(json.loads, out.read_text().splitlines()) if r}

    todo = [(r, p) for r in runs for p in args.parts if (str(r), p) not in done]
    print(f"{len(runs)} runs, {len(todo)} parts to do, device {args.device}", flush=True)
    if any(p == "generation" for _, p in todo):
        valid = load_reference("valid", ROTATIONS, "posedflow")
        train = load_reference("train", ROTATIONS, "posedflow")
        skeleton = SmplSkeleton(args.skeleton)
        stats = {}
        for name, poses in (("valid", valid), ("train", train)):
            j = skeleton.joints(poses).reshape(len(poses), -1).numpy().astype(np.float64)
            stats[name] = (j.mean(0), np.cov(j, rowvar=False))
        train_dev = train.to(args.device)
    if any(p == "inpaint" for _, p in todo):
        gt, body = valid_poses(ROTATIONS, 1000)[:n_inpaint], Body()

    for run, part in todo:
        match = CELL.search(str(run))
        start = time.perf_counter()
        if part == "generation":
            model = load_model(run, device=args.device, weights="weights_ema").eval()
            result = generation(model, valid, train, train_dev, skeleton, stats)
            brief = ", ".join(f"{k} {result[k]['mean']:.4f}" for k in ("FID_val", "FID_own", "APD", "dNN"))
            h = result["harness"]
            brief += (f", W {h['wasserstein_deg']:.2f}, recall {h['recall_deg']:.2f}, precision {h['precision_deg']:.2f}, "
                      f"ll {h['log_likelihood'] if h['log_likelihood'] is not None else float('nan'):.2f}")
        elif part == "inpaint":
            result = inpaint(run, args.device, gt, body)
            brief = f"mean {result['mean']:.2f}, best {result['best']:.2f}, APD {result['apd']:.2f}"
        else:
            try:
                result = hybrid(run, args.device, n_hybrid)
            except PartFailed as e:  # no row: a later call tries it again; the other runs go on
                print(f"{run.parent.name} hybrid FAILED, skipped:\n{e}", flush=True)
                continue
            brief = result["raw"] + " | traj " + " ".join(f"{v:.2f}" for v in result["trajectory"].values())
        seconds = time.perf_counter() - start
        print(f"{run.parent.name} {part} ({seconds / 60:.1f} min): {brief}", flush=True)
        row = dict(run=str(run), cell=match.group(1) if match else None, seed=int(match.group(2)) if match else None,
                   part=part, result=result, seconds=seconds, device=args.device, smoke=args.smoke,
                   settings=dict(inpaint=INPAINT, sets=SETS, set_size=SET_SIZE, harness=HARNESS.__dict__,
                                 rotations=ROTATIONS))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
