"""Evaluate runs of the zoo for explore/evaluation/posedflow_vs_rsgm.ipynb on another machine (e.g. a GPU server), and
write the results into the notebook's cache: copy the cache folder back and the notebook reuses them.

    python scripts/evaluate_posedflow_comparison.py --run ~/data/experiments/rsgm_posedflow_large_s10 --device cuda:0
    python scripts/evaluate_posedflow_comparison.py \
        --run "RSGM 4096x7, their data=~/data/experiments/rsgm_posedflow_large_s10" \
        --run "RSGM 4096x7, their data, best (8k)=~/data/experiments/rsgm_posedflow_large_s10/run_0001/checkpoint_008000"

Both stages of the notebook, with the same settings: our harness (Wasserstein, recall, precision, log-likelihood on
their test split, the training poses as yardstick) and the protocol of the PoseD-Flow paper (FID, FID own, APD, dNN
over 20 sets of 500 samples). Every result is saved as <cache>/<label>/<stage>.json together with what it depends on
(the run, the size and time of the weights file, the settings), exactly as the notebook writes it. The notebook
compares these settings with the home folder as ~ and file times in whole seconds, so results computed under
/home/<user> are found under /Users/<user> after `rsync -a` (which keeps the file times).

Needs on the machine: the data of the notebook (ROTATIONS, train and test), the SMPL skeleton and their FID
statistics. SKELETON and GT_STATS below are part of the settings and must stay as in the notebook; if the files lie
elsewhere on this machine, say where with --skeleton and --gt-stats (only read from there).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# the code of this checkout, before an editable install of another one
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hp.evaluation.data import load_reference  # noqa: E402
from hp.evaluation.evaluate import EvalConfig, evaluate  # noqa: E402
from hp.evaluation.posedflow_protocol import SmplSkeleton, apd, confidence, fid, nearest_neighbour_distance  # noqa: E402
from hp.models.io import MANIFEST, load_model  # noqa: E402

# ---- the settings of the notebook (keep them equal, or the notebook will not reuse the results) ----
ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"
SKELETON = "~/data/body_models/smpl/smpl_neutral_skeleton.npz"
GT_STATS = "~/Documents/PoseDFlow/examples/gt_mean_var_v2.npz"
HARNESS = EvalConfig()
N_CONTRAST = 100_000
SETS, SET_SIZE = 20, 500
CACHE = "~/data/experiments/posedflow_comparison/cache"
# ------------------------------------------------------------------------------------------------------------------


def run_folder(path):
    path = Path(path).expanduser()
    if (path / MANIFEST).is_file():
        return path
    runs = sorted(p for p in path.glob("run_[0-9][0-9][0-9][0-9]") if (p / "history.json").is_file() and (p / MANIFEST).is_file())
    return runs[-1] if runs else None


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def tag(path):
    p = Path(path).expanduser()
    return p.parent.name if re.fullmatch(r"run_\d{4}", p.name) else p.name


def identity_of(run: Path, weights: str, overrides: dict) -> dict:
    file = run / f"{weights}.pt"
    return {"kind": "run", "run": str(run), "weights": weights, "size": file.stat().st_size,
            "mtime": file.stat().st_mtime, "overrides": overrides}


def chunked_log_prob(model, chunk: int):
    """model.log_prob in pieces of `chunk` poses: the exact divergence of a wide network needs memory per pose.
    The result is the same; the harness's sample_batch (part of the settings) stays as it is."""
    full = model.log_prob

    def log_prob(pose, cond=None):
        return torch.cat([full(pose[i:i + chunk], cond) for i in range(0, len(pose), chunk)])
    model.log_prob = log_prob
    return model


def save(cache: Path, label: str, stage: str, settings: dict, result) -> Path:
    path = cache / slug(label) / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(json.dumps(settings))
    path.write_text(json.dumps({"name": label, "stage": stage, "settings": settings, "result": result}, indent=1))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True,
                        help="a run folder, folder of runs or checkpoint folder, optionally as 'label=path' (the label "
                             "names the cache folder; the notebook finds the result by its settings anyway)")
    parser.add_argument("--weights", default="weights_ema")
    parser.add_argument("--sample-steps", type=int, default=None, help="as sample_steps= in MODELS (default: as trained)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stages", nargs="+", choices=("harness", "paper"), default=["harness", "paper"])
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--skeleton", default=SKELETON, help="where the SMPL skeleton is on this machine")
    parser.add_argument("--gt-stats", default=GT_STATS, help="where their FID statistics are on this machine")
    parser.add_argument("--ll-chunk", type=int, default=250, help="poses per log_prob call (memory)")
    parser.add_argument("--smoke", action="store_true", help="tiny settings, to test the script (other cache entries)")
    args = parser.parse_args()

    harness_cfg, n_contrast, sets, set_size = HARNESS, N_CONTRAST, SETS, SET_SIZE
    if args.smoke:
        harness_cfg = EvalConfig(n_wasserstein=100, n_queries=100, n_generated=500, n_likelihood=20, repeats=2)
        n_contrast, sets, set_size = 2_000, 2, 100
    cache = Path(args.cache).expanduser()
    overrides = {} if args.sample_steps is None else {"sample_steps": args.sample_steps}

    models = []
    for item in args.run:
        label, path = item.split("=", 1) if "=" in item else (None, item)
        run = run_folder(path)
        if run is None:
            sys.exit(f"no finished run in {path}")
        models.append((label or tag(path), run))

    test = load_reference("test", ROTATIONS, "posedflow")
    train = load_reference("train", ROTATIONS, "posedflow")
    print(f"test {len(test):,}   train {len(train):,}", flush=True)

    harness_settings = {"config": harness_cfg.__dict__, "rotations": ROTATIONS, "test": len(test), "contrast": n_contrast}
    paper_settings = {"sets": sets, "size": set_size, "rotations": ROTATIONS, "train": len(train), "skeleton": SKELETON,
                      "gt_stats": GT_STATS}
    if "harness" in args.stages:
        contrast = train[torch.from_numpy(np.random.default_rng(0).choice(len(train), n_contrast, replace=False))]
    if "paper" in args.stages:
        skeleton = SmplSkeleton(args.skeleton)
        gt = np.load(Path(args.gt_stats).expanduser())
        muj, covj = gt["muj"].astype(np.float64), gt["covj"].astype(np.float64)
        own = skeleton.joints(train).reshape(len(train), -1).numpy()
        mu_own, cov_own = own.mean(0), np.cov(own, rowvar=False)
        del own
        train_dev = train.to(args.device)

    def build(run):
        model = load_model(run, device=args.device, weights=args.weights)
        for k, v in overrides.items():
            setattr(model, k, v)
        return chunked_log_prob(model, args.ll_chunk)

    for label, run in models:
        identity = identity_of(run, args.weights, overrides)
        print(f"=== {label}: {run}", flush=True)
        if "harness" in args.stages:
            start = time.perf_counter()
            report = evaluate(build(run), test, harness_cfg, contrast=contrast, progress=True)
            path = save(cache, label, "harness", {"model": identity, **harness_settings}, report)
            ll = report["likelihood"]
            print(f"  harness ({(time.perf_counter() - start) / 60:.1f} min): Wasserstein "
                  f"{report['wasserstein']['generated']['joint_mean_deg']['mean']:.2f}, recall {report['recall']['mean_deg']['mean']:.2f}, "
                  f"precision {report['precision']['mean_deg']['mean']:.2f}, log-lik. {ll['mean'] if ll else float('nan'):.2f} -> {path}", flush=True)
        if "paper" in args.stages:
            start = time.perf_counter()
            model = build(run).eval()
            torch.manual_seed(0)
            out = {"FID": [], "FID own": [], "APD": [], "dNN": []}
            for _ in range(sets):
                poses = model.sample(set_size).float().cpu()
                joints = skeleton.joints(poses)
                out["FID"].append(fid(joints, muj, covj))
                out["FID own"].append(fid(joints, mu_own, cov_own))
                out["APD"].append(apd(joints))
                out["dNN"].append(nearest_neighbour_distance(poses, train_dev).mean().item())
            result = {name: confidence(values) for name, values in out.items()}
            path = save(cache, label, "paper", {"model": identity, **paper_settings}, result)
            print(f"  paper ({(time.perf_counter() - start) / 60:.1f} min): " +
                  ", ".join(f"{k} {v['mean']:.4f}" for k, v in result.items()) + f" -> {path}", flush=True)


if __name__ == "__main__":
    main()
