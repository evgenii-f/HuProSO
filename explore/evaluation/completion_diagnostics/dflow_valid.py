"""D-Flow (run(), PoseD-Flow's lr and running weight) on the first N of the 1000 validation poses of the Optuna
studies, K=10, all four occlusions. Run from the root of the repository:

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/dflow_valid.py 300 "small std c10" "small lik c10"
    PYTHONPATH=src python explore/evaluation/completion_diagnostics/dflow_valid.py 1000 --ode heun-edm50 \\
        --device cuda:0 --out ~/data/experiments/completion/dflow_valid_arch.jsonl "small std c10" "abl norm"

--ode picks the ODE inside D-Flow: "euler100" (PoseD-Flow's protocol, the default) or "heun-edm50" (Heun on the EDM
times, 50 network calls: the best RSGM row of docs/experiments/2026-11-13_pose_completion_rsgm_vs_dposer.md). The
PoseD-Flow checkpoint always runs its own protocol. --out appends one JSON line per model (per occlusion MPVPE mean,
APD, best of K, and their means over the occlusions); a model already in the file with the same poses and ODE is
skipped, so an interrupted run continues. Memory heavy on MPS (backpropagation through the ODE): one process at a time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run, summary, valid_poses  # noqa: E402

ROT = "~/data/AMASS_rotations_posedflow_c10"
E = "~/data/experiments/"
ODES = {"euler100": None, "heun-edm50": dict(sampler="heun", schedule="edm", ode_steps=25)}
MODELS = {
    "posedflow": dict(posedflow="~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt"),
    # the MLPs of the report (512x4 = "small", 1536x6 = "search"), three training seeds of the small one
    "small std c10": dict(run=E + "rsgm_posedflow_small_s10/run_0001"),
    "small std c10 s11": dict(run=E + "rsgm_posedflow_small_s11/run_0001"),
    "small std c10 s12": dict(run=E + "rsgm_posedflow_small_s12/run_0001"),
    "small lik c10": dict(run=E + "rsgm_posedflow_small_lik_s10/run_0001"),
    "small lik c10 @10k": dict(run=E + "rsgm_posedflow_small_lik_s10/run_0001/checkpoint_010000"),
    "search std c10": dict(run=E + "rsgm_posedflow_s10/run_0001"),
    "search lik c10": dict(run=E + "rsgm_posedflow_search_lik_s10/run_0001"),
    "small std full": dict(run=E + "full/rsgm_posedflow_small_s10/run_0001"),
    "small lik full": dict(run=E + "full/rsgm_posedflow_small_lik_s10/run_0001"),
    "search lik full": dict(run=E + "full/rsgm_posedflow_search_lik_s10/run_0001"),
    # the networks of the architecture ablation (docs/experiments/2026-11-14_score_network_architectures.md), c10, std
    "abl base": dict(run=E + "rsgm_posedflow_abl_base_s10/run_0001"),
    "abl residual": dict(run=E + "rsgm_posedflow_abl_residual_s10/run_0001"),
    "abl norm": dict(run=E + "rsgm_posedflow_abl_norm_s10/run_0001"),
    "res norm": dict(run=E + "rsgm_posedflow_res_norm_s10/run_0001"),
    "timeres small": dict(run=E + "rsgm_posedflow_timeres_small_s10/run_0001"),
    "timeres": dict(run=E + "rsgm_posedflow_timeres_s10/run_0001"),
    "deep": dict(run=E + "rsgm_posedflow_deep_s10/run_0001"),
    "dposer like full": dict(run=E + "full/rsgm_posedflow_dposer_like_s10/run_0001"),
    # with the per-joint Brownian time (joint_scale, section 10 of the report), trained with the GPU noise
    "abl norm aniso": dict(run=E + "igso_speed/rsgm_posedflow_abl_norm_aniso_s10/run_0001"),
    "dposer like aniso full": dict(run=E + "full/rsgm_posedflow_dposer_like_aniso_s10/run_0001"),
    # DPoser itself (Euclidean diffusion on axis-angle), as in evaluate_completion.py
    "DPoser full": dict(run=E + "full/dposer_posedflow_s10/run_0001"),
    "DPoser c10 RSGM budget": dict(run=E + "dposer_posedflow_rsgm_budget_s10/run_0001"),
}
# stage A of the D-Flow comparison of the networks (the plan in the report of the ablation)
ARCH = ["posedflow", "small std c10", "small std c10 s11", "small std c10 s12", "abl base", "abl residual", "abl norm",
        "res norm", "timeres small", "timeres", "deep", "dposer like full"]


def done_in(path: Path) -> set:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r["model"], r["poses"], r["ode"]) for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("poses", type=int, help="the first n of the 1000 validation poses")
    parser.add_argument("models", nargs="*", help=f"names of MODELS, or 'arch' for {ARCH} (default: all)")
    parser.add_argument("--ode", choices=list(ODES), default="euler100")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=500, help="poses per batch (times K in one GPU batch)")
    parser.add_argument("--out", default=None, help="a .jsonl file the rows are appended to")
    args = parser.parse_args()
    names = ARCH if args.models == ["arch"] else (args.models or list(MODELS))
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        parser.error(f"unknown models {unknown}; known: {list(MODELS)}")
    out = Path(args.out).expanduser() if args.out else None
    done = done_in(out) if out else set()
    gt, body = valid_poses(ROT, 1000)[:args.poses], Body()
    for name in names:
        spec = MODELS[name]
        ode = "posedflow" if "posedflow" in spec else args.ode
        if (name, args.poses, ode) in done:
            print(f"{name}: already in {out}, skipped", flush=True)
            continue
        path = Path(spec.get("run", spec.get("posedflow"))).expanduser()
        if not path.exists():
            print(f"{name}: {path} missing, skipped", flush=True)
            continue
        model = build(spec, args.device)
        prior = None if "posedflow" in spec else ODES[args.ode]
        start, cells, means, bests = time.time(), {}, [], []
        for occ in OCCLUSIONS:
            torch.manual_seed(10)
            pred = run(model, gt, occ, 10, batch=args.batch, together=True, progress=False, prior=prior)
            s = summary(metrics(pred, gt, occ, body))
            cells[occ] = {key: float(s[key]) for key in ("mpvpe_mean", "apd", "mpvpe_min")}
            means.append(s["mpvpe_mean"])
            bests.append(s["mpvpe_min"])
        seconds = time.time() - start
        row = " | ".join(f"{c['mpvpe_mean']:5.1f}/{c['apd']:5.2f}/{c['mpvpe_min']:5.1f}" for c in cells.values())
        print(f"{ode} | {name:19s} | {row} | mean {np.mean(means):.2f} best {np.mean(bests):.2f} | {seconds:.0f}s",
              flush=True)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a") as f:
                f.write(json.dumps(dict(model=name, spec=spec, poses=args.poses, ode=ode, occlusions=cells,
                                        mean=float(np.mean(means)), best=float(np.mean(bests)), seconds=seconds,
                                        device=args.device)) + "\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
