"""Pose completion as in PoseD-Flow (Tab. 2), for explore/evaluation/pose_completion.ipynb: computes the completions
of every model and occlusion and saves them in the cache the notebook reads (hp/evaluation/completion.py), e.g. on a
server:

    python scripts/evaluate_completion.py --device cuda:0
    python scripts/evaluate_completion.py --models "PoseD-Flow: their checkpoint" --occlusions left_leg trunk
    python scripts/evaluate_completion.py --poses 20 --device mps     # a quick check, cached apart
    python scripts/evaluate_completion.py --device cuda:0 --together --batch 250   # faster: 2500 rows per GPU call

--together runs the 10 hypotheses of --batch poses as one batch: fewer, larger GPU calls, other random draws than
theirs, the same method. The memory grows with --batch * 10 * the ODE steps (the gradients go through all of them).
Results of either way are reused by the other.

Every model is completed by D-Flow (hp/models/dflow.py) with their settings: 10 hypotheses, 200 optimizer steps (300
for the arms), lr 0.1, 100 Euler steps of the ODE, running cost 1e-3. A finished (model, occlusion) is skipped. MODELS
must stay as in the notebook; paths with ~ are resolved on every machine.
"""
import argparse
import sys
import time
from pathlib import Path

# the code of this checkout, before an editable install of another one
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hp.evaluation.completion import BATCH, OCCLUSIONS, cached_run, test_poses  # noqa: E402

ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"  # their test set is complete there
CACHE = "~/data/experiments/completion/cache"
MODELS = {
    "PoseD-Flow: their checkpoint": dict(posedflow="~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt"),
    "RSGM 512x4: their c10": dict(run="~/data/experiments/rsgm_posedflow_small_s10/run_0001"),
    "RSGM 1536x6: their c10": dict(run="~/data/experiments/rsgm_posedflow_s10/run_0001"),
    "RSGM 512x4: their c10, seed 11": dict(run="~/data/experiments/rsgm_posedflow_small_s11/run_0001"),
    "RSGM 512x4: their c10, seed 12": dict(run="~/data/experiments/rsgm_posedflow_small_s12/run_0001"),
    "RSGM 512x4: their full": dict(run="~/data/experiments/full/rsgm_posedflow_small_s10/run_0001"),
    # the same network with other ODEs inside D-Flow (the samplers of the Pareto comparison): half the network calls
    "RSGM 512x4: their c10, ODE Euler 50": dict(run="~/data/experiments/rsgm_posedflow_small_s10/run_0001",
                                                 prior=dict(ode_steps=50)),
    "RSGM 512x4: their c10, Heun EDM 50 NFE": dict(run="~/data/experiments/rsgm_posedflow_small_s10/run_0001",
                                                   prior=dict(sampler="heun", schedule="edm", ode_steps=25)),
    # DPoser (Euclidean diffusion on axis-angle) through the same D-Flow: separates the prior from their own completion
    "DPoser: their full": dict(run="~/data/experiments/full/dposer_posedflow_s10/run_0001"),
    "DPoser: their c10, RSGM budget": dict(run="~/data/experiments/dposer_posedflow_rsgm_budget_s10/run_0001"),
    # their start pushes the observation forward through the ODE; in R^n that leaves the typical set of N(0, I)
    "DPoser: their full, inverted start": dict(run="~/data/experiments/full/dposer_posedflow_s10/run_0001",
                                               prior=dict(invert=True)),
    "RSGM 512x4: their c10, inverted start": dict(run="~/data/experiments/rsgm_posedflow_small_s10/run_0001",
                                                  prior=dict(invert=True)),
    # DPoser's own completion (DPoser-X, the method of the DPoser rows of PoseD-Flow's table), not D-Flow
    "DPoser: their full, DPoser method": dict(run="~/data/experiments/full/dposer_posedflow_s10/run_0001",
                                              method="dposer"),
    "DPoser: their c10, RSGM budget, DPoser method": dict(
        run="~/data/experiments/dposer_posedflow_rsgm_budget_s10/run_0001", method="dposer"),
    # inpainting with the reverse process of RSGM (hp/models/inpainting.py): one point per Optuna study
    # (config/completion_tuning/rsgm_inpaint_*.yaml), chosen on 1000 validation poses by the rule of the studies (the
    # lowest mean among the points of the front with APD >= 6.40, that of D-Flow with Heun EDM 50)
    "RSGM 512x4: their c10, inpaint t52": dict(
        run="~/data/experiments/rsgm_posedflow_small_s10/run_0001", method="inpaint",
        prior=dict(steps=400, schedule="edm", guidance=0.057708678925692564, resample=5, replace=False)),
    "RSGM deep: their c10, inpaint t75": dict(
        run="~/data/experiments/rsgm_posedflow_deep_s10/run_0001", method="inpaint",
        prior=dict(steps=200, schedule="edm", guidance=0.09753705685072556, resample=5, replace=False)),
    "RSGM abl residual: their c10, inpaint t81": dict(
        run="~/data/experiments/rsgm_posedflow_abl_residual_s10/run_0001", method="inpaint",
        prior=dict(steps=400, schedule="edm", guidance=0.04934967328387599, resample=5, replace=False)),
    "RSGM abl norm: their c10, inpaint t118": dict(
        run="~/data/experiments/rsgm_posedflow_abl_norm_s10/run_0001", method="inpaint",
        prior=dict(steps=200, schedule="edm", guidance=0.09515197497012039, resample=5, replace=False)),
    # DPoser trained like RSGM c10, one point per study (config/completion_tuning/dposer_c10_*_mean_apd.yaml) by the
    # same rule; and, outside the rule, DPoser's method with their defaults and 3 x 200 steps (trial 3 of that study)
    "DPoser: their c10, RSGM budget, inpaint t68": dict(
        run="~/data/experiments/dposer_posedflow_rsgm_budget_s10/run_0001", method="inpaint",
        prior=dict(steps=200, schedule="edm", guidance=1.0888627538152378, resample=5, replace=True)),
    "DPoser: their c10, RSGM budget, DPoser method t28": dict(
        run="~/data/experiments/dposer_posedflow_rsgm_budget_s10/run_0001", method="dposer",
        prior=dict(lr=0.12276386856394805, t_max=0.17371331329113274, t_min=0.0397763049393363, iterations=3,
                   steps_per_iter=100)),
    "DPoser: their c10, RSGM budget, DPoser method t3": dict(
        run="~/data/experiments/dposer_posedflow_rsgm_budget_s10/run_0001", method="dposer",
        prior=dict(lr=0.1, t_max=0.15, t_min=0.049999999999999996, iterations=3, steps_per_iter=200)),
    # the models trained on the full data, one point per study (config/completion_tuning/*full*.yaml) by the same rule
    "RSGM dposer-like: their full, inpaint t108": dict(
        run="~/data/experiments/full/rsgm_posedflow_dposer_like_s10/run_0001", method="inpaint",
        prior=dict(steps=200, schedule="edm", guidance=0.07182167279436949, resample=5, replace=False)),
    "RSGM 512x4: their full, inpaint t16": dict(
        run="~/data/experiments/full/rsgm_posedflow_small_s10/run_0001", method="inpaint",
        prior=dict(steps=200, schedule="edm", guidance=0.06107655015655633, resample=5, replace=False)),
    "DPoser: their full, inpaint t90": dict(
        run="~/data/experiments/full/dposer_posedflow_s10/run_0001", method="inpaint",
        prior=dict(steps=100, schedule="edm", guidance=0.9806648250542185, resample=5, replace=True)),
    "DPoser: their full, DPoser method t28": dict(
        run="~/data/experiments/full/dposer_posedflow_s10/run_0001", method="dposer",
        prior=dict(lr=0.16364850392520697, t_max=0.1560343553378305, t_min=0.05502625394728783, iterations=3,
                   steps_per_iter=100)),
    # PoseD-Flow with D-Flow settings of its Optuna study (config/completion_tuning/posedflow_dflow_mean_apd.yaml),
    # trial 17: better than their protocol (trial 0) on validation in mean, best of 10 and APD; 200 steps for all
    "PoseD-Flow: their checkpoint, D-Flow t17": dict(
        posedflow="~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt",
        dflow=dict(lr=0.05, running_weight=0.001, steps=200)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--occlusions", nargs="+", default=list(OCCLUSIONS), choices=list(OCCLUSIONS))
    parser.add_argument("--poses", type=int, default=None, help="the first n test poses (default: their 5000)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rotations", default=ROTATIONS)
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--batch", type=int, default=BATCH, help=f"poses per batch (theirs: {BATCH})")
    parser.add_argument("--together", action="store_true", help="the hypotheses of a batch in one GPU batch")
    args = parser.parse_args()
    speed = {"batch": args.batch, "together": True} if args.together else ({"batch": args.batch} if args.batch != BATCH else {})

    gt = test_poses(args.rotations, args.poses)
    print(f"{len(gt)} test poses, {len(args.models)} model(s) x {len(args.occlusions)} occlusion(s)", flush=True)
    for name in args.models:
        for occlusion in args.occlusions:
            start = time.perf_counter()
            cached_run(Path(args.cache), name, MODELS[name], gt, occlusion, args.device, rotations=args.rotations, **speed)
            print(f"{name} / {occlusion}: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
