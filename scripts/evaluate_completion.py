"""Pose completion as in PoseD-Flow (Tab. 2), for explore/evaluation/pose_completion.ipynb: computes the completions
of every model and occlusion and saves them in the cache the notebook reads (hp/evaluation/completion.py), e.g. on a
server:

    python scripts/evaluate_completion.py --device cuda:0
    python scripts/evaluate_completion.py --models "PoseD-Flow: their checkpoint" --occlusions left_leg trunk
    python scripts/evaluate_completion.py --poses 20 --device mps     # a quick check, cached apart

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

from hp.evaluation.completion import OCCLUSIONS, cached_run, test_poses  # noqa: E402

ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"  # their test set is complete there
CACHE = "~/data/experiments/completion/cache"
MODELS = {
    "PoseD-Flow: their checkpoint": dict(posedflow="~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt"),
    "RSGM 512x4: their c10": dict(run="~/data/experiments/rsgm_posedflow_small_s10/run_0001"),
    "RSGM 1536x6: their c10": dict(run="~/data/experiments/rsgm_posedflow_s10/run_0001"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--occlusions", nargs="+", default=list(OCCLUSIONS), choices=list(OCCLUSIONS))
    parser.add_argument("--poses", type=int, default=None, help="the first n test poses (default: their 5000)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rotations", default=ROTATIONS)
    parser.add_argument("--cache", default=CACHE)
    args = parser.parse_args()

    gt = test_poses(args.rotations, args.poses)
    print(f"{len(gt)} test poses, {len(args.models)} model(s) x {len(args.occlusions)} occlusion(s)", flush=True)
    for name in args.models:
        for occlusion in args.occlusions:
            start = time.perf_counter()
            cached_run(Path(args.cache), name, MODELS[name], gt, occlusion, args.device, rotations=args.rotations)
            print(f"{name} / {occlusion}: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
