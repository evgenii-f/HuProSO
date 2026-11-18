"""Motion denoising as in PoseD-Flow (Tab. 3, hp/evaluation/denoising.py): denoises the noisy joints of the validation
or the test sequences with every model and noise level and caches the predictions and per-frame metrics in
~/data/experiments/denoising/cache/<model>/<split>_<sigma>mm.pt. On a server (the file of the sequences,
~/data/experiments/denoising/sequences.pt, and smpl_mean_params.npz copied there from the Mac):

    python scripts/evaluate_denoising.py --split valid --device cuda:0
    python scripts/evaluate_denoising.py --split valid --models "w/o prior" --sigmas 0.04 --device cpu
    python scripts/evaluate_denoising.py --split valid --summary            # the table of what is cached
    python scripts/evaluate_denoising.py --split test --models "PoseD-Flow: their checkpoint" --device cuda:0

The test is run once per row, with settings chosen before on validation (--split test must be given explicitly).
D-Flow keeps the gradients of all ODE steps of all frames of --batch sequences: the memory grows with --batch.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hp.evaluation.denoising import PAPER, SIGMAS, cached_run, summary  # noqa: E402

CACHE = "~/data/experiments/denoising/cache"
# RSGM in DPoser's optimization, per noise level: the best trials of the Optuna studies on validation
# (config/denoising_tuning/rsgm_sds_<mm>mm.yaml, RSGM 512x4 c10, 40 trials each): 40 mm trial 36, 18.69 mm;
# 100 mm trial 29, 30.87 mm. t_min = t_min_fraction * t_max.
SDS = {0.04: dict(t_max=0.05949959532179853, t_min=0.05949959532179853 * 0.38708311854899335,
                  sds_weight=7.7486572605596375, lr=0.04133605348828209, steps_per_iter=67),
       0.1: dict(t_max=0.05147455761154559, t_min=0.05147455761154559 * 0.8807515453505762,
                 sds_weight=4.381518863971957, lr=0.042200143285960834, steps_per_iter=133)}
# the same at DPoser's number of steps (3 x 60 at 40 mm, 3 x 80 at 100 mm), for a comparison at an equal budget
SDS_DPOSER_STEPS = {0.04: {**SDS[0.04], "steps_per_iter": 60}, 0.1: {**SDS[0.1], "steps_per_iter": 80}}
RUNS = {
    "RSGM 512x4: their c10": "~/data/experiments/rsgm_posedflow_small_s10/run_0001",
    "RSGM 1536x6: their c10": "~/data/experiments/rsgm_posedflow_s10/run_0001",
    "RSGM 512x4: their full": "~/data/experiments/full/rsgm_posedflow_small_s10/run_0001",
    "RSGM res_norm: their c10": "~/data/experiments/rsgm_posedflow_res_norm_s10/run_0001",
}
MODELS = {
    # DPoser's optimization without its prior term: the "w/o prior" row
    "w/o prior": dict(method="dposer"),
    "PoseD-Flow: their checkpoint": dict(posedflow="~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt"),
    # D-Flow suits the flow model, not the diffusion models (10 validation sequences at 40 mm: RSGM 24-28 mm, still
    # falling at 600 steps; DPoser via D-Flow 24.2): one declared row with their settings, not tuned
    "RSGM 512x4: their c10, D-Flow": dict(run=RUNS["RSGM 512x4: their c10"]),
    # RSGM in DPoser's optimization (score distillation on SO(3)^J), the procedure we tune: settings per noise level
    # chosen on validation with RSGM 512x4 c10 (config/denoising_tuning/rsgm_sds_<mm>mm.yaml) and used for every RSGM
    **{f"{name}, DPoser method": dict(run=run, method="dposer", settings=SDS) for name, run in RUNS.items()},
    # the first guess of the settings (10 validation sequences at 40 mm), before the Optuna studies; its test was run
    # by mistake with the commit before the tuned settings (2026-11-17), and is reported as such
    **{f"{name}, DPoser method, first guess": dict(run=run, method="dposer",
                                                   settings={sigma: dict(t_max=0.05, t_min=0.01, sds_weight=10.0)
                                                             for sigma in (0.04, 0.1)})
       for name, run in RUNS.items()},
    "RSGM 512x4: their c10, DPoser method, DPoser's steps": dict(run=RUNS["RSGM 512x4: their c10"], method="dposer",
                                                                settings=SDS_DPOSER_STEPS),
    # DPoser retrained on their full data: its own optimization (the DPoser rows of their table) and D-Flow
    "DPoser: their full, DPoser method": dict(run="~/data/experiments/full/dposer_posedflow_s10/run_0001",
                                              method="dposer"),
    "DPoser: their full, D-Flow": dict(run="~/data/experiments/full/dposer_posedflow_s10/run_0001"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["valid", "test"])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS), choices=list(SIGMAS))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=10, help="sequences per batch (speed and memory only)")
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--summary", action="store_true", help="only print what is cached")
    args = parser.parse_args()

    print(f"{'model':48s} " + " ".join(f"{'sigma ' + str(round(s * 1000)) + ' mm':>24s}" for s in args.sigmas))
    for name in args.models:
        cells = []
        for sigma in args.sigmas:
            m = cached_run(args.cache, name, MODELS[name], args.split, sigma, None if args.summary else args.device,
                           args.batch)
            if m is None:
                cells.append(f"{'-':>24s}")
                continue
            s = summary(m)
            cells.append(f"{s['mpjpe']:7.2f} / {s['mpvpe']:6.2f} ({s['input_mpjpe']:5.1f})")
        print(f"{name:48s} " + " ".join(cells), flush=True)
    print("cells: MPJPE / MPVPE (MPJPE of the input), mm; "
          f"paper AMASS 40 / 100 mm: " + ", ".join(f"{k} {v[0]} / {v[1]}" for k, v in PAPER.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
