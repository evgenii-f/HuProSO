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
# RSGM in DPoser's optimization, per noise level: a first guess from 10 validation sequences at 40 mm, replaced by
# the settings chosen on validation (config/denoising_tuning/rsgm_sds_<mm>mm.yaml) before the test
SDS = {0.04: dict(t_max=0.05, t_min=0.01, sds_weight=10.0),
       0.1: dict(t_max=0.05, t_min=0.01, sds_weight=10.0)}
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
