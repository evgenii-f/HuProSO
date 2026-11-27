"""Holes and convergence of Gibbs inpainting with a Fisher energy: for every occlusion, the mean energy of a model's
own completions minus the energy of the truth, and the share of completions below the truth (gibbs.py --inpaint --out
files, 200 validation poses, K = 10). If the model is right and the chains have converged, a completion and the truth
are draws of the same conditional: the gap is near 0 and the share near 0.5. A large negative gap means the energy
prefers its own wrong completions (a hole); a positive one, chains not yet converged. With --against, the paired
difference of the mean MPVPE (mm, bootstrap over poses) of each run against the first. From the root:

    python explore/ace_energy/completion_energy.py \
        pl/energy_pl.pt:pl/inpaint_gibbs_200.pt pl/energy_bm.pt:pl/inpaint_bm_200.pt \
        pl/energy_bm_w123.pt:pl/inpaint_bm_w123_long_3000.pt

(paths relative to ~/data/experiments/ace/; each argument is fit:completions).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose import bootstrap  # noqa: E402
from gibbs import load  # noqa: E402
from pseudo_lik import ACE  # noqa: E402
from hp.evaluation.completion import KEEP, valid_poses  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="fit:completions, relative to ~/data/experiments/ace/")
    args = parser.parse_args()
    first = None
    print("| fit | completions | occlusion | MPVPE mean | E completion - E truth (nats) | share below truth "
          "| MPVPE - first run |")
    print("|---|---|---|---|---|---|---|")
    for run in args.runs:
        fit, comp = run.split(":")
        model = load(str(ACE / fit))
        occ = torch.load(ACE / comp, weights_only=True)["occlusions"]
        n = next(iter(occ.values()))["mpvpe"].shape[0]
        gt = valid_poses(n=1000)[:n][:, KEEP].double()
        e_gt = model.energy(gt)
        errs = {o: r["mpvpe"].double().mean(1).numpy() for o, r in occ.items()}
        first = errs if first is None else first
        for o, r in occ.items():
            e = model.energy(r["poses"].double())
            diff = bootstrap(errs[o] - first[o]) if errs is not first else "–"
            print(f"| {fit} | {comp} | {o} | {errs[o].mean():.2f} | {float((e - e_gt[:, None]).mean()):+.1f} | "
                  f"{float((e < e_gt[:, None]).double().mean()):.2f} | {diff} |", flush=True)
        mean = np.mean(list(errs.values()), 0)
        diff = bootstrap(mean - np.mean(list(first.values()), 0)) if errs is not first else "–"
        print(f"| {fit} | {comp} | **mean** | {mean.mean():.2f} | | | {diff} |", flush=True)


if __name__ == "__main__":
    main()
