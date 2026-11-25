"""Unconditional generation with the ACE energies as a quick check, on the validation poses: the reference RSGM run's
sampler with its own score ("network") or with the score of an energy (complete.energy_model). From the root:

    python explore/ace_energy/generate.py ~/data/experiments/ace/energy_b_ridge_r0.001.pt ... --samples 5000

Metrics: the harness (hp.evaluation.evaluate, one repeat, no likelihood): Wasserstein, recall and precision (degrees
per joint) against the validation poses; and of PoseD-Flow's protocol (hp.evaluation.posedflow_protocol) FID_val (the
joint statistics of the validation poses), APD and the dNN of 500 samples to the training poses. The rows "train poses"
and "valid poses" are real poses; the last row has the harness of the training poses. Fewer samples than in the
protocols, so the numbers compare only within this table. --out saves the samples.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from complete import energy_model  # noqa: E402
from fit import REFERENCE, ROTATIONS, SPLIT  # noqa: E402
from hp.evaluation.completion import build  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.evaluation.evaluate import EvalConfig, evaluate  # noqa: E402
from hp.evaluation.posedflow_protocol import SmplSkeleton, apd, fid, nearest_neighbour_distance  # noqa: E402

N_CONTRAST = 100_000
SET = 500  # poses per set of FID and APD, as in PoseD-Flow's protocol; also the samples of dNN


def joint_metrics(poses, skeleton, reference, train) -> dict:
    joints = [skeleton.joints(p) for p in poses.split(SET) if len(p) == SET]
    return dict(fid=float(np.mean([fid(j, *reference) for j in joints])), apd=float(np.mean([apd(j) for j in joints])),
                dnn=float(nearest_neighbour_distance(poses[:SET], train).mean()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fits", nargs="*", help="energy_*.pt of fit.py or embedding.py")
    parser.add_argument("--no-network", action="store_true", help="without the row of the reference network")
    parser.add_argument("--band", type=float, nargs=2, default=(0.0, float("inf")), metavar=("S_LO", "S_HI"),
                        help="noise levels s(t) at which the energy is used, the network elsewhere")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="a folder for the samples, one .pt per model")
    args = parser.parse_args()
    valid, train = load_reference("valid", ROTATIONS, SPLIT), load_reference("train", ROTATIONS, SPLIT)
    skeleton = SmplSkeleton()
    joints = skeleton.joints(valid).reshape(len(valid), -1).numpy()
    reference = joints.mean(0), np.cov(joints, rowvar=False)
    order = lambda n, k, seed: torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:k]  # noqa: E731
    contrast = train[order(len(train), N_CONTRAST, 0)]
    cfg = EvalConfig(n_generated=args.samples, n_wasserstein=min(2000, args.samples), n_queries=min(2000, args.samples),
                     n_likelihood=0, repeats=1, sample_batch=args.batch)
    rows = [("train poses", contrast[:args.samples], None),
            ("valid poses", valid[order(len(valid), args.samples, 1)], None)]
    print("| model | Wasserstein | recall | precision | FID_val | APD (cm) | dNN | minutes |\n"
          "|---|---|---|---|---|---|---|---|")
    models = [] if args.no_network else [("network", None)]
    models += [(Path(f).stem, f) for f in args.fits]

    def line(name, report, m, minutes):
        if report is None:
            print(f"| {name} | – | – | – | {m['fid']:.4f} | {m['apd']:.2f} | {m['dnn']:.4f} | – |", flush=True)
            return
        print(f"| {name} | {report['wasserstein']['generated']['joint_mean_deg']['mean']:.2f} | "
              f"{report['recall']['mean_deg']['mean']:.2f} | {report['precision']['mean_deg']['mean']:.2f} | "
              f"{m['fid']:.4f} | {m['apd']:.2f} | {m['dnn']:.4f} | {minutes:.1f} |", flush=True)

    for name, poses, report in rows:
        line(name, report, joint_metrics(poses, skeleton, reference, train), 0)
    for name, path in models:
        t0 = time.perf_counter()
        if path is None:
            model = build(dict(run=REFERENCE), args.device)
        else:
            model = energy_model(path, REFERENCE, args.device, tuple(args.band))[0]
        model.has_log_prob = False  # this instance only: no likelihood
        sample, kept = model.sample, []
        model.sample = lambda n, cond=None: kept.append(sample(n, cond)) or kept[-1]
        report = evaluate(model, valid, cfg)  # without contrast: it would move the random draws of this row only
        poses = torch.cat(kept).float().cpu()
        if args.out:
            out = Path(args.out).expanduser()
            out.mkdir(parents=True, exist_ok=True)
            torch.save(poses, out / f"{name}.pt")
        line(name, report, joint_metrics(poses, skeleton, reference, train), (time.perf_counter() - t0) / 60)
    if models:  # the training poses in the place of the last model's samples, measured with the same random draws
        pieces = iter(contrast[:args.samples].split(args.batch))
        model.sample = lambda n, cond=None: next(pieces)
        report = evaluate(model, valid, cfg)
        print(f"| train poses in the place of the samples | "
              f"{report['wasserstein']['generated']['joint_mean_deg']['mean']:.2f} | "
              f"{report['recall']['mean_deg']['mean']:.2f} | {report['precision']['mean_deg']['mean']:.2f} | "
              f"– | – | – | – |", flush=True)


if __name__ == "__main__":
    main()
