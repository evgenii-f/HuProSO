"""The noise d of DPoser learned by the ELBO (dposer_elbo, phase 1 of docs/elbo_noise_plan.md): its convergence over
the checkpoints, d per joint and axis-angle coordinate, the agreement of the runs (rule 1: the correlation of log d),
and the three terms of -ELBO on validation poses. From the root of the repository:

    PYTHONPATH=src python explore/evaluation/learned_noise.py <run> [<run> ...]
    PYTHONPATH=src python explore/evaluation/learned_noise.py <run> ... --poses 2000 --loglik 200

--loglik n also gives the exact log-likelihood of the probability flow ODE on n of the poses (above the ELBO).
HP_DEVICE: the device (default mps). Everything is per pose in nats of the density of the z-scored vectors.
"""
import argparse
import os
from pathlib import Path

import torch

from hp.data.amass_cfg import AmassJoints4Learning
from hp.evaluation.completion import KEEP, valid_poses
from hp.models.io import load_model

DEVICE = os.environ.get("HP_DEVICE", "mps")


def log_d(directory: Path) -> torch.Tensor:
    model = load_model(directory, device="cpu", weights="weights_ema")
    if getattr(model, "noise_logits", None) is None:
        raise SystemExit(f"{directory} has no learned noise (a dposer_elbo run of the config elbo)")
    return torch.log(model.noise_d.detach().double())


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--poses", type=int, default=2000, help="validation poses for the ELBO terms")
    parser.add_argument("--loglik", type=int, default=0, help="poses for the exact log-likelihood (0: none)")
    args = parser.parse_args()
    runs = [Path(r).expanduser() for r in args.runs]

    final = {}
    for run in runs:
        final[run] = log_d(run)
        print(f"== {run}\n  checkpoint   corr(log d, final)   max |log d - final|")
        for ckpt in sorted(run.glob("checkpoint_*")):
            ld = log_d(ckpt)
            print(f"  {ckpt.name:12s} {corr(ld, final[run]):18.3f} {(ld - final[run]).abs().max():20.3f}")

    print("\nd per joint, axis-angle coordinates x / y / z" + ("" if len(runs) == 1 else ", the runs and their geometric mean"))
    mean = torch.stack(list(final.values())).mean(0)
    for j, name in enumerate(AmassJoints4Learning):
        cells = ["/".join(f"{v:.2f}" for v in torch.exp(ld[3 * j:3 * j + 3])) for ld in final.values()]
        if len(runs) > 1:
            cells.append("/".join(f"{v:.2f}" for v in torch.exp(mean[3 * j:3 * j + 3])))
        print(f"  {name.name:14s} " + "   ".join(cells))
    if len(runs) > 1:
        print("\ncorrelation of log d between the runs (rule 1: > 0.8)")
        for i, a in enumerate(runs):
            for b in runs[i + 1:]:
                print(f"  {a.parent.name}/{a.name} - {b.parent.name}/{b.name}: {corr(final[a], final[b]):.3f}")

    gt = valid_poses(n=args.poses)[:, KEEP]
    print(f"\n-ELBO on {args.poses} validation poses (nats per pose): prior / reconstruction / diffusion = total")
    for run in runs:
        model = load_model(run, device=DEVICE, weights="weights_ema")
        with torch.no_grad():
            x0 = model.normalize(gt.to(DEVICE))
            g = torch.Generator(device=DEVICE).manual_seed(0)
            terms = [sum(chunk) for chunk in zip(*(
                [t.sum().item() for t in model.elbo_terms(x, g)] for x in x0.split(500)))]
        terms = [t / len(x0) for t in terms]
        line = f"  {run.parent.name}/{run.name}: {terms[0]:.3f} / {terms[1]:.2f} / {terms[2]:.2f} = {sum(terms):.2f}"
        if args.loglik:
            exact = -model.log_prob_normalized(x0[:args.loglik]).mean().item()
            line += f"; exact -log p on {args.loglik}: {exact:.2f}"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
