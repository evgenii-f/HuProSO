"""Mixing the scores of the reference network and a partner, (1 - lam) network + lam partner, by the validation loss in
the bins below s = 0.04 (docs/experiments/ace/2026-11-26_network_ideas.md, section 2026-11-26: ideas from ACE for the
network). The partner is an energy of fit.py (`--fit`) or another RSGM run with the same isotropic noise (`--run`).

The loss of the mixture is quadratic in lam, so one pass of the predictions gives the whole curve: lam is fitted on the
draws of seed 1 (per bin and one for all bins) and evaluated on the draws of seed 2 (the same validation pool). From
the root:

    python explore/ace_energy/mix_scores.py --fit ~/data/experiments/ace/energy_b_rich_products_jd_r0.001.pt \
        --run ~/data/experiments/factorial/rsgm_posedflow_fact_RD_s10/run_0001
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_energy import (REFERENCE, ROTATIONS, SPLIT, energy_predictions, load_energy, load_model,  # noqa: E402
                            load_reference, network_predictions, stratified_draws)


def parts(pred_net, pred_other, draws):
    """[bins, 3, N]: |r_n|^2, r_n . r_o and |r_o|^2 of every draw, r the residual of a prediction."""
    out = []
    for pn, po, d in zip(pred_net, pred_other, draws):
        rn, ro = pn - d["target"], po - d["target"]
        out.append(torch.stack([(rn ** 2).sum((1, 2)), (rn * ro).sum((1, 2)), (ro ** 2).sum((1, 2))]))
    return torch.stack(out)


def mixed(p, lam):
    """The loss [bins, N] of (1 - lam) network + lam partner."""
    return (1 - lam) ** 2 * p[:, 0] + 2 * lam * (1 - lam) * p[:, 1] + lam ** 2 * p[:, 2]


def best_lam(m):
    """lam minimizing the mean loss of the means m [..., 3], clamped to [0, 1]."""
    return ((m[..., 0] - m[..., 1]) / (m[..., 0] - 2 * m[..., 1] + m[..., 2])).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", nargs="*", default=[], help="energies of fit.py as partners")
    parser.add_argument("--run", nargs="*", default=[], help="RSGM run folders as partners (isotropic noise)")
    parser.add_argument("--poses", type=int, default=2500, help="draws per bin and seed")
    parser.add_argument("--s-max", type=float, default=0.04)
    parser.add_argument("--batch", type=int, default=250)
    args = parser.parse_args()

    net = load_model(Path(REFERENCE).expanduser(), device="cpu", weights="weights_ema").eval()
    first = args.fit[0] if args.fit else "~/data/experiments/ace/energy_b_rich_products_jd_r0.001.pt"
    _, ref, bins, _ = load_energy(first)
    ks = [k for k in range(len(bins.centres)) if float(bins.edges[k + 1].exp()) <= args.s_max]
    pool = load_reference("valid", ROTATIONS, SPLIT)
    draws = {seed: [stratified_draws(ref, bins, k, pool, args.poses, seed) for k in ks] for seed in (1, 2)}
    predict = lambda f: {seed: [f(d) for d in ds] for seed, ds in draws.items()}  # noqa: E731
    base = predict(lambda d: network_predictions(net, d, args.batch))
    partners = {}
    for path in args.fit:
        energy = load_energy(path)[0]
        partners[Path(path).stem] = predict(lambda d: energy_predictions(energy, d, args.batch))
    for path in args.run:
        other = load_model(Path(path).expanduser(), device="cpu", weights="weights_ema").eval()
        partners[Path(path).parent.name] = predict(lambda d: network_predictions(other, d, args.batch))

    for name, pred in partners.items():
        p1, p2 = parts(base[1], pred[1], draws[1]), parts(base[2], pred[2], draws[2])
        m = p1.mean(-1)  # [bins, 3]
        lam_bin, lam = best_lam(m), float(best_lam(m.sum(0)))
        corr = m[:, 1] / (m[:, 0] * m[:, 2]).sqrt()
        print(f"== {name}: partner alone {float(p2[:, 2].mean()):.3f}, network alone {float(p2[:, 0].mean()):.3f}")
        print("   correlation of the residuals per bin", " ".join(f"{float(c):.3f}" for c in corr))
        print("   lam per bin", " ".join(f"{float(v):.2f}" for v in lam_bin), f"| one lam {lam:.3f}")
        for chosen, label in ((lam, "one lam"), (lam_bin[:, None], "lam per bin")):
            d = mixed(p2, chosen) - p2[:, 0]
            se = float((d.var(-1) / d.shape[-1]).sum().sqrt()) / len(ks)
            print(f"   {label}, fitted on seed 1, on seed 2: loss change {float(d.mean()):+.3f} (s.e. {se:.3f})")
        print("   per bin at one lam", " ".join(f"{float(v):+.2f}" for v in (mixed(p2, lam) - p2[:, 0]).mean(-1)))


if __name__ == "__main__":
    main()
