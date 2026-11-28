"""The network proposes, the Fisher energy corrects: the K completions of the reference network (complete.py --network
--rank, diag2/inpaint_network_rank.pt) refined by a few sweeps over the hidden joints of the energy's conditionals
(gibbs.py), the visible joints kept at the truth. The protocol of completion stays (K = 10 hypotheses, their mean
error and APD). A sweep updates every hidden joint once (a random order), with the conditional sharpened by beta:
exp(beta <F_j, x_j>) drawn exactly (beta = inf: its mode U V^T, iterated conditional modes). Validation only: the grid
(--betas x --sweeps) is chosen on the poses --choose (default the first 100 of the 200): the lowest mean MPVPE over
the occlusions among the settings whose best of 10 is at most --best-slack mm above the network's (refinement must not
buy the mean by collapsing the hypotheses), and reported on the others; the network unrefined is the baseline on the
same poses. From the root:

    python explore/ace_energy/refine.py --fit ~/data/experiments/ace/pl/energy_bm.pt
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
from gibbs import load, sample_fisher  # noqa: E402
from pseudo_lik import ACE, proper_svd  # noqa: E402
from hp.evaluation.completion import KEEP, OCCLUSIONS, TOES, Body, at_rest, metrics, valid_poses  # noqa: E402

J = 19


def refine(model, x: torch.Tensor, free, sweeps: int, beta: float, gen: torch.Generator, mh: int = 3) -> torch.Tensor:
    w, a = model.coupling().detach(), model.a.detach()
    extra = bool(getattr(model, "degrees", ()))
    assert not getattr(model, "pairs", False), "refine.py has no pair terms of degree 2"
    x = x.double().clone()
    v = x.reshape(len(x), -1)
    free = torch.as_tensor(list(free))
    with torch.no_grad():
        for _ in range(sweeps):
            for j in free[torch.randperm(len(free), generator=gen)].tolist():
                f = (a[9 * j:9 * j + 9] + v @ w[:, 9 * j:9 * j + 9]).reshape(-1, 3, 3)
                if beta == float("inf"):
                    if extra:
                        raise ValueError("the mode of a conditional with one-body terms is not closed: use a finite beta")
                    u, _, vv = proper_svd(f)
                    x[:, j] = u @ vv.transpose(-1, -2)
                elif not extra:
                    x[:, j] = sample_fisher(beta * f, gen)
                else:
                    old = model.one_body(x[:, j], j)
                    for _ in range(mh):
                        new = sample_fisher(beta * f, gen)
                        g = model.one_body(new, j)
                        ok = torch.log(torch.rand(len(x), dtype=x.dtype, generator=gen)) < beta * (g - old)
                        x[ok, j], old = new[ok], torch.where(ok, g, old)
                v = x.reshape(len(x), -1)
    return x


def scored(pred19: torch.Tensor, gt: torch.Tensor, occ: str, body) -> dict:
    """MPVPE [N, K] and APD [N] of completions pred19 [N, K, 19, 3, 3] (hidden toes at rest, as complete.py)."""
    pred = gt[:, None].repeat(1, pred19.shape[1], 1, 1, 1)
    pred[:, :, KEEP] = pred19.float()
    hidden_toes = [t for t in TOES if t in OCCLUSIONS[occ]]
    pred = at_rest(pred, hidden_toes) if hidden_toes else pred
    m = metrics(pred, gt, occ, body)
    return dict(mpvpe=m["mpvpe"], apd=m["apd"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", default=str(ACE / "pl" / "energy_bm.pt"))
    parser.add_argument("--hypotheses", default=str(ACE / "diag2" / "inpaint_network_rank.pt"))
    parser.add_argument("--betas", type=float, nargs="+", default=[1.0, 2.0, 4.0, float("inf")])
    parser.add_argument("--sweeps", type=int, nargs="+", default=[1, 3, 10])
    parser.add_argument("--choose", type=int, nargs=2, default=[0, 100], help="poses [from, to) that choose")
    parser.add_argument("--best-slack", type=float, default=1.5)
    args = parser.parse_args()
    model = load(args.fit)
    if getattr(model, "degrees", ()):
        args.betas = [b for b in args.betas if b != float("inf")]
    hyp = torch.load(Path(args.hypotheses).expanduser(), weights_only=True)["occlusions"]
    n = next(iter(hyp.values()))["poses"].shape[0]
    gt, body = valid_poses(n=1000)[:n], Body()
    choose = torch.zeros(n, dtype=torch.bool)
    choose[args.choose[0]:args.choose[1]] = True
    report = ~choose
    rows = {"network": {}}
    for occ, h in hyp.items():
        rows["network"][occ] = scored(h["poses"], gt, occ, body)
    for beta in args.betas:
        for sweeps in args.sweeps:
            name = f"beta {beta:g}, {sweeps} sweeps"
            rows[name] = {}
            for occ, h in hyp.items():
                free = [i for i, j in enumerate(KEEP) if j in OCCLUSIONS[occ]]
                gen = torch.Generator().manual_seed(0)
                x = refine(model, h["poses"].reshape(-1, J, 3, 3), free, sweeps, beta, gen)
                rows[name][occ] = scored(x.reshape(n, -1, J, 3, 3), gt, occ, body)
            print(f"  {name}: done", flush=True)

    def mean_over(r, mask, key="mpvpe"):
        pick = {"mpvpe": lambda m: m["mpvpe"].mean(1), "best": lambda m: m["mpvpe"].min(1), "apd": lambda m: m["apd"]}
        return np.mean([pick[key](r[o]) for o in r], 0)[mask.numpy()]

    print(f"\n== choice on poses {args.choose[0]}-{args.choose[1] - 1}: mean MPVPE (mm) over the occlusions, APD (cm)")
    print("| refinement | choose: MPVPE | best of 10 | APD | report: MPVPE | best of 10 | APD |")
    print("|---|---|---|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(f"{mean_over(r, m, k).mean():.2f}" for m in (choose, report)
                                          for k in ("mpvpe", "best", "apd")) + " |")
    limit = mean_over(rows["network"], choose, "best").mean() + args.best_slack
    allowed = [k for k in rows if k != "network" and mean_over(rows[k], choose, "best").mean() <= limit]
    if not allowed:
        print(f"\nno setting keeps best of 10 within {args.best_slack} mm of the network's ({limit:.2f})")
        return
    best = min(allowed, key=lambda k: mean_over(rows[k], choose).mean())
    print(f"\nchosen: {best}. On the {int(report.sum())} report poses, per occlusion (MPVPE mean / best of 10, APD):")
    print("| occlusion | network | refined | refined - network |")
    print("|---|---|---|---|")
    for occ in hyp:
        a, b = rows["network"][occ], rows[best][occ]
        m = report.numpy()
        print(f"| {occ} | {a['mpvpe'][m].mean():.2f} / {a['mpvpe'][m].min(1).mean():.2f}, {a['apd'][m].mean():.2f} | "
              f"{b['mpvpe'][m].mean():.2f} / {b['mpvpe'][m].min(1).mean():.2f}, {b['apd'][m].mean():.2f} | "
              f"{bootstrap(b['mpvpe'][m].mean(1) - a['mpvpe'][m].mean(1))} |")
    d = mean_over(rows[best], report) - mean_over(rows["network"], report)
    print(f"| **mean** | {mean_over(rows['network'], report).mean():.2f} | {mean_over(rows[best], report).mean():.2f} | "
          f"{bootstrap(d)} |")


if __name__ == "__main__":
    main()
