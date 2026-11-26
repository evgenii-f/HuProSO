"""Does the energy prefer the truth to the hypotheses of completion? If E(ground truth) lies above the energy of a wrong
hypothesis, the energy itself is wrong there (a model problem); if below, the sampler failed to find the truth (a method
problem). On the 200 validation poses of complete.py, from the root of the repository:

    python explore/ace_energy/gt_energy.py --pose 189
    python explore/ace_energy/gt_energy.py --levels 0.05 0.2 0.5 1.0 --save ~/data/experiments/ace/gt_energy_high.pt
    python explore/ace_energy/gt_energy.py --shares ~/data/experiments/ace/gt_energy.pt ~/data/experiments/ace/gt_energy_high.pt

For every occlusion, every energy (linear P+L, MLP K8) and the noise levels s, the hypotheses of three rows (the
network, P+L, MLP; K = 10 each, the --out files of complete.py) are scored together with the ground truth:

- "GT < own": share of poses where the truth has a lower energy than every hypothesis of the energy's own row;
- "GT < pool": the same against all 30 hypotheses;
- "GT rank": the mean share of the 30 hypotheses below the truth (0: the truth is lowest);
- "net < own": share of poses where the lowest of the network's hypotheses is below the lowest of the energy's own.

With --shares FILE ... (files saved before), no energy is computed: for every energy, occlusion and s, the mean share
of the hypotheses below the truth, of the energy's own row and of the network's, over all poses and over the worst 20%
(the poses where the energy's own row has the highest mean error). With good samples of a calibrated model and a fair
sampler it is about 0.5; the network's samples are the reference.

The visible joints of the hypotheses are exactly the truth's, so they differ from it only in the hidden joints. With
--pose, one pose with legs hidden is printed hypothesis by hypothesis. E / std is the energy E_t in nats (embedding.py:
E~_t = std(t) E_t), up to a constant per s. The energies of the clean poses only: not of the noised points that the
sampler visits.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from complete import energy_model  # noqa: E402
from hp.evaluation.completion import KEEP, valid_poses  # noqa: E402

ACE = Path("~/data/experiments/ace").expanduser()
REFERENCE = "~/data/experiments/factorial/rsgm_posedflow_fact_RDW_s10/run_0001"
ROWS = {"network": "diag2/inpaint_network_rank.pt", "P+L": "diag2/inpaint_products.pt",
        "MLP": "diag5/inpaint_mlp_pl_k8.pt"}
ENERGIES = {"P+L": "energy_b_rich_products_jd_r0.001.pt", "MLP": "energy_mlp_pl_k8.pt"}


@torch.no_grad()
def energies(energy, model, x: torch.Tensor, levels, device: str, batch: int = 1000) -> torch.Tensor:
    """E / std [levels, *x.shape[:-3]] of poses x [..., 19, 3, 3]."""
    shape, flat = x.shape[:-3], x.reshape(-1, 19, 3, 3).float()
    out = []
    for s in levels:
        t = model.t_of_s(torch.tensor(s, dtype=torch.float64)).float().to(device)
        std = float(np.sqrt(-np.expm1(-s)))
        e = torch.cat([energy(flat[i:i + batch].to(device), t.expand(len(flat[i:i + batch]))).cpu().double()
                       for i in range(0, len(flat), batch)])
        out.append((e / std).reshape(shape))
    return torch.stack(out)


def shares(paths) -> None:
    """The table of --shares from files saved by this script."""
    print("| energy | occlusion | s | own below GT | worst 20% | network below GT | worst 20% |")
    print("|---|---|---|---|---|---|---|")
    for path in paths:
        d = torch.load(Path(path).expanduser(), weights_only=True)
        for key, v in d["results"].items():
            ename, occ = key.split("/")
            own = d["rows"].index(ename)
            mean = v["err"][:, own].mean(1)
            worst = mean > mean.quantile(0.8)
            for i, s in enumerate(d["levels"]):
                a = (v["hyp"][i][:, own] < v["gt"][i][:, None]).double().mean(1)
                b = (v["hyp"][i][:, 0] < v["gt"][i][:, None]).double().mean(1)
                print(f"| {ename} | {occ} | {s:g} | {a.mean():.2f} | {a[worst].mean():.2f} | {b.mean():.2f} | "
                      f"{b[worst].mean():.2f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.002, 0.02])
    parser.add_argument("--pose", type=int, default=189, help="a pose to print hypothesis by hypothesis (legs hidden)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", default=str(ACE / "gt_energy.pt"))
    parser.add_argument("--shares", nargs="+", default=None, metavar="FILE", help="only the table of shares of files")
    args = parser.parse_args()
    if args.shares:
        shares(args.shares)
        return
    gt = valid_poses(n=1000)[:200][:, KEEP]  # [200, 19, 3, 3]
    files = {r: torch.load(ACE / f, weights_only=True)["occlusions"] for r, f in ROWS.items()}
    saved = {}
    for ename, efile in ENERGIES.items():
        model, energy, info = energy_model(str(ACE / efile), REFERENCE, args.device)
        print(f"\n== energy {ename}: {info}", flush=True)
        print("| occlusion | s | GT < own | GT < pool | GT rank | net < own |")
        print("|---|---|---|---|---|---|")
        for occ in files["network"]:
            hyp = torch.stack([files[r][occ]["poses"].float() for r in ROWS], 1)  # [200, 3, 10, 19, 3, 3]
            err = torch.stack([files[r][occ]["mpvpe"].double() for r in ROWS], 1)  # [200, 3, 10]
            e_gt = energies(energy, model, gt, args.levels, args.device)  # [L, 200]
            e_h = energies(energy, model, hyp, args.levels, args.device)  # [L, 200, 3, 10]
            saved[(ename, occ)] = dict(gt=e_gt, hyp=e_h, err=err)
            own = list(ROWS).index(ename)
            for i, s in enumerate(args.levels):
                e = e_h[i]
                a = (e_gt[i] < e[:, own].min(-1).values).double().mean()
                b = (e_gt[i] < e.flatten(1).min(-1).values).double().mean()
                c = (e.flatten(1) < e_gt[i][:, None]).double().mean()
                net = (e[:, 0].min(-1).values < e[:, own].min(-1).values).double().mean()
                print(f"| {occ} | {s:g} | {a:.2f} | {b:.2f} | {c:.2f} | {net:.2f} |")
            if occ == "legs":
                p = args.pose
                print(f"\npose {p}, legs hidden, s = {args.levels}: E/std of the truth "
                      + ", ".join(f"{float(e_gt[i, p]):.1f}" for i in range(len(args.levels))))
                for r, row in enumerate(ROWS):
                    order = err[p, r].argsort()
                    cells = [f"{float(err[p, r, k]):.0f} mm: " + "/".join(
                        f"{float(e_h[i, p, r, k]):.1f}" for i in range(len(args.levels))) for k in order]
                    print(f"  {row}: " + "; ".join(cells))
                print()
    torch.save(dict(levels=args.levels, rows=list(ROWS), results={f"{e}/{o}": v for (e, o), v in saved.items()}),
               args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
