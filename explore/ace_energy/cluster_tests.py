"""Tests of one component of a mixture of linear energies (fit_cluster.py) on the validation poses of its cluster, next
to the linear energy fitted on all poses (P+L) and the reference network (fact_RDW s10). The cluster of a pose: its
nearest k-means centroid (mixture_clusters.py, the 171 entries of the 19 local rotations), always taken from the truth
(an oracle choice of the component; a mixture would have to infer it). Validation only. Parts (--parts):

- loss: denoising score matching per bin of log s, --n draws (seed 0, t uniform over the loss range, as fit.py; a pool
  of fewer poses is repeated), the same draws for every model; of the validation poses of the cluster and of the
  other clusters;
- holes: as gt_energy.py, on the first 200 validation poses of complete.py whose truth is in the cluster: the share of
  the hypotheses (network, P+L, MLP and, with --completion, the component's own) whose energy lies below the truth's;
- inpaint: the errors of the --out files of complete.py (the component's --completion, P+L, network) on those poses,
  with the paired difference component - P+L and its bootstrap 95% interval over the poses;
- denoise: motion denoising of the 100 validation sequences (hp.evaluation.denoising, DPoser's optimization with the
  settings of RSGM: SDS of scripts/evaluate_denoising.py) at 40 and 100 mm for the component, P+L and the network;
  MPJPE over all frames and over the frames whose truth is in the cluster. Saved per row and noise level in --out
  (denoise_<row>_<mm>mm.pt), resumed from <...>.partial after a stop.

From the root of the repository:

    python explore/ace_energy/cluster_tests.py --fit ~/data/experiments/ace/energy_cluster8_4.pt --k 8 --index 4 \\
        --completion ~/data/experiments/ace/cluster/inpaint_cluster8_4.pt --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from complete import energy_model  # noqa: E402
from diagnose import bootstrap  # noqa: E402
from evaluate_denoising import SDS  # noqa: E402
from fit import REFERENCE, ROTATIONS, SPLIT, draw  # noqa: E402
from gt_energy import ROWS, energies  # noqa: E402
from inspect_energy import energy_predictions, load_energy, network_predictions  # noqa: E402
from hp.evaluation.completion import KEEP, build, valid_poses  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.evaluation.denoising import Body, gt_joints, metrics, noisy_joints, run, split_sequences  # noqa: E402

ACE = Path("~/data/experiments/ace").expanduser()
BASE = "energy_b_rich_products_jd_r0.001.pt"  # P+L


def clusters(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """The nearest centroid of every pose x [..., 19, 3, 3]."""
    flat = x.reshape(-1, 171).float()
    return torch.cat([torch.cdist(flat[i:i + 100_000], centroids).argmin(1)
                      for i in range(0, len(flat), 100_000)]).reshape(x.shape[:-3])


def loss_part(fits: dict, centroids, index: int, n: int, device: str) -> None:
    _, ref, bins, _ = load_energy(next(iter(fits.values())))
    valid = load_reference("valid", ROTATIONS, SPLIT)
    inside = clusters(valid, centroids) == index
    net = build(dict(run=REFERENCE), device)
    models = {"network": None, **{name: load_energy(path, device)[0] for name, path in fits.items()}}
    for where, pool in (("cluster", valid[inside]), ("other clusters", valid[~inside])):
        g = torch.Generator().manual_seed(0)
        x0 = pool[torch.randperm(len(pool), generator=g)[torch.arange(n) % len(pool)]]  # repeated if fewer than n
        t = ref.loss_t_min + (1 - ref.loss_t_min) * torch.rand(len(x0), dtype=torch.float64, generator=g)
        xt, target, std = draw(ref, x0, t, g)
        d = dict(x0=x0, t=t, xt=xt, target=target, std=std)
        k = torch.bucketize(ref.s(t).log(), bins.edges[1:-1])
        print(f"\n== loss, {where}: {len(x0):,} draws of {len(pool):,} validation poses ({min(n, len(pool)):,} poses)")
        print("| model | mean | " + " | ".join(f"s {float(c.exp()):.0e}" for c in bins.centres) + " |")
        print("|---|---|" + "---|" * len(bins.centres))
        base = None
        for name, energy in models.items():
            pred = network_predictions(net, d) if energy is None else energy_predictions(energy, d)
            per = ((pred - target) ** 2).sum((1, 2))
            cells = [f"{float(per[k == b].mean()):.2f}" for b in range(len(bins.centres))]
            print(f"| {name} | {float(per.mean()):.3f} | " + " | ".join(cells) + " |", flush=True)
            if name == "P+L":
                base = per
            elif base is not None:
                print(f"  {name} - P+L, paired over the draws: {bootstrap((per - base).numpy())}")


def holes_part(fits: dict, truth: torch.Tensor, index: int, completion, levels, pose: int, device: str) -> None:
    rows = dict(ROWS)
    files = {r: torch.load(ACE / f, weights_only=True)["occlusions"] for r, f in rows.items()}
    if completion:
        rows["component"] = completion
        files["component"] = torch.load(Path(completion).expanduser(), weights_only=True)["occlusions"]
    keep = (truth == index).nonzero().flatten()
    gt = valid_poses(n=1000)[:200][:, KEEP][keep]
    print(f"\n== holes: {len(keep)} of the 200 poses have the truth in cluster {index}; the share of each row's "
          f"hypotheses below the truth, mean over these poses")
    print("| energy | occlusion | s | " + " | ".join(rows) + " |")
    print("|---|---|---|" + "---|" * len(rows))
    for name, path in fits.items():
        model, energy, _ = energy_model(str(path), REFERENCE, device)
        e_gt = energies(energy, model, gt, levels, device)  # [L, P]
        for occ in files["network"]:
            hyp = torch.stack([files[r][occ]["poses"][keep].float() for r in rows], 1)  # [P, R, 10, 19, 3, 3]
            e_h = energies(energy, model, hyp, levels, device)  # [L, P, R, 10]
            for i, s in enumerate(levels):
                below = (e_h[i] < e_gt[i][:, None, None]).double().mean((0, 2))
                print(f"| {name} | {occ} | {s:g} | " + " | ".join(f"{float(v):.2f}" for v in below) + " |")
            if occ == "legs" and pose in keep:
                p = int((keep == pose).nonzero())
                err = torch.stack([files[r][occ]["mpvpe"][keep[p]].double() for r in rows])  # [R, 10]
                print(f"  pose {pose}, legs, {name}: truth " + "/".join(f"{float(e_gt[i, p]):.0f}"
                                                                       for i in range(len(levels))))
                for r, row in enumerate(rows):
                    order = err[r].argsort()
                    print(f"    {row}: " + "; ".join(f"{float(err[r, j]):.0f} mm " + "/".join(
                        f"{float(e_h[i, p, r, j]):.0f}" for i in range(len(levels))) for j in order))
        print(flush=True)


def inpaint_part(truth: torch.Tensor, index: int, completion: str) -> None:
    keep = (truth == index).numpy()
    rows = {"component": Path(completion).expanduser(), "P+L": ACE / ROWS["P+L"], "network": ACE / ROWS["network"]}
    files = {r: torch.load(f, weights_only=True)["occlusions"] for r, f in rows.items()}
    print(f"\n== inpainting at t81, the {int(keep.sum())} poses of the cluster; MPVPE mean / best of 10, mm")
    print("| occlusion | " + " | ".join(rows) + " | component - P+L | all 200: component | all 200: P+L |")
    print("|---|" + "---|" * (len(rows) + 3))
    per = {r: [] for r in rows}
    for occ in files["P+L"]:
        cells = []
        for r in rows:
            e = files[r][occ]["mpvpe"].double().numpy()
            per[r].append(e.mean(1))
            cells.append(f"{e[keep].mean():.2f} / {e[keep].min(1).mean():.2f}")
        diff = (per["component"][-1] - per["P+L"][-1])[keep]
        print(f"| {occ} | " + " | ".join(cells) + f" | {bootstrap(diff)} | {per['component'][-1].mean():.2f} | "
              f"{per['P+L'][-1].mean():.2f} |")
    mean = {r: np.mean(v, 0) for r, v in per.items()}
    print("| **mean** | " + " | ".join(f"{mean[r][keep].mean():.2f}" for r in rows)
          + f" | {bootstrap((mean['component'] - mean['P+L'])[keep])} | {mean['component'].mean():.2f} | "
          f"{mean['P+L'].mean():.2f} |", flush=True)


def denoise_part(fits: dict, centroids, index: int, sigmas, out: Path, batch: int, device: str,
                 first=None) -> None:
    gt, names = split_sequences("valid", first=first)
    inside = (clusters(gt[:, :, KEEP], centroids) == index).numpy()  # [N, T]
    print(f"\n== motion denoising, {len(gt)} validation sequences; {inside.mean():.2f} of the frames in the cluster "
          f"({int(inside.any(1).sum())} sequences have such frames)")
    body = Body(device=device)
    out.mkdir(parents=True, exist_ok=True)
    makers = {"network": lambda: build(dict(run=REFERENCE), device),
              **{name: (lambda p=path: energy_model(str(p), REFERENCE, device)[0]) for name, path in fits.items()}}
    for sigma in sigmas:
        noisy = noisy_joints(gt_joints(gt, body), sigma)
        result = {}
        for name, make in makers.items():
            path = out / f"denoise_{name.replace('+', '')}_{round(sigma * 1000)}mm.pt"
            key = json.dumps(dict(row=name, fit=str(fits.get(name, REFERENCE)), sigma=sigma, names=names,
                                  settings=SDS[sigma]))
            if path.is_file() and torch.load(path, weights_only=True)["key"] == key:
                result[name] = {k: v.numpy() for k, v in torch.load(path, weights_only=True)["metrics"].items()}
                continue
            print(f"{name}, {round(sigma * 1000)} mm:", flush=True)
            pred = run(make(), gt, noisy, sigma, body, "dposer", batch, partial=path.with_suffix(".partial"), key=key,
                       **SDS[sigma])
            m = metrics(pred, gt, noisy, body)
            torch.save(dict(key=key, pred=pred.half(), metrics={k: torch.from_numpy(v) for k, v in m.items()}), path)
            path.with_suffix(".partial").unlink(missing_ok=True)
            result[name] = m
        print(f"\nsigma {round(sigma * 1000)} mm, MPJPE mm (input {result['network']['input_mpjpe'].mean():.1f}):")
        print("| row | all frames | frames in the cluster | other frames |")
        print("|---|---|---|---|")
        for name, m in result.items():
            e = m["mpjpe"]
            print(f"| {name} | {e.mean():.2f} | {e[inside].mean():.2f} | {e[~inside].mean():.2f} |")
        if "component" in result and "P+L" in result:
            d = result["component"]["mpjpe"] - result["P+L"]["mpjpe"]
            s = inside.any(1)  # per sequence: the mean over its frames in the cluster
            per_seq = np.array([d[i][inside[i]].mean() for i in np.nonzero(s)[0]])
            print(f"  component - P+L on the frames of the cluster, by sequence: {bootstrap(per_seq)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", required=True, help="the component (energy_*.pt of fit_cluster.py)")
    parser.add_argument("--base", default=str(ACE / BASE), help="the energy fitted on all poses (P+L)")
    parser.add_argument("--clusters", default=str(ACE / "mixture_clusters.pt"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--index", type=int, default=4)
    parser.add_argument("--completion", default=None, help="the component's --out file of complete.py")
    parser.add_argument("--parts", nargs="+", default=["loss", "holes", "inpaint", "denoise"],
                        choices=["loss", "holes", "inpaint", "denoise"])
    parser.add_argument("--n", type=int, default=5000, help="validation poses per pool for the loss")
    parser.add_argument("--levels", type=float, nargs="+", default=[0.002, 0.02])
    parser.add_argument("--pose", type=int, default=189)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.04, 0.1])
    parser.add_argument("--batch", type=int, default=10, help="sequences per batch of denoising")
    parser.add_argument("--first", type=int, default=None, help="only the first sequences (a check)")
    parser.add_argument("--out", default=str(ACE / "cluster"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    c = torch.load(Path(args.clusters).expanduser(), weights_only=True)[args.k]
    fits = {"component": Path(args.fit).expanduser(), "P+L": Path(args.base).expanduser()}
    print(f"cluster {args.index} of K = {args.k}; component {fits['component']}; P+L {fits['P+L']}", flush=True)
    if "loss" in args.parts:
        loss_part(fits, c["centroids"], args.index, args.n, args.device)
    if "holes" in args.parts:
        holes_part(fits, c["truth"], args.index, args.completion, args.levels, args.pose, args.device)
    if "inpaint" in args.parts and args.completion:
        inpaint_part(c["truth"], args.index, args.completion)
    if "denoise" in args.parts:
        denoise_part(fits, c["centroids"], args.index, args.sigmas, Path(args.out).expanduser(), args.batch,
                     args.device, args.first)


if __name__ == "__main__":
    main()
