"""Where the linear ACE energy falls short of the network (docs/experiments/2026-11-23_ace_descriptors.md, section
2026-11-26). Four analyses, from the root of the repository, on the CPU:

    python explore/ace_energy/diagnose.py completion            # completion runs of complete.py --out, side by side
    python explore/ace_energy/diagnose.py loss                  # the loss gap by knee flexion and distance to train
    python explore/ace_energy/diagnose.py couplings             # d score_j / d x_i of the energies and the network
    python explore/ace_energy/diagnose.py pairbasis             # can functions of two joints carry the couplings?

completion: the completion files (complete.py --out, 200 validation poses, K = 10, the 4 occlusions) by name: the mean
error, >200 mm, the paired difference to a base row with a bootstrap 95% interval over the poses, per occlusion; the
hypothesis of lowest energy (rows made with --rank); the share of the network's gain over a random training limb that a
row keeps; and the gap to the network by the distance of a pose to its nearest training pose (thirds) and by the
flexion of the left knee (thirds, occlusion left_leg).

loss: the loss gap to the reference network, sum over joints of |score - target|^2, mean over the bins below --s-max
(as collective_test.py), of validation poses: all, the far third (the poses whose nearest training pose is farthest)
and the rest; and of the knees by their flexion, with the share of the gap that the worst 2% of the bent knees carry.

couplings: the blocks d s_j / d x_i (3 x 3, s the scaled score, finite differences of 0.03 rad) at noisy validation
poses, by the distance of i and j along the tree: their rms, the ratio of the energy's to the network's, and the cosine
between them. --networks adds other runs, compared with the reference network as the energies are.

pairbasis: the network's symmetrised block of a pair, regressed on functions of the two joints only: the products of
the derivatives of D^1 of the two (81 columns, the form of the products of fit.py), and with D^2 as well; test R^2 on
held-out poses. With --third, also the gain of every third joint (its 3 principal local coordinates times the 81
columns): on the tree path between the two, off it, and all third joints at once.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit import REFERENCE, ROTATIONS, SPLIT, draw, tree_distance  # noqa: E402
from hp.evaluation.completion import KEEP, OCCLUSIONS, Body, metrics, summary, valid_poses  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.rsgm import PARENTS19, exp_frame  # noqa: E402
from hp.models.so3_harmonics import wigner_d  # noqa: E402
from inspect_energy import FOLDER, energy_predictions, load_energy, network_predictions, stratified_draws  # noqa: E402

OCCS = ["left_leg", "legs", "arms", "trunk"]
NAMES = ["LUpLeg", "RUpLeg", "Spine0", "LLeg", "RLeg", "Spine1", "LFoot", "RFoot", "Spine2", "Neck", "LShoulder",
         "RShoulder", "Head", "LArm", "RArm", "LElbow", "RElbow", "LHand", "RHand"]
KNEES = [3, 4]
COMPLETION = {  # name: complete.py --out file in FOLDER
    "network": "diag/network.pt", "network (ranked)": "diag2/inpaint_network_rank.pt",
    "old base": "diag/energy.pt", "limits only": "diag3/inpaint_jd_r1e+09.pt",
    "base + limits 1e-4": "diag3/inpaint_jd_r0.0001.pt", "base + pairs 1e-4": "diag3/inpaint_rich_pairs_r0.0001.pt",
    "products 1e-4": "diag3/inpaint_rich_products_r0.0001.pt", "P+L 1e-4": "diag3/inpaint_rich_products_jd_r0.0001.pt",
    "P+L 1e-3": "diag2/inpaint_products.pt", "P+L 1e-2": "diag3/inpaint_rich_products_jd_r0.01.pt",
    "no descriptors": "diag3/inpaint_abl_no_descriptors_r0.001.pt",
    "squares": "diag3/inpaint_rich_products_jd_sq_r0.001.pt", "t5": "diag2/inpaint_t5.pt",
    "100k": "diag4/inpaint_products_jd_100k.pt",
    "network no-DPS": "diag/network_nodps.pt", "old no-DPS": "diag/energy_nodps.pt",
    "P+L no-DPS": "diag2/nodps_products.pt", "network SDS": "diag2/sds_network.pt",
    "P+L SDS": "diag2/sds_products.pt", "t5 SDS": "diag2/sds_t5.pt",
}
DIFFS = [("products 1e-4", "old base"), ("base + pairs 1e-4", "old base"), ("base + limits 1e-4", "old base"),
         ("P+L 1e-4", "products 1e-4"), ("P+L 1e-3", "P+L 1e-4"), ("P+L 1e-2", "P+L 1e-3"),
         ("no descriptors", "P+L 1e-3"), ("squares", "P+L 1e-3"), ("t5", "P+L 1e-3"), ("100k", "P+L 1e-3"),
         ("network no-DPS", "network"), ("network SDS", "network"), ("old no-DPS", "old base"),
         ("P+L no-DPS", "P+L 1e-3"), ("P+L SDS", "P+L 1e-3"), ("t5 SDS", "t5")]
FITS = {"P+L 1e-3": "energy_b_rich_products_jd_r0.001.pt", "t5": "energy_b_rich_products_jd_t5_r0.001.pt",
        "P+L 1e-2": "energy_b_rich_products_jd_r0.01.pt", "100k": "energy_b_rich_products_jd_100k_r0.001.pt",
        "old base": "energy_b_smooth_w123_p123.pt"}


def angle(r: torch.Tensor) -> torch.Tensor:
    """Rotation angles in degrees of rotations [..., 3, 3]."""
    return torch.rad2deg(torch.arccos(((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))


def nearest_train(poses: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    """The mean joint angle (degrees) of every pose to its chordally nearest training pose."""
    best, near = torch.full((len(poses),), 9e9), torch.zeros(len(poses), dtype=torch.long)
    q = poses.float().flatten(1)
    for i in range(0, len(train), 200_000):
        d, j = torch.cdist(q, train[i:i + 200_000].float().flatten(1)).min(1)
        near = torch.where(d < best, j + i, near)
        best = torch.minimum(best, d)
    return angle((poses.transpose(-1, -2).double() @ train[near].double())).mean(1)


def bootstrap(x: np.ndarray, n: int = 4000) -> str:
    """Mean and 95% interval over poses of x [poses]."""
    idx = np.random.default_rng(0).integers(0, len(x), (n, len(x)))
    b = x[idx].mean(1)
    return f"{x.mean():+.2f} [{np.quantile(b, .025):+.2f}, {np.quantile(b, .975):+.2f}]"


def completion(args) -> None:
    rows, ranks = {}, {}
    for name, f in COMPLETION.items():
        p = FOLDER / f
        if not p.exists():
            print("missing:", name, p)
            continue
        o = torch.load(p, weights_only=True)["occlusions"]
        if any(k not in o for k in OCCS):
            print("incomplete:", name, p)
            continue
        rows[name] = np.stack([o[k]["mpvpe"].double().numpy() for k in OCCS])  # [occ, poses, K]
        if o["left_leg"].get("energies") is not None:  # [occ, levels, poses, K] -> error of the lowest per level
            e = np.stack([o[k]["energies"].double().numpy() for k in OCCS])
            ranks[name] = [np.take_along_axis(rows[name], e[:, i].argmin(-1)[..., None], -1)[..., 0]
                           for i in range(e.shape[1])]
    net = rows["network"]
    print("Validation, 200 poses, K = 10, MPVPE mm; gap to the network: paired, mean over the occlusions")
    print("| row | mean | best of K | >200 mm | " + " | ".join(OCCS) + " | gap to network |")
    print("|---|---|---|---|" + "---|" * len(OCCS) + "---|")
    for name, a in rows.items():
        occ = " | ".join(f"{v:.1f}" for v in a.mean((1, 2)))
        print(f"| {name} | {a.mean():.2f} | {a.min(2).mean():.2f} | {100 * (a > 200).mean():.1f}% | {occ} | "
              f"{bootstrap((a.mean(2) - net.mean(2)).mean(0))} |")
    print("\n| a - b | all | " + " | ".join(OCCS) + " |\n|---|---|" + "---|" * len(OCCS))
    for a, b in DIFFS:
        if a in rows and b in rows:
            d = rows[a].mean(2) - rows[b].mean(2)
            print(f"| {a} - {b} | {bootstrap(d.mean(0))} | " + " | ".join(bootstrap(x) for x in d) + " |")
    print("\nThe hypothesis of lowest energy (s = 0.002, 0.02, 0.2): mean error, and its paired change to the mean "
          "of K")
    for name, r in ranks.items():
        print(f"{name}: " + "; ".join(f"{x.mean():.2f} ({bootstrap((x - rows[name].mean(2)).mean(0))})" for x in r))
    train = load_reference("train", ROTATIONS, SPLIT)
    full = valid_poses(n=1000)[:200]
    # a baseline without information: the hidden joints of K random training poses in place of the hidden ones
    body, g = Body(), torch.Generator().manual_seed(0)
    sub = train[torch.randperm(len(train), generator=g)[:200_000]].float()
    blind = []
    for occ in OCCS:
        pred = full[:, None].expand(-1, 10, -1, -1, -1).clone()
        pick = sub[torch.randint(0, len(sub), (200, 10), generator=g)]
        for i, k in enumerate(KEEP):
            if k in OCCLUSIONS[occ]:
                pred[:, :, k] = pick[:, :, i]
        blind.append(summary(metrics(pred, full, occ, body))["mpvpe_mean"])
    print("\nThe share of the network's gain over a random training limb that a row keeps (mean of K), per occlusion; "
          "the baseline: " + ", ".join(f"{o} {b:.1f} mm" for o, b in zip(OCCS, blind)))
    for name, a in rows.items():
        share = [(b - x) / (b - y) for b, x, y in zip(blind, a.mean((1, 2)), net.mean((1, 2)))]
        print(f"{name:20s} " + " ".join(f"{100 * x:5.0f}%" for x in share))
    gt = full[:, KEEP]
    dist = nearest_train(gt, train)
    knee = angle(gt[:, 3].double())
    thirds = lambda v: [v.argsort()[a:b].numpy() for a, b in ((0, 67), (67, 134), (134, 200))]  # noqa: E731
    td, tk = thirds(dist), thirds(knee)
    print("\nGap to the network (mm) by the distance to the nearest training pose, thirds "
          + ", ".join(f"{dist[i].min():.1f}-{dist[i].max():.1f} deg" for i in td)
          + "; in occlusion legs; in left_leg by left knee flexion, thirds "
          + ", ".join(f"{knee[i].min():.0f}-{knee[i].max():.0f} deg" for i in tk))
    for name, a in rows.items():
        if name.startswith("network") or "DPS" in name or "SDS" in name:
            continue
        d = a.mean(2) - net.mean(2)
        print(f"{name:20s} all occlusions " + " ".join(f"{d[:, i].mean():6.1f}" for i in td)
              + " | legs " + " ".join(f"{d[1, i].mean():6.1f}" for i in td)
              + " | left_leg by knee " + " ".join(f"{d[0, i].mean():6.1f}" for i in tk))


def loss(args) -> None:
    energies = {n: load_energy(FOLDER / FITS[n])[0] for n in args.fits}
    _, ref, bins, _ = load_energy(FOLDER / FITS["P+L 1e-3"])
    net = load_model(Path(REFERENCE).expanduser(), device="cpu", weights="weights_ema").eval()
    valid = load_reference("valid", ROTATIONS, SPLIT)
    dist = nearest_train(valid, load_reference("train", ROTATIONS, SPLIT))
    limit = torch.quantile(dist, 2 / 3)
    ks = [k for k in range(len(bins.centres)) if float(bins.edges[k + 1].exp()) <= args.s_max]
    gap = {n: [] for n in energies}  # per draw [N, 19], per bin
    flex, far = [], []
    for k in ks:
        g = torch.Generator().manual_seed(1000 * 60 + k)
        pick = torch.randperm(len(valid), generator=g)[:args.poses]
        d = stratified_draws(ref, bins, k, valid, args.poses, 60)
        assert torch.equal(valid[pick], d["x0"])
        ln = ((network_predictions(net, d) - d["target"]) ** 2).sum(2)
        for n, e in energies.items():
            gap[n].append(((energy_predictions(e, d) - d["target"]) ** 2).sum(2) - ln)
        far.append(dist[pick] > limit)
        flex.append(angle(d["x0"][:, KNEES].double()))
        print(f"bin {k}: s {float(bins.centres[k].exp()):.1e}", flush=True)
    far, flex = torch.stack(far), torch.stack(flex)  # [bins, N], [bins, N, 2]
    cuts = [0, 18, 66, 180]
    print(f"\nValidation, {args.poses} poses per bin, {len(ks)} bins with s <= {args.s_max}; loss gap to the network, "
          "mean over the bins (a pose's gap is the sum over joints)")
    print("| fit | all | far third | rest | knees 0-18 deg | 18-66 | >66 | worst 2% of >66 carry |")
    print("|---|---|---|---|---|---|---|---|")
    for n, v in gap.items():
        v = torch.stack(v)  # [bins, N, 19]
        tot = v.sum(-1)
        a = tot.mean(1).mean()
        f = torch.stack([tot[b][far[b]].mean() for b in range(len(ks))]).mean()
        r = torch.stack([tot[b][~far[b]].mean() for b in range(len(ks))]).mean()
        kn = v[..., KNEES]  # gap of every knee, [bins, N, 2]
        per = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            m = (flex >= lo) & (flex < hi)
            per.append(kn[m].mean())
        bent = kn[flex >= 66].sort().values
        worst = bent[-max(1, len(bent) // 50):].sum() / bent.sum()
        print(f"| {n} | {a:.3f} | {f:.3f} | {r:.3f} | " + " | ".join(f"{x:.3f}" for x in per)
              + f" | {100 * worst:.0f}% of {len(bent)} |")
    print("(knee columns: the mean gap of one knee in one draw, by the knee's flexion in the clean pose)")


def scaled(model, x: torch.Tensor, t: torch.Tensor, std: torch.Tensor, network: bool) -> torch.Tensor:
    if network:
        with torch.no_grad():
            return torch.cat([model.score(x[i:i + 1000].float(), t[i:i + 1000].float()).double()
                              * std[i:i + 1000, :, None] for i in range(0, len(x), 1000)])
    return torch.cat([model.scaled_score(x[i:i + 1000].float(), t[i:i + 1000].float()).detach().double()
                      for i in range(0, len(x), 1000)])


def score_jacobian(model, xt, t, std, network: bool, h: float = 0.03) -> torch.Tensor:
    """[N, 19 (j), 3, 19 (i), 3]: d s_j / d x_i by central differences along x_i exp(h e_a)."""
    out = torch.zeros(len(xt), 19, 3, 19, 3)
    for i in range(19):
        for a in range(3):
            v = torch.zeros(len(xt), 19, 3, dtype=torch.float64)
            v[:, i, a] = h
            out[:, :, :, i, a] = ((scaled(model, exp_frame(xt, v), t, std, network)
                                   - scaled(model, exp_frame(xt, -v), t, std, network)) / (2 * h)).float()
    return out


def noisy(ref, x0: torch.Tensor, s: float, seed: int = 9):
    t = torch.full((len(x0),), float(ref.t_of_s(torch.tensor(s, dtype=torch.float64))), dtype=torch.float64)
    xt, _, std = draw(ref, x0, t, torch.Generator().manual_seed(seed))
    return xt, t, std


def couplings(args) -> None:
    models = {n: (load_energy(FOLDER / FITS[n])[0], False) for n in args.fits}
    _, ref, _, _ = load_energy(FOLDER / FITS["P+L 1e-3"])
    net = load_model(Path(REFERENCE).expanduser(), device="cpu", weights="weights_ema").eval()
    for f in args.networks:
        models[Path(f).parent.name] = (load_model(Path(f).expanduser(), device="cpu", weights="weights_ema").eval(),
                                       True)
    valid = load_reference("valid", ROTATIONS, SPLIT)
    x0 = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(3))[:args.poses]]
    td = torch.tensor([[tree_distance(min(i, j), max(i, j)) if i != j else 0 for i in range(19)] for j in range(19)])
    dists = sorted(set(td.flatten().tolist()) - {0})
    for s in args.s:
        xt, t, std = noisy(ref, x0, s)
        base = score_jacobian(net, xt, t, std, True).permute(0, 1, 3, 2, 4)  # [N, j, i, 3, 3]
        jac = {n: score_jacobian(m, xt, t, std, w).permute(0, 1, 3, 2, 4) for n, (m, w) in models.items()}
        print(f"\ns = {s}, {args.poses} validation poses: by tree distance, rms of a block of the network; for every "
              "model, rms ratio to the network / cosine with it")
        print("| distance | pairs | network rms | " + " | ".join(jac) + " |\n|---|---|---|" + "---|" * len(jac))
        for d in dists:
            m = td == d
            c = base[:, m]
            cells = []
            for b in jac.values():
                b = b[:, m]
                cells.append(f"{float(b.pow(2).mean().sqrt() / c.pow(2).mean().sqrt()):.2f} / "
                             f"{float((b * c).sum() / (b.norm() * c.norm())):.2f}")
            print(f"| {d} | {int(m.sum()) // 2} | {float(c.pow(2).mean().sqrt()):.3f} | " + " | ".join(cells) + " |",
                  flush=True)


def path_joints(i: int, k: int) -> set:
    """The joints on the tree path between i and k, without i and k."""
    def up(j):
        out = [j]
        while PARENTS19[out[-1]] >= 0:
            out.append(PARENTS19[out[-1]])
        return out

    a, b = up(i), up(k)
    common = [q for q in a if q in b]
    if common:
        top = common[0]
        return (set(a[:a.index(top) + 1]) | set(b[:b.index(top) + 1])) - {i, k}
    return (set(a) | set(b)) - {i, k}


def pairbasis(args) -> None:
    net = load_model(Path(REFERENCE).expanduser(), device="cpu", weights="weights_ema").eval()
    valid = load_reference("valid", ROTATIONS, SPLIT)
    n, ntr = args.poses, int(0.75 * args.poses)
    x0 = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(4))[:n]]
    train = load_reference("train", ROTATIONS, SPLIT)[::200].double()
    centre = train.flatten(2).mean(0)  # principal local coordinates of every joint, scaled to unit variance
    axes = torch.zeros(19, 9, 3, dtype=torch.float64)
    for j in range(19):
        z = train[:, j].flatten(1) - centre[j]
        _, sv, vt = torch.linalg.svd(z, full_matrices=False)
        axes[j] = vt[:3].T / (sv[:3] / len(z) ** .5)
    pairs = {}
    for i in range(19):
        for k in range(i + 1, 19):
            pairs.setdefault(tree_distance(i, k), []).append((i, k))
    rng = np.random.default_rng(1)
    chosen = {d: [pairs[d][j] for j in rng.choice(len(pairs[d]), min(args.per_distance, len(pairs[d])),
                                                   replace=False)] for d in args.distances}
    for s in args.s:
        xt, t, std = noisy(net, x0, s)
        h = score_jacobian(net, xt, t, std, True)
        hh = 1e-4

        def dw(j, l):  # [N, 3 (direction), dim]: derivative of D^l(x_j exp(h e_a))
            out = []
            for a in range(3):
                v = torch.zeros(n, 1, 3, dtype=torch.float64)
                v[:, :, a] = hh
                out.append(((wigner_d(exp_frame(xt[:, j:j + 1], v)[:, 0], l)
                             - wigner_d(exp_frame(xt[:, j:j + 1], -v)[:, 0], l)) / (2 * hh)).flatten(1))
            return torch.stack(out, 1)

        u = torch.einsum("njp,jpa->nja", xt.flatten(2) - centre, axes).clamp(-4, 4)

        def r2(x, y, lam):
            a = x[:ntr].reshape(-1, x.shape[-1])
            g = a.T @ a
            g += lam * g.diagonal().mean() * torch.eye(len(g), dtype=torch.float64)
            w = torch.linalg.solve(g, a.T @ y[:ntr].reshape(-1))
            return 1 - float((y[ntr:] - x[ntr:] @ w).pow(2).sum() / y[ntr:].pow(2).sum())

        print(f"\ns = {s}, {n} validation poses ({ntr} fit, the rest test): test R^2 of the network's block of a pair")
        head = "| distance | pair | D1 x D1 (81) | + D2 (1156) |"
        if args.third:
            head += " best third joint | on path | off path | all third joints |"
        print(head + "\n|" + "---|" * (head.count("|") - 1))
        for d, ps in chosen.items():
            for i, k in ps:
                y = (0.5 * (h[:, k, :, i, :] + h[:, i, :, k, :].transpose(1, 2))).reshape(n, 9).double()
                di, dk = {1: dw(i, 1), 2: dw(i, 2)}, {1: dw(k, 1), 2: dw(k, 2)}
                cols = lambda ls: torch.cat([torch.einsum("nap,ncq->ncapq", di[a], dk[b]).reshape(n, 9, -1)  # noqa
                                             for a, b in ls], -1)
                b1 = cols([(1, 1)])
                base = r2(b1, y, 1e-4)
                both = r2(cols([(1, 1), (2, 1), (1, 2), (2, 2)]), y, 1e-3)
                line = f"| {d} | {NAMES[i]}-{NAMES[k]} | {base:.2f} | {both:.2f} |"
                if args.third:
                    gain = {q: r2(torch.cat([b1] + [b1 * u[:, q, c][:, None, None] for c in range(3)], -1), y, 1e-3)
                            - base for q in range(19) if q not in (i, k)}
                    on = [v for q, v in gain.items() if q in path_joints(i, k)]
                    off = [v for q, v in gain.items() if q not in path_joints(i, k)]
                    best = max(gain, key=gain.get)
                    full = torch.cat([b1] + [b1 * u[:, q, c][:, None, None] for q in gain for c in range(3)], -1)
                    every = max(r2(full, y, lam) for lam in (3e-3, 3e-2)) - base
                    line += (f" {NAMES[best]} +{gain[best]:.3f} | {np.mean(on) if on else float('nan'):+.3f} | "
                             f"{np.mean(off):+.3f} | {every:+.3f} |")
                print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    sub.add_parser("completion")
    p = sub.add_parser("loss")
    p.add_argument("--fits", nargs="+", default=["P+L 1e-3", "t5", "P+L 1e-2", "100k"], choices=list(FITS))
    p.add_argument("--poses", type=int, default=4000)
    p.add_argument("--s-max", type=float, default=0.04)
    p = sub.add_parser("couplings")
    p.add_argument("--fits", nargs="+", default=["P+L 1e-3", "t5", "old base"], choices=list(FITS))
    p.add_argument("--networks", nargs="*", default=[], help="other runs (folders) to compare with the network")
    p.add_argument("--poses", type=int, default=240)
    p.add_argument("--s", type=float, nargs="+", default=[0.005, 0.05])
    p = sub.add_parser("pairbasis")
    p.add_argument("--poses", type=int, default=6000)
    p.add_argument("--s", type=float, nargs="+", default=[0.02])
    p.add_argument("--distances", type=int, nargs="+", default=[1, 2, 5, 8, 10])
    p.add_argument("--per-distance", type=int, default=3)
    p.add_argument("--third", action="store_true")
    args = parser.parse_args()
    {"completion": completion, "loss": loss, "couplings": couplings, "pairbasis": pairbasis}[args.what](args)


if __name__ == "__main__":
    main()
