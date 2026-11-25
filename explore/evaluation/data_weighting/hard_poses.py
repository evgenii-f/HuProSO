"""Which poses are hard, and could another choice of the training poses help? The analyses of the data-weighting report
(docs/experiments/2026-11-22_data_weighting.md), validation and training poses only. Run from the root:

    PYTHONPATH=src python explore/evaluation/data_weighting/hard_poses.py difficulty
    PYTHONPATH=src python explore/evaluation/data_weighting/hard_poses.py train-sample --device mps   # about 3 minutes
    PYTHONPATH=src python explore/evaluation/data_weighting/hard_poses.py train-report
    PYTHONPATH=src python explore/evaluation/data_weighting/hard_poses.py signals
    PYTHONPATH=src python explore/evaluation/data_weighting/hard_poses.py dopt      # about 8 minutes, CPU; needs
                                                                     # hp.models.so3_harmonics (dev from 8f5785b)
    PYTHONPATH=src python explore/evaluation/data_weighting/hard_poses.py study     # the Optuna study rdw_inpaint
    PYTHONPATH=src python explore/evaluation/data_weighting/hard_poses.py denoising
    PYTHONPATH=src python explore/evaluation/data_weighting/hard_poses.py test

difficulty    do D-Flow, inpainting and SDS find the same poses hard, and does the rarity predict it (200 and 1000
              validation poses, fact_RDW, beta 0); rarity against the error of motion denoising.
train-sample  inpainting (the fixed point of evaluate.py) of 300 training poses, 60 per quintile of rarity, beta 0 and
              3.3; saved to ~/data/experiments/data_weighting/hard_poses/train_sample.npz.
train-report  the errors and changes of those poses by rarity, next to the 1000 validation poses.
signals       the denoising score matching loss of beta 0 and the disagreement of three seeds of RSGM 512x4 against the
              error, as signals for hard-example mining / active learning.
dopt          D-optimality: leverage and the MaxVol extrapolation grade gamma of linear features (the entries of D^l of
              the rotations), and the D-optimal sampling weights, against the rarity.
study         the noise of near-uniform trials, the trial of the rule, and the curve of hv_best against Wasserstein.
test          the test points of the six fronts (config/completion_test_points.yaml): band mean and hypervolume of best of
              10 from the cached metrics of the test rows (5000 poses, K = 10), by the rule of select_test_points.py.
denoising     MPJPE and MPVPE of beta 0 against 3.3 (100 validation chunks, two seeds).

Rarity of a pose: the RMS angle per joint to its 5th nearest c100 training pose (rarity.py); of a validation or test
pose the same against the c100 poses.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import roma
import torch
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rarity as R  # noqa: E402

from hp.evaluation.completion import KEEP, OCCLUSIONS, Body, build, metrics, run, valid_poses  # noqa: E402

E = Path("~/data/experiments").expanduser()
OUT = E / "data_weighting/hard_poses"
O = list(OCCLUSIONS)
METHODS = ("dflow", "inpaint", "sds")
INPAINT = dict(steps=200, schedule="edm", guidance=0.09515197497012039, resample=5, replace=False)
RUNS = {"b0": "factorial/rsgm_posedflow_fact_RDW_s10", "b3.3": "data_weighting/rsgm_posedflow_rdw_b3.3_s10"}
EVAL = {"b0": "uniform", "b3.3": "b3.3"}  # the names of evaluate.py
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype,
                                      np.dtypes.Float64DType, np.dtypes.Float32DType])


def rho(a, b):
    return spearmanr(a, b)[0]


def residual(y, *xs):
    """Ranks of y with the ranks of xs regressed out."""
    x = np.column_stack([np.ones(len(y))] + [rankdata(v) for v in xs])
    y = rankdata(y)
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def ci(x):
    x = np.asarray(x)
    return f"{x.mean():+.1f} ± {1.96 * x.std(ddof=1) / np.sqrt(len(x)):.1f}"


def errors(name: str):
    """Per occlusion the mean MPVPE (over K) [1000] and APD of evaluate.py, inpainting, 1000 validation poses."""
    d = {o: torch.load(E / f"data_weighting/eval/{name}_{o}.pt", weights_only=True) for o in O}
    return {o: d[o]["mpvpe"] for o in O}, {o: d[o]["apd"] for o in O}


def first200(name: str):
    folder = E / "completion/cache/valid_first200" / name
    return {o: torch.load(folder / f"{o}_metrics.pt", weights_only=True)["plain"] for o in O}


def mean_rank(d):
    return np.mean([rankdata(d[o]) for o in O], axis=0)


def reference():
    """The c100 poses, learned joints, as flattened matrices [R, 171] and as rotations [R, 19, 3, 3]."""
    q = torch.as_tensor(torch.load(R.C10, weights_only=True))[::R.REF_STRIDE]
    rot = roma.unitquat_to_rotmat(q[:, R.JOINTS].double()).float()
    return rot.reshape(len(rot), -1), rot


def rarity_of(x: torch.Tensor, ref: torch.Tensor, cols=None) -> np.ndarray:
    """RMS angle (deg) per joint of rotations x [N, 19, 3, 3] to the 5th nearest reference pose."""
    q = x.reshape(len(x), -1).float()
    if cols is not None:
        q, ref = q[:, cols], ref[:, cols]
    j, out = q.shape[1] // 9, []
    for part in q.split(512):
        kth = (part @ ref.T).topk(R.K, 1).values[:, -1]
        chord2 = (6 * j - 2 * kth).clamp_min(0) / j
        out.append(torch.acos((1 - chord2 / 4).clamp(-1, 1)).rad2deg())
    return torch.cat(out).numpy()


def hidden_columns(occlusion):
    keep = list(KEEP)
    return [9 * i + c for i, j in enumerate(keep) if j in OCCLUSIONS[occlusion] for c in range(9)]


def validation():
    """Validation poses [1000, 19, 3, 3] and their rarity."""
    ref, _ = reference()
    x = valid_poses(n=1000, seed=0)[:, list(KEEP)]
    return x, rarity_of(x, ref), ref


def train_index():
    """The 300 training poses of train-sample: 60 random ones per quintile of rarity, in order of rarity."""
    r = torch.load(E / "data_weighting/rarity.pt", weights_only=True)
    g, order, n, idx = torch.Generator().manual_seed(0), r.argsort(), len(r), []
    for k in range(5):
        part = order[k * n // 5:(k + 1) * n // 5]
        idx.append(part[torch.randperm(len(part), generator=g)[:60]])
    return torch.cat(idx), r


def train_rotations(idx):
    q = torch.as_tensor(torch.load(R.C10, weights_only=True))[idx]
    return roma.unitquat_to_rotmat(q.double()).float()  # [300, 22, 3, 3]


# ---- the analyses ----

def difficulty(_):
    m = {k: first200(f"uniform_{k}") for k in METHODS}
    e = {(k, o): m[k][o]["mpvpe"].numpy().mean(1) for k in METHODS for o in O}
    a = {(k, o): m[k][o]["apd"].numpy() for k in METHODS for o in O}
    print("200 validation poses, fact_RDW (beta 0), same occlusion, Spearman of the mean MPVPE between methods")
    for x, y in (("dflow", "inpaint"), ("dflow", "sds"), ("inpaint", "sds")):
        print(f"  {x} / {y}:", [round(float(rho(e[x, o], e[y, o])), 2) for o in O])
    print("error against APD, per occlusion:",
          {k: [round(float(rho(e[k, o], a[k, o])), 2) for o in O] for k in METHODS})
    x, r, ref = validation()
    mm = {k: mean_rank({o: e[k, o] for o in O}) for k in METHODS}
    for p, q in (("dflow", "inpaint"), ("dflow", "sds"), ("inpaint", "sds")):
        print(f"  mean rank over 4 occlusions, {p} / {q}: {rho(mm[p], mm[q]):.2f}, beyond rarity "
              f"{rho(residual(mm[p], r[:200]), residual(mm[q], r[:200])):.2f}")
    for k in METHODS:
        hard = set(np.argsort(-mm[k])[:20])
        print(f"  {k}: mean rank vs rarity {rho(mm[k], r[:200]):.2f}; hardest 10% in the rarest 10% / 30%: "
              f"{len(hard & set(np.argsort(-r[:200])[:20])) / 20:.2f} / {len(hard & set(np.argsort(-r[:200])[:60])) / 20:.2f}")
    err, _ = errors("uniform")
    em = {o: err[o].mean(1) for o in O}
    ll = torch.load(E / "data_weighting/eval/uniform_loglik.pt", weights_only=True)["loglik"]
    print("1000 validation poses, inpainting: correlation of the errors of two occlusions")
    print(np.array([[round(float(rho(em[p], em[q])), 2) for q in O] for p in O]))
    count = np.zeros(1000, int)
    for o in O:
        count[np.argsort(-em[o])[:100]] += 1
    print("  poses in the hardest 10% of 0..4 occlusions:", np.bincount(count, minlength=5))
    print(f"  mean rank vs rarity {rho(mean_rank(em), r):.2f}, vs -loglik {rho(mean_rank(em), -ll):.2f}; "
          f"rarity vs -loglik {rho(r, -ll):.2f}")
    print(f"  rarity (deg): training median {np.median(torch.load(E / 'data_weighting/rarity.pt', weights_only=True)):.1f}, "
          f"validation median {np.median(r):.1f}")
    hidden = {o: rarity_of(x, ref, hidden_columns(o)) for o in O}
    print("  rarity of the hidden part vs error:", [round(float(rho(em[o], hidden[o])), 2) for o in O])
    # motion denoising: the error of a frame against its rarity
    from hp.evaluation.denoising import sequences

    seq = sequences("valid")[0][:100]
    rr = rarity_of(seq.reshape(-1, 21, 3, 3)[:, list(KEEP)], ref)
    for s in ("40", "100"):
        for name in ("RSGM_512x4:_their_c10,_DPoser_method", "RSGM_512x4:_their_c10,_D-Flow"):
            j = torch.load(E / f"denoising/cache/{name}/valid_{s}mm.pt", weights_only=True)["metrics"]["mpjpe"].numpy().ravel()
            print(f"denoising {s} mm {name}: frame MPJPE vs rarity {rho(j, rr):.2f}; by rarity quintile "
                  f"{np.round([j[np.argsort(rr)[k * 1200:(k + 1) * 1200]].mean() for k in range(5)], 2)} mm")


def train_sample(args):
    OUT.mkdir(parents=True, exist_ok=True)
    idx, r = train_index()
    gt, body, out = train_rotations(idx)[:, 1:22], Body(), {"r": r[idx].numpy()}
    for name, path in RUNS.items():
        model = build(dict(run=str(E / path / "run_0001")), args.device)
        for o in O:
            t0 = time.time()
            torch.manual_seed(10)
            pred = run(model, gt, o, 10, batch=500, together=True, progress=False, method="inpaint", prior=INPAINT)
            m = metrics(pred, gt, o, body)
            out[f"{name}_{o}_e"], out[f"{name}_{o}_apd"] = m["mpvpe"], m["apd"]
            print(f"{name} {o}: {time.time() - t0:.0f} s", flush=True)
    np.savez(OUT / "train_sample.npz", **out)


def train_report(_):
    z = np.load(OUT / "train_sample.npz")
    _, r, _ = validation()
    for label, f in (("mean", lambda a: a.mean(1)), ("best of 10", lambda a: a.min(1))):
        t = {k: np.mean([f(z[f"{k}_{o}_e"]) for o in O], 0) for k in RUNS}
        v = {k: np.mean([f(errors(EVAL[k])[0][o]) for o in O], 0) for k in RUNS}
        print(f"\n{label} (mm), four occlusions, inpainting; quintiles common -> rare")
        print("  train, beta 0:     ", np.round(t["b0"].reshape(5, 60).mean(1), 1), f"all {t['b0'].mean():.1f}")
        d = t["b3.3"] - t["b0"]
        print("  train, 3.3 - 0:    ", [ci(x) for x in d.reshape(5, 60)], "all", ci(d))
        d, q = v["b3.3"] - v["b0"], np.searchsorted(np.quantile(r, [.2, .4, .6, .8]), r)
        print("  valid, beta 0:     ", [round(float(v["b0"][q == k].mean()), 1) for k in range(5)], f"all {v['b0'].mean():.1f}")
        print("  valid, 3.3 - 0:    ", [ci(d[q == k]) for k in range(5)], "all", ci(d))
    train_r = torch.load(E / "data_weighting/rarity.pt", weights_only=True).numpy()
    print("validation poses per quintile of the training rarity:",
          np.bincount(np.searchsorted(np.quantile(train_r, [.2, .4, .6, .8]), r), minlength=5))
    t0 = np.mean([z[f"b0_{o}_e"].mean(1) for o in O], 0)
    v0 = np.mean([errors("uniform")[0][o].mean(1) for o in O], 0)
    for lo, hi in ((5, 8), (8, 11), (11, 15), (15, 30)):
        a, c = (z["r"] >= lo) & (z["r"] < hi), (r >= lo) & (r < hi)
        print(f"  rarity {lo}-{hi} deg, beta 0 mean: train {t0[a].mean():.1f} (n={a.sum()}), valid {v0[c].mean():.1f} (n={c.sum()})")
    print("training poses: error vs rarity", [round(float(rho(z[f'b0_{o}_e'].mean(1), z['r'])), 2) for o in O],
          "error vs APD", [round(float(rho(z[f"b0_{o}_e"].mean(1), z[f"b0_{o}_apd"])), 2) for o in O])


def signals(_):
    from hp.models import igso3
    from hp.models.rsgm import exp_frame

    idx, _ = train_index()
    x = torch.cat([valid_poses(n=1000, seed=0)[:, list(KEEP)].double(),
                   train_rotations(idx)[:, R.JOINTS].double()])
    paths = {"b0": E / RUNS["b0"] / "run_0001",
             **{f"s{i}": E / f"rsgm_posedflow_small_s1{i}/run_0001" for i in (0, 1, 2)}}
    models = {k: build(dict(run=str(p)), "cpu").eval() for k, p in paths.items()}
    b0, gen = models["b0"], torch.Generator().manual_seed(1)
    levels, draws = torch.linspace(0.02, 0.95, 12, dtype=torch.float64), 4
    loss, dis = np.zeros((len(x), len(levels))), np.zeros(len(x))
    for i, level in enumerate(levels):
        for _ in range(draws):
            t = level.expand(len(x))
            s, std = b0.joint_s(t), b0.joint_std(t)
            omega, axis = b0.angles.sample(s, gen), igso3.random_axes((len(x), 19), gen, device="cpu")
            xt = exp_frame(x, omega[..., None] * axis)
            target = std[..., None] * igso3.d_log_density(omega, s)[..., None] * axis
            with torch.no_grad():
                out = {k: (torch.cat([m.score(xt[a:a + 512].float(), t[a:a + 512].float()) for a in range(0, len(x), 512)])
                           * std[..., None].float()).double() for k, m in models.items()}
            loss[:, i] += ((out["b0"] - target) ** 2).sum((1, 2)).numpy() / draws
            dis += torch.stack([out[k] for k in out if k != "b0"]).var(0).sum((1, 2)).numpy() / draws / len(levels)
    _, r, _ = validation()
    z, ll = np.load(OUT / "train_sample.npz"), torch.load(E / "data_weighting/eval/uniform_loglik.pt", weights_only=True)["loglik"]
    err, _ = errors("uniform")
    ve = np.mean([err[o].mean(1) for o in O], 0)
    e = {(k, o): first200(f"uniform_{k}")[o]["mpvpe"].numpy().mean(1) for k in METHODS for o in O}
    common = np.mean([mean_rank({o: e[k, o] for o in O}) for k in METHODS], 0)
    te = np.mean([z[f"b0_{o}_e"].mean(1) for o in O], 0)
    rows = {"rarity": np.concatenate([r, z["r"]]), "loss, all noise": loss.mean(1), "loss, low noise": loss[:, :4].mean(1),
            "loss, high noise": loss[:, -4:].mean(1), "disagreement of three seeds": dis,
            "-log p": np.concatenate([-ll, np.full(300, np.nan)])}
    print("| signal | error, 1000 valid | beyond rarity | common difficulty, 200 valid | beyond rarity | error, 300 train | vs rarity (valid / train) |")
    for k, a in rows.items():
        b = a[1000:]
        train = "-" if np.isnan(b).any() else f"{rho(b, te):.2f}"
        with_r = f"{rho(a[:1000], r):.2f} / {'-' if np.isnan(b).any() else f'{rho(b, z[chr(114)]):.2f}'}"
        print(f"| {k} | {rho(a[:1000], ve):.2f} | {rho(residual(a[:1000], r), residual(ve, r)):.2f} | {rho(a[:200], common):.2f} | "
              f"{rho(residual(a[:200], r[:200]), residual(common, r[:200])):.2f} | {train} | {with_r} |")


def maxvol(f, tol=1.05, iterations=6000):
    """Rows [p] of f [N, p] with the largest volume (Goreinov et al.), and max |f A^-1| per row (gamma)."""
    import scipy.linalg as sl

    sub = np.random.default_rng(0).choice(len(f), min(40000, len(f)), False)
    sel = sub[sl.qr(f[sub].T, pivoting=True, mode="economic")[2][:f.shape[1]]].copy()
    c = sl.solve(f[sel].T, f.T).T
    for it in range(iterations):
        i, j = np.unravel_index(np.abs(c).argmax(), c.shape)
        v = c[i, j]
        if abs(v) < tol:
            break
        row = c[i].copy()
        row[j] -= 1
        c -= np.outer(c[:, j] / v, row)
        sel[j] = i
    return sel, np.abs(c).max(1), it


def dopt(_):
    import scipy.linalg as sl

    from hp.models.rsgm import global_rotations
    from hp.models.so3_harmonics import pair_wigner_features, wigner_features

    torch.set_num_threads(4)
    _, xr = reference()
    xv, r, _ = validation()
    idx, _ = train_index()
    xt, z = train_rotations(idx)[:, R.JOINTS], np.load(OUT / "train_sample.npz")
    rc = torch.load(E / "data_weighting/rarity.pt", weights_only=True).numpy()[::R.REF_STRIDE]
    err, _ = errors("uniform")
    ve, te = mean_rank({o: err[o].mean(1) for o in O}), mean_rank({o: z[f"b0_{o}_e"].mean(1) for o in O})

    def features(x, kind):
        parts = []
        for b in x.float().split(8192):
            f = [wigner_features(b, (1,) if kind == "D1" else (1, 2))]
            if kind == "D1+D2+pairs":
                f.append(pair_wigner_features(global_rotations(b), (1,)))
            parts.append(torch.cat(f, 1))
        f = torch.cat(parts)
        return torch.cat([torch.ones(len(f), 1), f], 1).double().numpy()

    ess = lambda w: w.sum() ** 2 / (w ** 2).sum() / len(w)
    for kind in ("D1", "D1+D2", "D1+D2+pairs"):
        f, fv, ft = (features(x, kind) for x in (xr, xv, xt))
        lam, u = np.linalg.eigh(f.T @ f / len(f))
        keep = lam > 1e-5 * lam.mean()
        w = u[:, keep] / np.sqrt(lam[keep] + 1e-4 * lam.mean())
        q, qv, qt = f @ w, fv @ w, ft @ w
        h, hv, ht = ((a ** 2).sum(1) for a in (q, qv, qt))
        sel, gam, it = maxvol(q)
        a = q[sel]
        gv, gt = (np.abs(sl.solve(a.T, b.T).T).max(1) for b in (qv, qt))
        print(f"\n{kind}: {f.shape[1]} features, {keep.sum()} kept, {it} MaxVol swaps")
        print(f"  error rank, valid / train: leverage {rho(hv, ve):.2f} / {rho(ht, te):.2f}, gamma {rho(gv, ve):.2f} / "
              f"{rho(gt, te):.2f}, rarity {rho(r, ve):.2f} / {rho(z['r'], te):.2f}; beyond rarity (valid): leverage "
              f"{rho(residual(hv, r), residual(ve, r)):.2f}, gamma {rho(residual(gv, r), residual(ve, r)):.2f}")
        print(f"  gamma of the validation poses: > 1.05 in {(gv > 1.05).mean():.3f}, max {gv.max():.2f}; leverage, "
              f"training: max / median {h.max() / np.median(h):.0f}; vs rarity {rho(h[::5], rc[::5]):.2f}")
        lo, hi = 0.0, 5.0
        for _ in range(50):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if ess(np.clip((h / np.median(h)) ** m, .1, 10)) > .7 else (lo, m)
        wl, wr = np.clip((h / np.median(h)) ** m, .1, 10), np.clip((rc / np.median(rc)) ** 1.163, .1, 10)
        top = lambda v: v[np.argsort(-v)[:len(v) // 10]].sum() / v.sum()
        print(f"  weights at ESS 0.70: share of the top 10% leverage {top(wl):.2f}, rarity {top(wr):.2f}; Spearman "
              f"{rho(wl[::5], wr[::5]):.2f}")
        v = np.ones(len(h))
        for k in range(1, 21):
            v /= v.mean()
            l = np.linalg.cholesky((q * v[:, None]).T @ q / len(q) + 1e-9 * np.eye(q.shape[1]))
            lev = (sl.solve_triangular(l, q.T, lower=True) ** 2).sum(0)
            if k in (1, 2, 5, 20):
                print(f"  D-optimal weights before multiplicative step {k}: ESS {ess(v):.3f}", flush=True)
            v = v * lev / q.shape[1]


def study(_):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    path = E / "data_weighting/optuna/rdw_inpaint.db"
    s = optuna.load_study(study_name="rdw_inpaint", storage=f"sqlite:///{path}")
    keys = ["generation.wasserstein", "generation.precision", "generation.fid_val", "generation.fid_own",
            "generation.log_likelihood", "inpaint.hv_best", "inpaint.band_mean", "inpaint.best", "inpaint.mean", "ess"]
    a = {k: np.array([t.user_attrs[k] for t in s.trials]) for k in keys}
    w, hv = a[keys[0]], a["inpaint.hv_best"]
    near = w < 14.65
    print(f"near-uniform trials {np.flatnonzero(near)} (W < 14.65): hv_best {hv[near].mean():.2f}, sd {hv[near].std(ddof=1):.2f}")
    for k in keys[:-1]:
        print(f"  {k}: near-uniform mean {a[k][near].mean():.4f}, trial 12 {a[k][12]:.4f}, trial 12 - mean "
              f"{a[k][12] - a[k][near].mean():+.4f}; trial 0 {a[k][0]:.4f}")
    ok = np.flatnonzero(w <= w[0] + 0.5)
    pick = ok[np.argmax(hv[ok])]
    print(f"rule (Wasserstein <= {w[0] + 0.5:.2f}): trials {ok}, best hv_best: trial {pick} "
          f"({ {k: round(v, 3) for k, v in s.trials[pick].params.items()} }), W {w[pick]:.2f}, hv_best {hv[pick]:.2f}")
    q = np.polyfit(w, hv, 2)
    print(f"hv_best against W: Spearman {rho(w, hv):.2f}, quadratic fit residual sd {(hv - np.polyval(q, w)).std(ddof=3):.2f}; "
          f"fit at W 14.52 / 15.02 / 15.5 / 17: {np.polyval(q, [14.52, 15.02, 15.5, 17]).round(1)}")


def test(_):
    import yaml

    from hp.evaluation.completion import cache_path
    from hp.evaluation.fronts import APD_RANGE, hypervolume

    points, cache = yaml.safe_load(open("config/completion_test_points.yaml")), E / "completion/cache"
    print("| front | points | band mean, test (mm) | hv best of 10, test | band mean, valid (mm) | hv best of 10, valid |")
    names = [f"RDW {b} {m}" for m in ("dflow", "inpaint", "dposer") for b in ("b0", "b3p3")]
    for name in names + ["DPoser dposer", "DPoser inpaint", "RSGM inpaint", "RSGM dflow", "PoseD-Flow dflow"]:
        rows, valid = [], []
        for p in points[name]:
            m = [torch.load(cache_path(cache, p["label"], o).with_name(f"{o}_metrics.pt"), weights_only=True)["plain"]
                 for o in O]
            assert all(len(x["apd"]) == 5000 for x in m)
            rows.append(dict(mean=float(np.mean([x["mpvpe"].double().mean(1).mean() for x in m])),
                             best_of_k=float(np.mean([x["mpvpe"].double().min(1).values.mean() for x in m])),
                             apd=float(np.mean([x["apd"].double().mean() for x in m]))))
            valid.append(p["validation"])
        width = APD_RANGE[1] - APD_RANGE[0]
        cells = [f"{115 - hypervolume(x, 'mean') / width:.2f} | {hypervolume(x, 'best_of_k'):.2f}" for x in (rows, valid)]
        print(f"| {name} | {len(rows)} | {cells[0]} | {cells[1]} |")


def denoising(_):
    rows = {}
    for beta, folder in (("0", "denoising_b0"), ("3.3", "denoising_b3p3")):
        for line in (E / f"pairs_extra/{folder}/results.jsonl").read_text().splitlines():
            x = json.loads(line)
            if x["label"].startswith("ctrl"):
                rows[beta, x["label"][-3:], x["sigma"]] = (x["mpjpe"], x["mpvpe"])
    for seed in ("s10", "s11"):
        for sigma in (0.04, 0.1):
            a, b = rows["0", seed, sigma], rows["3.3", seed, sigma]
            print(f"{seed} {sigma * 1000:.0f} mm: MPJPE {a[0]:.2f} -> {b[0]:.2f} ({b[0] - a[0]:+.2f}), "
                  f"MPVPE {a[1]:.2f} -> {b[1]:.2f} ({b[1] - a[1]:+.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["difficulty", "train-sample", "train-report", "signals", "dopt", "study", "test",
                                         "denoising"])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    {"difficulty": difficulty, "train-sample": train_sample, "train-report": train_report, "signals": signals,
     "dopt": dopt, "study": study, "test": test, "denoising": denoising}[args.what](args)


if __name__ == "__main__":
    main()
