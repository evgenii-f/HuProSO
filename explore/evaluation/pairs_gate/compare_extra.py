"""The rule of ~/data/experiments/pairs_extra/PLAN.md on the harness and denoising rows of the pairs x data weighting
models (section 22 of docs/experiments/2026-11-14_score_network_architectures.md), validation:

    python explore/evaluation/pairs_gate/compare_extra.py [folder]          # default ~/data/experiments/pairs_extra

Printed: the reproduction of the stored fact_RDW rows, pairs - control with the rule, beta 3.3 - beta 0.
"""
import json
import sys
from pathlib import Path

import numpy as np

E = Path("~/data/experiments").expanduser()
F = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else E / "pairs_extra"
HARNESS = ("wasserstein", "recall", "precision")
KEYS = HARNESS + ("ll", "FID_val", "APD", "dNN")
MODELS = ("ctrl_s10", "ctrl_s11", "pairs_s10", "pairs_s11")


def values(result):
    h = result["harness"]
    out = {"wasserstein": h["wasserstein_deg"], "recall": h["recall_deg"], "precision": h["precision_deg"],
           "ll": h["log_likelihood"], **{k: result[k]["mean"] for k in ("FID_val", "APD", "dNN")}}
    return {k: float("nan") if v is None else v for k, v in out.items()}


def harness(beta):
    rows = {}
    for l in (F / f"harness_{beta}.jsonl").read_text().splitlines():
        r = json.loads(l)
        kind = "pairs" if "/pairs/" in r["run"] else "ctrl"
        seed = r["run"].split("/")[-2].rsplit("_s", 1)[-1]
        rows[f"{kind}_s{seed}", r["part"]] = values(r["result"])
    return rows


def denoising(beta):
    rows = {}
    for l in (F / f"denoising_{beta}" / "results.jsonl").read_text().splitlines():
        r = json.loads(l)
        rows[r["label"], r["sigma"]] = (r["mpjpe"], r["mpvpe"])
    return rows


stored = {}
for l in (E / "factorial/responses.jsonl").read_text().splitlines():
    r = json.loads(l)
    if r["cell"] == "fact_RDW" and r["part"].startswith("generation"):
        stored[f"ctrl_s{r['seed']}", r["part"]] = values(r["result"])

H = {b: harness(b) for b in ("b0", "b3p3")}
D = {b: denoising(b) for b in ("b0", "b3p3")}
print("reproduction (new - stored fact_RDW rows):")
for (m, p), v in stored.items():
    print(f"  {m} {p}: " + " ".join(f"{k} {H['b0'][m, p][k] - v[k]:+.4f}" for k in KEYS if v[k] == v[k]))
for b in H:
    print(f"\n== {b}: harness values")
    for p in ("generation", "generation_ode"):
        for m in MODELS:
            print(f"  {p:14s} {m:9s} " + " ".join(f"{k} {H[b][m, p][k]:.4f}" for k in KEYS if H[b][m, p][k] == H[b][m, p][k]))
    print(f"== {b}: pairs - control, seed 10 / 11 / mean")
    gains, harms, d = {}, {}, {}
    for p in ("generation", "generation_ode"):
        for k in KEYS:
            v = [H[b][f"pairs_s{s}", p][k] - H[b][f"ctrl_s{s}", p][k] for s in (10, 11)]
            if v[0] == v[0]:
                d[p, k] = v
                print(f"  {p:14s} {k:12s} {v[0]:+.4f} / {v[1]:+.4f} / {np.mean(v):+.4f}")
    limit = {"wasserstein": 0.15, "recall": 0.10, "precision": 0.15}
    good = [k for k in HARNESS if all(np.mean(d[p, k]) <= -limit[k] and max(d[p, k]) < 0 for p in ("generation", "generation_ode"))]
    bad = [k for k in HARNESS if all(np.mean(d[p, k]) >= limit[k] for p in ("generation", "generation_ode"))]
    any_bad = [k for k in HARNESS if any(np.mean(d[p, k]) >= limit[k] for p in ("generation", "generation_ode"))]
    ll = d["generation", "ll"]
    ll_gain = np.mean(ll) >= 0.5 and min(ll) > 0
    print(f"  rule harness {b}: gain {(len(good) >= 2 and not any_bad) or ll_gain} (two better: {good}), "
          f"harm {bool(bad) or np.mean(ll) <= -0.5} ({bad})")
    print(f"== {b}: denoising MPJPE / MPVPE, pairs - control, seed 10 / 11 / mean")
    dn = {}
    for s in sorted({k[1] for k in D[b]}):
        for i, name in enumerate(("MPJPE", "MPVPE")):
            v = [D[b][f"pairs_s{x}", s][i] - D[b][f"ctrl_s{x}", s][i] for x in (10, 11)]
            dn[s, name] = v
            print(f"  {round(s * 1000)} mm {name}: {v[0]:+.3f} / {v[1]:+.3f} / {np.mean(v):+.3f}")
    levels = sorted({k[0] for k in dn})
    gain = any(np.mean(dn[a, "MPJPE"]) <= -0.10 and max(dn[a, "MPJPE"]) < 0 and
               all(np.mean(dn[c, "MPJPE"]) <= 0.05 for c in levels if c != a) for a in levels)
    harm = any(min(dn[a, "MPJPE"]) >= 0.10 for a in levels)
    print(f"  rule denoising {b}: gain {gain}, harm {harm}")
print("\n== beta 3.3 - beta 0, mean of the two seeds")
for kind in ("ctrl", "pairs"):
    for p in ("generation", "generation_ode"):
        v = {k: np.mean([H["b3p3"][f"{kind}_s{s}", p][k] - H["b0"][f"{kind}_s{s}", p][k] for s in (10, 11)]) for k in KEYS}
        print(f"  {kind:6s} {p:14s} " + " ".join(f"{k} {x:+.4f}" for k, x in v.items() if x == x))
    for s in sorted({k[1] for k in D["b0"]}):
        v = [[D["b3p3"][f"{kind}_s{x}", s][i] - D["b0"][f"{kind}_s{x}", s][i] for x in (10, 11)] for i in (0, 1)]
        print(f"  {kind:6s} denoising {round(s * 1000)} mm MPJPE s10/s11/mean {v[0][0]:+.3f}/{v[0][1]:+.3f}/{np.mean(v[0]):+.3f} "
              f"MPVPE {v[1][0]:+.3f}/{v[1][1]:+.3f}/{np.mean(v[1]):+.3f}")
