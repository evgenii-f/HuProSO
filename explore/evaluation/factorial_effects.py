"""Effects of the full 2^5 factorial study (section 13 of docs/experiments/2026-11-14_score_network_architectures.md)
from the rows of scripts/evaluate_factorial.py. Run from the root of the repository:

    python explore/evaluation/factorial_effects.py ~/data/experiments/factorial/responses.jsonl
    python explore/evaluation/factorial_effects.py responses.jsonl --responses inpaint_mean FID_val --order 2

For every response: the main effects, the 2-way and the 3-way interactions (each = mean over the cells where the
product of the signs of its factors is +1, minus the mean where it is -1: the change from low to high), their
standard error from the 4- and 5-way interactions (assumed to be noise: SE = RMS of those six effects), and |t|.
Only the seed-10 cell of each of the 32 cells enters the effects; further seeds of a cell are listed as a direct
check of the noise. At the end the rule of section 13 picks a cell.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

FACTORS = "RNADW"
NAMES = {"R": "residual", "N": "LayerNorm", "A": "per-joint noise", "D": "dropout", "W": "width 1024"}
APD_MIN = 6.40  # the inpainting APD the rule asks for (cm)


def signs(cell: str) -> np.ndarray:
    """+1 / -1 of the five factors of a cell name fact_<letters on> (fact_0: all off)."""
    on = cell.removeprefix("fact_")
    return np.array([1.0 if f in on else -1.0 for f in FACTORS])


def effects(table: dict, max_order: int = 5) -> dict:
    """{interaction (e.g. "NA"): effect} of the responses table {cell: y} of all 32 cells."""
    cells = sorted(table)
    if len(cells) != 2 ** len(FACTORS):
        raise ValueError(f"need all {2 ** len(FACTORS)} cells, got {len(cells)}")
    x = np.stack([signs(c) for c in cells])
    y = np.array([table[c] for c in cells])
    out = {}
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(range(len(FACTORS)), order):
            s = x[:, combo].prod(1)
            out["".join(FACTORS[i] for i in combo)] = y[s > 0].mean() - y[s < 0].mean()
    return out


def responses(rows) -> dict:
    """{(cell, seed): {response: value}} from the rows of evaluate_factorial.py."""
    out = {}
    for r in rows:
        if r.get("smoke") or not r.get("cell"):
            continue
        d, res = out.setdefault((r["cell"], r["seed"]), {}), r["result"]
        if r["part"] == "inpaint":
            d.update(inpaint_mean=res["mean"], inpaint_best=res["best"], inpaint_apd=res["apd"])
        elif r["part"] == "generation":
            for k in ("FID_val", "FID_own", "APD", "dNN"):
                d[k] = res[k]["mean"]
            d["FID_val_half"] = res["FID_val"].get("ci95", math.nan)
            h = res["harness"]
            d.update(wasserstein=h["wasserstein_deg"], recall=h["recall_deg"], precision=h["precision_deg"],
                     log_likelihood=h["log_likelihood"] if h["log_likelihood"] is not None else math.nan)
        elif r["part"] == "hybrid":
            for occ, m in res["hybrid"].items():
                d[f"hybrid_{occ}"] = m["mpvpe_mean"]
            t = res["trajectory"]
            d.update(first_phase=t["left_leg_100"], drift=t["left_leg_199"] - t["left_leg_100"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl")
    parser.add_argument("--responses", nargs="*", default=None, help="default: all that every cell has")
    parser.add_argument("--order", type=int, default=3, help="the highest order of interaction printed")
    args = parser.parse_args()
    data = responses(json.loads(l) for l in Path(args.jsonl).expanduser().read_text().splitlines() if l.strip())
    main_cells = {cell: d for (cell, seed), d in data.items() if seed == 10}
    if not main_cells:
        print("no seed-10 rows of the factorial (smoke rows are left out)")
        return 1
    names = args.responses or sorted(set.intersection(*(set(d) for d in main_cells.values())) - {"FID_val_half"})
    for name in names:
        table = {c: d[name] for c, d in main_cells.items() if name in d}
        try:
            eff = effects(table)
        except ValueError as e:
            print(f"{name}: {e}")
            continue
        noise = [v for k, v in eff.items() if len(k) >= 4]
        se = math.sqrt(np.mean(np.square(noise)))
        print(f"\n{name}: grand mean {np.mean(list(table.values())):.3f}, SE of an effect {se:.3f} (from the 4- and 5-way)")
        for k, v in sorted(((k, v) for k, v in eff.items() if len(k) <= args.order), key=lambda kv: (len(kv[0]), kv[0])):
            mark = " *" if abs(v) > 2 * se else ""
            print(f"  {k:4s} {v:+9.3f}  |t| {abs(v) / se:5.1f}{mark}  {' x '.join(NAMES[f] for f in k)}")
        repeats = [(c, s, d[name]) for (c, s), d in data.items() if s != 10 and name in d]
        for c, s, v in repeats:
            print(f"  seed {s} of {c}: {v:.3f} (seed 10: {table.get(c, math.nan):.3f})")

    # the rule of section 13
    need = ("inpaint_mean", "inpaint_apd", "FID_val")
    if all(all(k in d for k in need) for d in main_cells.values()) and "fact_N" in main_cells:
        ref = main_cells["fact_N"]
        limit = ref["FID_val"] + (ref.get("FID_val_half") if not math.isnan(ref.get("FID_val_half", math.nan)) else 0)
        ok = {c: d for c, d in main_cells.items() if d["inpaint_apd"] >= APD_MIN and d["FID_val"] <= limit}
        print(f"\nrule: inpainting APD >= {APD_MIN}, FID_val <= {limit:.4f} (fact_N + its half-width): "
              f"{len(ok)} of {len(main_cells)} cells")
        for c, d in sorted(ok.items(), key=lambda kv: kv[1]["inpaint_mean"])[:5]:
            print(f"  {c:12s} inpainting {d['inpaint_mean']:.2f} / {d.get('inpaint_best', math.nan):.2f} / "
                  f"{d['inpaint_apd']:.2f}, FID_val {d['FID_val']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
