"""The rule of the pairs x data weighting gate (section 20 of the score networks report) on its studies.

    python explore/evaluation/pairs_gate/compare.py [study dir] [front dir]

Defaults: ~/data/experiments/pairs_gate/optuna and ~/data/experiments/completion/optuna. Control seed 10 comes from the
fronts, all other rows from the gate. Printed: the reproduction check, pairs - control per beta with the rule, the same
differences corrected for the APD (post hoc, not a part of the rule), beta 3.3 - beta 0 per model, minutes per point.
"""
import os
import sys

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
GATE = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/data/experiments/pairs_gate/optuna")
FRONT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/data/experiments/completion/optuna")
OCC = ("left_leg", "legs", "arms", "trunk")
METHODS = ("inpaint", "sds", "dflow")
POINTS = {"b0": {"inpaint": [99, 111], "sds": [28, 34, 85, 107], "dflow": [53, 47]},
          "b3p3": {"inpaint": [55, 16], "sds": [23, 30, 71, 114], "dflow": [0, 34]}}


def trials(folder, name):
    path = f"{folder}/{name}.db"
    if not os.path.isfile(path):
        return None
    study = optuna.load_study(study_name=name, storage=f"sqlite:///{path}")
    return [t for t in study.trials if t.state.name == "COMPLETE"]


def row(t):  # mean, best of 10 (mm), APD (cm), then the mean of every occlusion
    a = lambda k: [t.user_attrs[f"{o}_{k}"] for o in OCC]
    return [np.mean(a("mpvpe_mean")), np.mean(a("mpvpe_min")), np.mean(a("apd"))] + a("mpvpe_mean")


def front_rows(beta, method):
    by = {t.number: t for t in trials(FRONT, f"rsgm_{method}_mean_apd_rdw_{beta}")}
    return [(by[n].params, row(by[n])) for n in POINTS[beta][method]]


def gate_trials(beta, name, method, front):
    ts = trials(GATE, f"gate_{beta}_{name}_{method}") or []
    out = [next((t for t in ts if t.params == p), None) for p, _ in front]
    return None if any(t is None for t in out) else out


def gate_rows(*args):
    ts = gate_trials(*args)
    return None if ts is None else np.array([row(t) for t in ts])


def fmt(v):
    return " / ".join(f"{x:+.2f}" for x in v[:3])


for method in METHODS:  # reproduction of seed 10 with this code
    for p, r in front_rows("b0", method):
        for t in trials(GATE, f"gate_b0_repro_s10_{method}") or []:
            if t.params == p:
                print(f"repro {method}: {row(t)[0]:.2f} / {row(t)[2]:.2f}, stored {r[0]:.2f} / {r[2]:.2f}, "
                      f"{'ok' if abs(row(t)[0] - r[0]) <= 0.1 else 'DIFFERS'}")

rows = {}
for beta in POINTS:
    print(f"\n== {beta}: pairs - control (d mean / d best of 10 / d APD; left_leg legs arms trunk)")
    passed = known = 0
    for method in METHODS:
        front = front_rows(beta, method)
        ctrl = {10: np.array([r for _, r in front]), 11: gate_rows(beta, "ctrl_s11", method, front)}
        pairs = {s: gate_rows(beta, f"pairs_s{s}", method, front) for s in (10, 11)}
        rows[beta, method] = {"control": ctrl, "pairs": pairs}
        if ctrl[11] is not None:
            print(f"{method:8s} control s11 - s10: {fmt((ctrl[11] - ctrl[10]).mean(0))}")
        d = {s: (pairs[s] - ctrl[s]).mean(0) for s in (10, 11) if pairs[s] is not None and ctrl[s] is not None}
        for s, v in d.items():
            print(f"{method:8s} s{s}: {fmt(v)}; " + " ".join(f"{x:+.2f}" for x in v[3:]))
        if len(d) < 2:
            continue
        known += 1
        m = (d[10] + d[11]) / 2
        ok = m[0] <= -1.0 and m[2] >= -0.2 and m[1] <= 0.3 and d[10][0] < 0 and d[11][0] < 0
        passed += ok
        print(f"{method:8s} both: {fmt(m)}; " + " ".join(f"{x:+.2f}" for x in m[3:]) + f" -> {'pass' if ok else 'no'}")
        # post hoc: the slope of the mean over APD between the first and last point of the control of the same seed
        slope = {s: (ctrl[s][-1, 0] - ctrl[s][0, 0]) / (ctrl[s][-1, 2] - ctrl[s][0, 2]) for s in (10, 11)}
        res = {s: d[s][0] - slope[s] * d[s][2] for s in (10, 11)}
        print(f"{method:8s} APD corrected d mean: s10 {res[10]:+.2f}, s11 {res[11]:+.2f}, both "
              f"{(res[10] + res[11]) / 2:+.2f} (slope {slope[10]:.1f}, {slope[11]:.1f} mm/cm)")
        for name, n in (("control", "ctrl_s11"), ("pairs s10", "pairs_s10"), ("pairs s11", "pairs_s11")):
            minutes = [t.user_attrs["minutes"] for t in gate_trials(beta, n, method, front)]
            print(f"{method:8s} minutes per point, {name}: {np.mean(minutes):.1f}")
    print(f"{beta}: {passed} of {known} methods pass" + ("" if known < 3 else
          f" -> pairs {'PASS' if passed >= 2 else 'do NOT pass'}"))

print("\n== beta 3.3 - beta 0, each at the settings of its own front (points paired by their order)")
for method in METHODS:
    for kind in ("control", "pairs"):
        a, b = (rows[k, method][kind] for k in ("b0", "b3p3"))
        both = []
        for s in (10, 11):
            if a[s] is not None and b[s] is not None:
                both.append((b[s] - a[s]).mean(0))
                print(f"{method:8s} {kind:8s} s{s}: {fmt(both[-1])}; " + " ".join(f"{x:+.2f}" for x in both[-1][3:]))
        if len(both) == 2:
            print(f"{method:8s} {kind:8s} both: {fmt(np.mean(both, 0))}")

print("\n== cells: mean / best of 10 / APD over the points of a method (seeds 10 and 11 averaged)")
for (beta, method), r in rows.items():
    for kind in ("control", "pairs"):
        v = [r[kind][s].mean(0)[:3] for s in (10, 11) if r[kind][s] is not None]
        print(f"{beta:5s} {method:8s} {kind:8s} " + " / ".join(f"{x:.2f}" for x in np.mean(v, 0)))
