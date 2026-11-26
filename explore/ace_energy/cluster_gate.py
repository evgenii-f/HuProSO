"""The one-point gate of the cluster block of RSGM (branch experimental/cluster-block, `rdw_clusters`;
docs/experiments/2026-11-23_ace_descriptors.md, section 2026-11-26: ideas from ACE for the network). The plan and the
rule, written before the runs, are in ~/data/experiments/clusters_gate/PLAN.md. From the root:

    python explore/ace_energy/cluster_gate.py configs --tuning ../HuProSO-cluster/config/completion_tuning
                                                         # the configs of scripts/tune_completion.py, from the front
    python explore/ace_energy/cluster_gate.py compare    # the block against fact_RDW at the same point

The point is inpainting t99 of the validation front of fact_RDW at beta 0 (1000 poses, K = 10). `configs` writes the
reproduction of fact_RDW s10 and the block of seeds 10 and 11 at it; `compare` reads the gate studies, the front and the
control of seed 11 of the pairs gate, and applies the rule: d mean <= -1.0 mm, d APD >= -0.2 cm, d best of 10 <= +0.3 mm.
"""
import argparse
from pathlib import Path

import numpy as np
import optuna
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "config" / "completion_tuning"
FRONTS = Path("~/data/experiments/completion/optuna").expanduser()
GATE = Path("~/data/experiments/clusters_gate").expanduser()
CONTROL = Path("~/data/experiments/pairs_gate/optuna").expanduser()
FRONT, POINT = "rsgm_inpaint_mean_apd_rdw_b0", 99
OCCLUSIONS = ("left_leg", "legs", "arms", "trunk")
MODELS = {"repro_s10": "~/data/experiments/factorial/rsgm_posedflow_fact_RDW_s10/run_0001",
          **{f"clusters_s{s}": f"~/data/experiments/clusters/rsgm_posedflow_rdw_clusters_s{s}/run_0001" for s in (10, 11)}}


def trials(folder: Path, name: str):
    path = folder / f"{name}.db"
    if not path.is_file():
        return []
    return optuna.load_study(study_name=name, storage=f"sqlite:///{path}").trials


def row(t) -> np.ndarray:
    """mean, best of 10 (mm), APD (cm), then the mean of every occlusion."""
    a = lambda k: [t.user_attrs[f"{o}_{k}"] for o in OCCLUSIONS]  # noqa: E731
    return np.array([np.mean(a("mpvpe_mean")), np.mean(a("mpvpe_min")), np.mean(a("apd"))] + a("mpvpe_mean"))


def configs(tuning: Path) -> None:
    config = yaml.safe_load(open(tuning / f"{FRONT}.yaml"))
    params = dict({t.number: t for t in trials(FRONTS, FRONT)}[POINT].params)
    for name, model in MODELS.items():
        out = dict(config, model=model, n_trials=1, first=[params])
        yaml.safe_dump(out, open(GATE / f"gate_b0_{name}_inpaint.yaml", "w"), sort_keys=False)
        print("wrote", GATE / f"gate_b0_{name}_inpaint.yaml")


def compare() -> None:
    front = {t.number: t for t in trials(FRONTS, FRONT)}[POINT]
    rows = {"fact_RDW s10 (front)": front}
    for name in MODELS:
        done = [t for t in trials(GATE / "optuna", f"gate_b0_{name}_inpaint") if t.state.name == "COMPLETE"]
        rows.update({name: t for t in done if t.params == front.params})
    rows.update({"fact_RDW s11 (pairs gate)": t for t in trials(CONTROL, "gate_b0_ctrl_s11_inpaint")
                 if t.state.name == "COMPLETE" and t.params == front.params})
    print("| model | mean | best of 10 | APD | " + " | ".join(OCCLUSIONS) + " |")
    for name, t in rows.items():
        print(f"| {name} | " + " | ".join(f"{v:.2f}" for v in row(t)) + " |")
    for seed, control in ((10, "fact_RDW s10 (front)"), (11, "fact_RDW s11 (pairs gate)")):
        if f"clusters_s{seed}" in rows and control in rows:
            d = row(rows[f"clusters_s{seed}"]) - row(rows[control])
            ok = d[0] <= -1.0 and d[2] >= -0.2 and d[1] <= 0.3
            print(f"block - control, seed {seed}: " + " / ".join(f"{v:+.2f}" for v in d) + (" -> pass" if ok else " -> no"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("configs", "compare"))
    parser.add_argument("--tuning", type=Path, default=CONFIGS,
                        help="config/completion_tuning of a checkout with the rdw studies (origin/dev has them)")
    args = parser.parse_args()
    configs(args.tuning) if args.what == "configs" else compare()
