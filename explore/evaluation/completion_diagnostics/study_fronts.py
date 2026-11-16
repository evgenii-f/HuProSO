"""The mean-APD Optuna studies of completion side by side: for every study, the lowest mean MPVPE among its finished
trials with APD at least a level (a reading of the front of mean and APD at fixed diversity), the point of the rule of
the studies (APD >= 6.40 cm, the lowest mean; else the highest APD of the front), and the front points in a band of
APD. Validation poses only (those of the studies). Run from the root:

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/study_fronts.py
    PYTHONPATH=src python explore/evaluation/completion_diagnostics/study_fronts.py dposer_c10_method_mean_apd \\
        --band 5 7.1

Mean and best of K in mm, APD in cm, each the mean over the four occlusions (tune_completion.averages).
"""
import argparse
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from tune_completion import STUDY_DIR, averages, open_study  # noqa: E402

STUDIES = [
    "rsgm_inpaint_mean_apd_residual",
    "rsgm_inpaint_mean_apd_norm",
    "rsgm_inpaint_mean_apd_deep",
    "rsgm_inpaint_mean_apd_search_lik_full",
    "dposer_c10_inpaint_mean_apd",
    "dposer_c10_method_mean_apd",
]
LEVELS = (4.5, 5.5, 6.0, 6.4, 7.0)
RULE_APD = 6.40  # D-Flow Heun EDM 50 of small std c10 on the same poses


def fmt(a: dict) -> str:
    return f"{a['mean']:.2f} / {a['best_of_k']:.2f} / {a['apd']:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("studies", nargs="*", default=STUDIES)
    parser.add_argument("--study-dir", default=STUDY_DIR)
    parser.add_argument("--band", nargs=2, type=float, default=None, help="also list the front points in this APD band")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_dir = Path(args.study_dir).expanduser()
    print("| study | trials | " + " | ".join(f"APD >= {level}" for level in LEVELS) + " | rule point |")
    print("|---" * (len(LEVELS) + 3) + "|")
    bands = []
    for name in args.studies:
        study = open_study(name, study_dir)
        done = [(averages(t), t) for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        cells = []
        for level in LEVELS:
            above = [a for a, _ in done if a["apd"] >= level]
            cells.append(f"{min(a['mean'] for a in above):.2f}" if above else "-")
        front = [(averages(t), t) for t in study.best_trials]
        chosen = [x for x in front if x[0]["apd"] >= RULE_APD]
        a, t = min(chosen, key=lambda x: x[0]["mean"]) if chosen else max(front, key=lambda x: x[0]["apd"])
        rule = f"t{t.number}: {fmt(a)}" + ("" if chosen else " (none >= 6.40)")
        print(f"| {name} | {len(done)} | " + " | ".join(cells) + f" | {rule} |")
        if args.band:
            low, high = args.band
            points = sorted((x for x in front if low <= x[0]["apd"] <= high), key=lambda x: x[0]["apd"])
            bands.append((name, [f"  t{t.number}: {fmt(a)}, {t.params}" for a, t in points]))
    for name, lines in bands:
        print(f"\n{name}, front points with APD in {args.band}:")
        print("\n".join(lines) or "  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
