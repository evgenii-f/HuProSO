"""The Optuna studies of completion on one plot: every finished trial as a faint point and the Pareto front of each
study, mean MPVPE against APD (accuracy against diversity) and best of 10 against APD. Validation poses only (those of
the studies, 1000). The rule points of the studies (APD >= 6.40 cm, the lowest mean) are stars; the dotted line is
the APD of D-Flow Heun EDM 50 on RSGM small std c10, the threshold of the rule. Run from the root:

    PYTHONPATH=src python explore/evaluation/completion_diagnostics/plot_study_fronts.py --out fronts.png --bands
    PYTHONPATH=src python explore/evaluation/completion_diagnostics/plot_study_fronts.py --apd 4 8 --mean 80 100

Mean and best of K in mm, APD in cm, each the mean over the four occlusions (tune_completion.averages).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import optuna  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from tune_completion import STUDY_DIR, averages, open_study  # noqa: E402

from hp.evaluation.fronts import front  # noqa: E402
from hp.visualization.fronts import draw_front  # noqa: E402

STUDIES = {  # name: (label, colour, marker)
    "rsgm_inpaint_mean_apd_residual": ("RSGM abl residual, inpainting", "tab:blue", "o"),
    "rsgm_inpaint_mean_apd_norm": ("RSGM abl norm, inpainting", "tab:cyan", "o"),
    "rsgm_inpaint_mean_apd_norm_aniso": ("RSGM abl norm aniso, inpainting", "navy", "P"),
    "rsgm_inpaint_mean_apd_deep": ("RSGM deep, inpainting", "tab:purple", "o"),
    "rsgm_inpaint_mean_apd_search_lik_full": ("RSGM 1536x6 lik full, inpainting", "tab:gray", "o"),
    "rsgm_inpaint_pareto": ("RSGM small, inpainting (mean / best of 10)", "tab:green", "o"),
    "dposer_c10_inpaint_mean_apd": ("DPoser c10, inpainting", "tab:red", "s"),
    "dposer_c10_method_mean_apd": ("DPoser c10, DPoser's method", "tab:orange", "^"),
    "rsgm_inpaint_mean_apd_full_dposer_like": ("RSGM dposer-like full, inpainting", "tab:olive", "D"),
    "rsgm_inpaint_mean_apd_full_dposer_like_aniso": ("RSGM dposer-like aniso full, inpainting", "darkgreen", "P"),
    "rsgm_inpaint_mean_apd_full_small":("RSGM small full, inpainting", "tab:brown", "D"),
    "dposer_full_method_mean_apd": ("DPoser full, DPoser's method", "tab:pink", "v"),
    "dposer_full_inpaint_mean_apd": ("DPoser full, inpainting", "black", "s"),
}
RULE_APD = 6.40  # D-Flow Heun EDM 50 of small std c10 on the same poses
PANELS = (("mean", "mean MPVPE (mm) ↓"), ("best_of_k", "best of 10 MPVPE (mm) ↓"))


BANDS = ((5.5, 6.4), (6.0, 7.0), (6.4, 7.5), (6.4, 8.5), (7.0, 8.5))  # of APD (cm), for --bands


def band_mean(rows: list, y: str, low: float, high: float, points: int = 2001) -> float:
    """The front of a study averaged over the APD band [low, high]: f(a), the lowest y among the trials with APD at
    least a (the staircase of the plot), integrated over the band and divided by its width; the hypervolume of the
    front in the band, in mm. f is defined on the whole band as long as a trial has APD >= high."""
    apd, value = np.array([r["apd"] for r in rows]), np.array([r[y] for r in rows])
    grid = np.linspace(low, high, points)
    f = np.array([value[apd >= a].min() if (apd >= a).any() else np.nan for a in grid])
    return float(np.trapezoid(f, grid) / (high - low))


def load(name: str, study_dir: Path) -> list:
    study = open_study(name, study_dir)
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append(dict(averages(t), number=t.number))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("studies", nargs="*", default=list(STUDIES))
    parser.add_argument("--study-dir", default=STUDY_DIR)
    parser.add_argument("--apd", nargs=2, type=float, default=(3.0, 9.0), help="APD range of the plots (cm)")
    parser.add_argument("--mean", nargs=2, type=float, default=(76.0, 105.0), help="mean range (mm)")
    parser.add_argument("--best", nargs=2, type=float, default=(44.0, 62.0), help="best of 10 range (mm)")
    parser.add_argument("--out", default="study_fronts.png")
    parser.add_argument("--bands", action="store_true",
                        help="also print the front of each study averaged over bands of APD (BANDS), a hypervolume in mm")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_dir = Path(args.study_dir).expanduser()
    limits = {"mean": args.mean, "best_of_k": args.best}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for name in args.studies:
        label, color, marker = STUDIES.get(name, (name, None, "o"))
        rows = load(name, study_dir)
        for ax, (y, _) in zip(axes, PANELS):
            ax.plot([r["apd"] for r in rows], [r[y] for r in rows], marker=marker, ls="", ms=4, alpha=0.25,
                    color=color, mec="none")
            points = front(rows, ("apd", "max"), (y, "min"))
            draw_front(ax, [r["apd"] for r in points], [r[y] for r in points], "steps", color=color, ls="-", lw=1.6,
                       corner="x")
        mean_front = front(rows, ("mean", "min"), ("apd", "max"))
        chosen = [r for r in mean_front if r["apd"] >= RULE_APD]
        if chosen:
            r = min(chosen, key=lambda r: r["mean"])
            for ax, (y, _) in zip(axes, PANELS):
                ax.plot(r["apd"], r[y], marker="*", ms=14, color=color, mec="black", mew=0.6, zorder=5)
        axes[0].plot([], [], marker=marker, ls="-", color=color, label=f"{label} ({len(rows)} trials)")
    for ax, (y, ylabel) in zip(axes, PANELS):
        ax.axvline(RULE_APD, ls=":", color="0.4", lw=1)
        ax.set_xlim(*args.apd)
        ax.set_ylim(*limits[y])
        ax.set_xlabel("APD (cm) →")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title("accuracy against diversity (validation, 1000 poses, K = 10)")
    axes[1].set_title("coverage against diversity; ★ rule point (APD ≥ 6.40, lowest mean)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)
    if args.bands:
        for y, _ in PANELS:
            print(f"\n{y}: the front averaged over bands of APD (mm)")
            print("| study | trials | " + " | ".join(f"APD {low}-{high}" for low, high in BANDS) + " |")
            print("|---" * (len(BANDS) + 2) + "|")
            for name in args.studies:
                rows = load(name, study_dir)
                cells = " | ".join(f"{band_mean(rows, y, low, high):.2f}" for low, high in BANDS)
                print(f"| {STUDIES.get(name, (name,))[0]} | {len(rows)} | {cells} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
