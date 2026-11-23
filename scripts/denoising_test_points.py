"""The rows of the motion denoising test, with where their settings come from and their validation numbers, written to
config/denoising_test_points.yaml for scripts/evaluate.py denoising --points:

    python scripts/denoising_test_points.py              # write the file
    python scripts/denoising_test_points.py --dry-run    # print it
    python scripts/denoising_test_points.py --check      # exit 1 if the file differs from what would be written

The rows are those run on the test of docs/experiments/2026-11-16_motion_denoising_rsgm_vs_dposer.md, each by a rule
written before its test (the configs of config/denoising_tuning); the rows run by mistake or after the fact (first
guess; lik / ln) are not points. A point keeps the label and the spec of its row of evaluate_denoising.MODELS, so its
test is in the cache already, and a new split or dataset (e.g. HPS) runs the same rows with one command.

Per point: the group (--groups of evaluate.py), the role, the source of the settings per noise level (the authors, or
the study and trial), and the validation MPJPE / MPVPE per noise level: of the row itself where its validation run is
cached, otherwise of the best trial of the study (on the network of the study, named in "of").
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_denoising import CACHE, MODELS  # noqa: E402

from hp.evaluation.denoising import SIGMAS  # noqa: E402

OUT = ROOT / "config" / "denoising_test_points.yaml"
STUDY_DIR = "~/data/experiments/denoising/optuna"


def study(name: str) -> dict:
    """The source of a row's settings: the best trial of a study per noise level, e.g. rsgm_sds_<mm>mm."""
    return {"study": {sigma: f"{name}_{round(sigma * 1000)}mm" for sigma in SIGMAS}}


AUTHORS = {"source": "the authors' settings"}
POINTS = {
    "baseline": [
        ("w/o prior", "DPoser's optimization without the prior term", {"source": "DPoser's settings"}),
        ("PoseD-Flow: their checkpoint", "PoseD-Flow, D-Flow", AUTHORS),
    ],
    "DPoser": [
        ("DPoser: their full, DPoser method", "DPoser, its own optimization", AUTHORS),
        ("DPoser: their full, DPoser method, tuned", "DPoser tuned like RSGM (rule of dposer_sds_<mm>mm.yaml)",
         study("dposer_sds")),
        ("DPoser: their full, D-Flow", "DPoser through D-Flow", {"source": "PoseD-Flow's settings"}),
    ],
    "RSGM sds": [
        ("RSGM 1536x6: their c10, DPoser method", "ours: the RSGM row with the lowest validation MPJPE at the tuned "
         "settings (rule of rsgm_sds_<mm>mm.yaml)", study("rsgm_sds")),
        ("RSGM 512x4: their c10, DPoser method", "RSGM at the tuned settings (the network they were tuned on)",
         study("rsgm_sds")),
        ("RSGM 512x4: their full, DPoser method", "RSGM at the tuned settings", study("rsgm_sds")),
        ("RSGM res_norm: their c10, DPoser method", "RSGM at the tuned settings", study("rsgm_sds")),
        ("RSGM abl residual: their c10, DPoser method", "RSGM at the tuned settings (declared extra row)",
         study("rsgm_sds")),
        ("RSGM 512x4: their c10, DPoser method, DPoser's steps", "the tuned settings at DPoser's own steps "
         "(rule of rsgm_sds_<mm>mm.yaml)", study("rsgm_sds")),
        ("RSGM 1536x6: their c10, DPoser method, DPoser's tuned budget", "RSGM at the tuned DPoser's steps (rule of "
         "rsgm_sds_budget_<mm>mm.yaml)", study("rsgm_sds_budget")),
    ],
    "RSGM dflow": [
        ("RSGM 512x4: their c10, D-Flow", "RSGM through D-Flow", {"source": "PoseD-Flow's settings"}),
    ],
}


def best_trials(name: str) -> dict:
    """The best trial of a study file: number, MPJPE, MPVPE and the network."""
    from hp.evaluation.denoising_results import load_study

    trials = load_study(Path(STUDY_DIR).expanduser() / f"{name}.db")
    best = min(trials, key=lambda r: r["mpjpe"])
    return {"trial": best["trial"], "mpjpe": round(best["mpjpe"], 3), "mpvpe": round(best["mpvpe"], 3),
            "of": f"{name} trial {best['trial']} ({best['model']})"}


def points(cache: str = CACHE) -> dict:
    from hp.evaluation.denoising_results import find, load_cache

    records = load_cache(cache, "valid", models=MODELS)
    out = {}
    for group, rows in POINTS.items():
        out[group] = []
        for label, role, source in rows:
            point = {"label": label, "role": role}
            studies = source.get("study")
            point["source"] = ({sigma: f"{name} trial {best_trials(name)['trial']}" for sigma, name in studies.items()}
                               if studies else source["source"])
            validation = {}
            for sigma in SIGMAS:
                r = find(records, label, sigma, "valid")
                if r is not None:
                    validation[sigma] = {"mpjpe": round(r["mpjpe"], 3), "mpvpe": round(r["mpvpe"], 3), "of": "this row"}
                elif studies:
                    best = best_trials(studies[sigma])
                    validation[sigma] = {"mpjpe": best["mpjpe"], "mpvpe": best["mpvpe"], "of": best["of"]}
            point["validation"] = validation
            point["spec"] = MODELS[label]
            out[group].append(point)
    return out


HEADER = """# The rows of the motion denoising test (scripts/denoising_test_points.py; do not edit by hand), run by
# scripts/evaluate.py denoising --points config/denoising_test_points.yaml [--groups ...]. Each was run once on the
# test by a rule written before it (docs/experiments/2026-11-16_motion_denoising_rsgm_vs_dposer.md). Validation:
# MPJPE / MPVPE (mm) on the 100 validation chunks, of the row itself or of the best trial of its study ("of").
"""


def text(cache: str = CACHE) -> str:
    return HEADER + yaml.safe_dump(points(cache), sort_keys=False, width=120, allow_unicode=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the file, write nothing")
    parser.add_argument("--check", action="store_true", help="exit 1 if the file differs from what would be written")
    parser.add_argument("--cache", default=CACHE)
    args = parser.parse_args(argv)
    new = text(args.cache)
    if args.dry_run:
        print(new)
        return 0
    if args.check:
        same = OUT.is_file() and OUT.read_text() == new
        print(f"{OUT}: {'up to date' if same else 'differs'}")
        return 0 if same else 1
    OUT.write_text(new)
    print(f"wrote {OUT}: {sum(len(v) for v in POINTS.values())} rows in {len(POINTS)} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
