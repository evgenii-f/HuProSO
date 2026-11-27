"""Choose the points of the completion fronts that go to the test, by a rule fixed before the test, and write them to
config/completion_test_points.yaml for scripts/evaluate_completion.py --points:

    python scripts/select_test_points.py                       # all fronts, written to the file
    python scripts/select_test_points.py --fronts "RSGM inpaint" "DPoser dposer" --dry-run

THE RULE (written 2026-11-19, before any of these points ran on the test):
  - A front is one family of models (RSGM, DPoser, PoseD-Flow) with one method (inpaint, dflow, dposer): the union of
    the trials of all its Optuna studies, over all networks of the family.
  - Its candidates: the finished trials of the studies of scripts/tune_completion.py with a recorded configuration
    (1000 validation poses, K = 10), without the trials copied from another study (imported_from); each with mean
    MPVPE, best of 10 and APD averaged over the four occlusions. Studies without a recorded configuration (made by an
    older version of the script, whose fixed settings are not known) and the .jsonl rows are not candidates.
  - N = 10 thresholds a of APD evenly spaced from 4.0 to 8.5 cm (every 0.5 cm; the range where the fronts have points
    with a mean below about 105 mm; N chosen on validation in section 3a of completion_fronts.ipynb: the N points keep
    at least 96 % of the hypervolume of every whole front, at about 16 GPU hours of test); for each, the candidate with the lowest mean MPVPE among those with APD >= a (a
    corner of the staircase of the front; hp.evaluation.fronts.spanning). Thresholds above the largest APD of the
    front give nothing; points chosen for several thresholds are run once.
  - Every chosen point runs once on the 5000 test poses; nothing is chosen or changed after seeing the test.
  - The fronts are compared on the test by the hypervolume of their points (hp.evaluation.fronts.hypervolume): the area
    between the staircase of the points and the reference, over APD 4.0-8.5 cm, against mean MPVPE 115 mm and,
    apart, best of 10 65 mm (HV_REFERENCE); with the test values of the same points.

THE DATA-WEIGHTING FRONTS (added 2026-11-24, before any of their points ran on the test): the studies rsgm_*_rdw_<b>
(fact_RDW trained with uniform sampling, b0, and with rarity sampling, b3p3; inpaint, sds and dflow) are the families
"RDW b0" and "RDW b3p3", apart from RSGM, so the fronts above keep the points chosen for them. The rule for their points
is the one above, with one addition fixed here: on the test RDW b3p3 is better than RDW b0 in a method if the band mean
(115 mm - hypervolume / 4.5 cm) of its points is lower by at least 1 mm and its hypervolume of best of 10 is not
lower; worse if the band mean is higher by at least 1 mm; otherwise no difference. This is the rule that was fixed
before the validation fronts were looked at (2026-11-24); there D-Flow and SDS came out better and inpaint without a
difference.

A point whose spec equals that of a row of evaluate_completion.MODELS keeps the label of that row (its test is in the
cache already). The other labels: "<family> <run folder>, <study> t<trial>".
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_completion import MODELS  # noqa: E402
from tune_completion import RUN_KEYS, STUDY_DIR, averages, model_spec, values_of  # noqa: E402

from hp.evaluation.fronts import APD_RANGE, TEST_POINTS, spanning  # noqa: E402
OUT = Path(__file__).resolve().parents[1] / "config" / "completion_test_points.yaml"


WEIGHTING = re.compile(r"_rdw_(b[0-9p]+)$")  # the studies of data weighting: a family of their own (see above)


def family_of(model: str, study: str = "") -> str:
    match = WEIGHTING.search(study)
    if match:
        return f"RDW {match.group(1)}"
    if model.endswith(".ckpt"):
        return "PoseD-Flow"
    return "DPoser" if Path(model).parent.name.startswith("dposer") else "RSGM"


def spec_of(config: dict, params: dict) -> dict:
    """The spec of evaluate_completion for a trial: as tune_completion runs it."""
    values = values_of(config, optuna.trial.FixedTrial(params))
    spec = model_spec(config["model"])
    if config["method"] == "dflow":
        dflow = {key: values.pop(key) for key in RUN_KEYS if key in values and key != "ode_steps"}
        if values.get("ode_steps") == 100:  # the default of run(): left out, as in the rows of MODELS
            values.pop("ode_steps")
        return {**spec, "dflow": dflow, **({"prior": values} if values else {})}
    return {**spec, "method": config["method"], "prior": values}


def candidates(study_dir: Path) -> dict:
    """{(family, method): [point]} over every study with a recorded configuration."""
    out = defaultdict(list)
    for path in sorted(study_dir.glob("*.db")):
        for summary in optuna.get_all_study_summaries(f"sqlite:///{path}"):
            study = optuna.load_study(study_name=summary.study_name, storage=f"sqlite:///{path}")
            config = study.user_attrs.get("config")
            if not config or config.get("poses") != 1000 or config.get("k", 10) != 10:
                continue
            for t in study.trials:
                if t.state != optuna.trial.TrialState.COMPLETE or "imported_from" in t.user_attrs:
                    continue
                a = averages(t)
                if any(v != v for v in a.values()):  # nan: an occlusion missing
                    continue
                out[family_of(config["model"], summary.study_name), config["method"]].append(dict(
                    study=summary.study_name, trial=t.number, **{k: round(v, 3) for k, v in a.items()},
                    minutes=t.user_attrs.get("minutes"), host=t.user_attrs.get("host"),
                    spec=spec_of(config, t.params)))
    return out


def choose(points: list, n: int = TEST_POINTS) -> list:
    return [dict(p, thresholds=thresholds) for p, thresholds in spanning(points, n, *APD_RANGE)]


def label_of(family: str, point: dict) -> str:
    same = [name for name, spec in MODELS.items() if json.dumps(spec, sort_keys=True) == json.dumps(point["spec"],
                                                                                                    sort_keys=True)]
    if same:
        return same[0]
    model = point["spec"].get("run") or "their checkpoint"
    folder = "/".join(Path(model).parts[-3:-1]) if "run" in point["spec"] else model
    folder = folder.replace("experiments/", "")
    return f"{family} {folder}, {point['study']} t{point['trial']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-dir", default=STUDY_DIR)
    parser.add_argument("--fronts", nargs="+", default=None, help='e.g. "RSGM inpaint" (default: all)')
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--n", type=int, default=TEST_POINTS, help="thresholds per front (the rule: 10)")
    parser.add_argument("--dry-run", action="store_true", help="only print the points")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    fronts = candidates(Path(args.study_dir).expanduser())
    out, total = {}, {False: 0.0, True: 0.0}
    for (family, method), points in sorted(fronts.items()):
        name = f"{family} {method}"
        if args.fronts and name not in args.fronts:
            continue
        chosen = choose(points, args.n)
        print(f"\n{name}: {len(points)} candidates, {len(chosen)} points")
        out[name] = []
        for p in chosen:
            label = label_of(family, p)
            done = label in MODELS
            hours = 5 * (p["minutes"] or 0) / 60  # the test has 5 times the poses of validation
            total[done] += 0 if done else hours
            print(f"  APD >= {p['thresholds']}: {p['mean']:.2f} / {p['best_of_k']:.2f} / {p['apd']:.2f}  "
                  f"{p['study']} t{p['trial']}  " + ("[a row of MODELS] " if done else f"[~{hours:.1f} h on {p['host']}] ")
                  + label)
            out[name].append(dict(label=label, study=p["study"], trial=p["trial"], thresholds=p["thresholds"],
                                  validation=dict(mean=p["mean"], best_of_k=p["best_of_k"], apd=p["apd"]),
                                  spec=p["spec"]))
    print(f"\nnew points: about {total[False]:.0f} GPU hours (5 x the minutes of their validation trial)")
    if not args.dry_run:
        header = ("# Points of the completion fronts for the test, chosen by the rule of scripts/select_test_points.py\n"
                  "# (fixed before the test); run by scripts/evaluate_completion.py --points. Validation numbers:\n"
                  "# mean and best of 10 MPVPE (mm), APD (cm), 1000 poses, K = 10.\n")
        Path(args.out).write_text(header + yaml.safe_dump(out, sort_keys=False, width=120))
        print("\nwritten", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
