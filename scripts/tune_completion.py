"""Choose the settings of a pose completion method with Optuna, on their VALIDATION poses (never the test poses).

    python scripts/tune_completion.py --device cuda:0                         # DPoser's method on RSGM 512x4 c10
    python scripts/tune_completion.py --method dflow --device cuda:0          # D-Flow on the same model, same budget
    python scripts/tune_completion.py --summary                               # only the table of the trials so far

The objective of a trial is the mean over the four occlusions of MPVPE mean (mm, the main number of their table) on
POSES fixed random validation poses with K hypotheses, the same draw and the same seeds in every trial (common random
numbers), so trials differ only by their settings. Per-occlusion MPVPE mean, best of K and APD are kept as user
attributes. One set of settings for all occlusions.

Spaces:
  - dposer (rsgm_complete through run(method="dposer")): t_max (the first time of the schedule), t_min as a fraction
    of t_max, lr, start (rest / mean / noise), scale (none / std), prior_weight (data_weight fixed at their 100);
    200 steps as theirs. The first trials are their settings and the best of our exploration.
  - dflow (their D-Flow): lr, invert (their start or the latent of the observation), running_weight; the steps and
    the ODE of their protocol. The first trial is their protocol.

The study lives in SQLite (STUDY_DIR / optuna.db) and can be stopped and run again; it continues. The best settings
are written to <study>.json next to it, to be copied into MODELS of evaluate_completion.py for the test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import torch

from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run, summary, valid_poses

# ---- settings --------------------------------------------------------------------------
MODEL = "~/data/experiments/rsgm_posedflow_small_s10/run_0001"  # RSGM 512x4, their c10, seed 10
ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"               # their data: the validation split and the statistics
STUDY_DIR = "~/data/experiments/completion/optuna"
POSES = 1000                                                      # validation poses (a fixed draw)
K = 10                                                            # hypotheses per pose
N_TRIALS = {"dposer": 80, "dflow": 25}                            # finished trials of a study
BATCH = 500
SEED = 10
# ----------------------------------------------------------------------------------------

FIRST = {  # enqueued before the sampler takes over
    "dposer": [dict(t_max=0.15, t_min_fraction=1 / 3, lr=0.1, start="rest", scale="none", prior_weight=0.1),  # theirs
               dict(t_max=0.05, t_min_fraction=0.1, lr=0.02, start="rest", scale="none", prior_weight=0.1)],  # explored
    "dflow": [dict(lr=0.1, invert=False, running_weight=1e-3)],  # their protocol
}


def options_of(method: str, trial) -> dict:
    """The options of run() (its `prior`) of a trial."""
    if method == "dposer":
        t_max = trial.suggest_float("t_max", 0.003, 0.3, log=True)
        options = dict(t_max=t_max, t_min=t_max * trial.suggest_float("t_min_fraction", 0.02, 0.9),
                       lr=trial.suggest_float("lr", 0.003, 0.3, log=True),
                       start=trial.suggest_categorical("start", ["rest", "mean", "noise"]),
                       prior_weight=trial.suggest_float("prior_weight", 0.01, 10.0, log=True), stats=ROTATIONS)
        if trial.suggest_categorical("scale", ["none", "std"]) == "std":
            options["scale"] = "std"
        return options
    return dict(invert=trial.suggest_categorical("invert", [False, True]),
                lr=trial.suggest_float("lr", 0.01, 0.3, log=True),
                running_weight=trial.suggest_float("running_weight", 1e-5, 1e-1, log=True))


def objective_of(args, model, gt, body):
    def objective(trial):
        options = options_of(args.method, trial)
        start, means = time.perf_counter(), []
        for occlusion in OCCLUSIONS:
            if args.method == "dposer":
                pred = run(model, gt, occlusion, K, batch=BATCH, together=True, progress=False, method="dposer",
                           prior=options)
            else:
                lr, weight = options.pop("lr"), options.pop("running_weight")
                pred = run(model, gt, occlusion, K, lr=lr, running_weight=weight, batch=BATCH, together=True,
                           progress=False, prior=options)
                options.update(lr=lr, running_weight=weight)
            s = summary(metrics(pred, gt, occlusion, body))
            for key in ("mpvpe_mean", "mpvpe_min", "apd"):
                trial.set_user_attr(f"{occlusion}_{key}", s[key])
            means.append(s["mpvpe_mean"])
            trial.report(float(np.mean(means)), len(means) - 1)
        trial.set_user_attr("minutes", (time.perf_counter() - start) / 60)
        return float(np.mean(means))
    return objective


def study_of(args):
    study_dir = Path(args.study_dir).expanduser()
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{study_dir / 'optuna.db'}",
                                         engine_kwargs={"connect_args": {"timeout": 120}})
    return optuna.create_study(study_name=args.study, storage=storage, load_if_exists=True, direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, n_startup_trials=15))


def report(study, study_dir: Path) -> None:
    frame = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    if frame.empty:
        print("no trials yet")
        return
    print(frame.sort_values("value").head(15).to_string(index=False))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        best = study.best_trial
        out = {"study": study.study_name, "value": best.value, "params": best.params, "attrs": best.user_attrs,
               "trials": len(done)}
        (study_dir / f"{study.study_name}.json").write_text(json.dumps(out, indent=1))
        print(f"\nbest: trial {best.number}, {best.value:.2f} mm, {best.params}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["dposer", "dflow"], default="dposer")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--study", default=None, help="default: completion_<method>_<run folder>")
    parser.add_argument("--study-dir", default=STUDY_DIR)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--poses", type=int, default=POSES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary", action="store_true", help="only print the trials so far")
    args = parser.parse_args()
    run_name = Path(args.model).expanduser().parent.name
    args.study = args.study or f"completion_{args.method}_{run_name}_v{args.poses}"
    study = study_of(args)
    if args.summary:
        report(study, Path(args.study_dir).expanduser())
        return 0
    if not study.trials:
        for params in FIRST[args.method]:
            study.enqueue_trial(params)
    n_trials = args.n_trials or N_TRIALS[args.method]
    finished = sum(t.state.is_finished() for t in study.trials)
    if finished < n_trials:
        torch.manual_seed(SEED)
        gt, body = valid_poses(ROTATIONS, args.poses), Body()
        model = build({"run": args.model}, args.device)
        print(f"study {args.study}: {finished} of {n_trials} trials done, {len(gt)} validation poses", flush=True)
        study.optimize(objective_of(args, model, gt, body), n_trials=n_trials - finished)
    report(study, Path(args.study_dir).expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
