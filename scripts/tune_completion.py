"""Choose the settings of a pose completion method with Optuna, on their VALIDATION poses (never the test poses).

    python scripts/tune_completion.py config/completion_tuning/rsgm_sds_pareto.yaml --device cuda:0
    python scripts/tune_completion.py config/completion_tuning/rsgm_sds_pareto.yaml --summary
    python scripts/tune_completion.py config/completion_tuning/rsgm_sds_pareto.yaml --device cuda:0 --n-trials 120
    python scripts/tune_completion.py --study rsgm_sds_pareto --summary          # by name, without the file
    python scripts/tune_completion.py config/completion_tuning/a.yaml config/completion_tuning/b.yaml --device cuda:1
                                        # several studies, one after another in one process (one GPU)
    python scripts/tune_completion.py config/completion_tuning/a.yaml --n-trials 25 --enqueue '{"lr": 0.05}'
                                        # 25 finished trials in all, the given settings first

A study is described by a YAML file in config/completion_tuning, and its name is the name of the file:

    model: ~/data/experiments/rsgm_posedflow_small_s10/run_0001     # a run folder, or a .ckpt: their PoseD-Flow
    method: dposer          # DPoser's optimization (run(method="dposer")), dflow (D-Flow) or inpaint (RSGM, DPoser)
    objective: mean         # mean: the mean over the occlusions of MPVPE mean; pareto: that and of MPVPE best of K;
                            # mean_apd: that and of APD (maximized)
    poses: 1000             # fixed random validation poses
    n_trials: 80            # finished trials of the study (may be raised later)
    params:                 # searched: {low, high, log} a float, {low, high, int: true} an integer, [a, b] a choice
      lr: {low: 0.01, high: 0.15, log: true}
      start: [noise, mean]
    fixed: {scale: std}     # the same in every trial
    first: [{lr: 0.05, start: noise}]                                # trials enqueued into a new study
    batch: 500              # optional: validation poses per GPU batch (x K hypotheses); part of the study's seeds
    import:                 # optional: finished trials of another study copied into a new one, not run again
      study: rsgm_sds_old   # (same model, method, poses, batch; their settings inside this one's space)
      trials: [2, 14]
      params: {scale: std}  # values of this study's params that the other one did not search

The values of a trial (fixed and searched) are options of run(): for dposer all of them go to the optimization
(dposer_complete or rsgm_complete, and start / scale / stats of run's RSGM starts); for inpaint, to rsgm_inpaint
or dposer_inpaint (steps, schedule, resample, guidance, replace); for dflow, lr, running_weight,
steps and ode_steps go to run() and the others to the D-Flow prior (e.g. invert). t_min_fraction stands for
t_min = t_max * t_min_fraction, and scale "none" for no scale.

Every trial runs on the same validation poses with the same seeds (common random numbers), so trials differ only by
their settings; per occlusion MPVPE mean, best of K and APD are kept as user attributes, with the minutes, the host
and the device of the trial. One set of settings for all
occlusions. A single mean rewards hypotheses that collapse onto one pose (the mean error of K hypotheses is smallest
when they agree), the best of K rewards covering the plausible completions; objective "pareto" keeps the front of the
two, objective "mean_apd" the front of the mean and the diversity (APD, higher is better: alone it would favour random
poses), and a setting is chosen from the front by a rule fixed before the test.

The study lives in STUDY_DIR / <name>.db and keeps its configuration, K and the git commit of its creation as user
attributes; every trial keeps the commit it ran on. Run again, it continues; it refuses to continue if the file now
says something else (other than n_trials) than the study was made with. The best settings (the Pareto front) are
written to <name>.json next to it, to be copied into MODELS of evaluate_completion.py for the test.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from omegaconf import OmegaConf

from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run, summary, valid_poses

STUDY_DIR = "~/data/experiments/completion/optuna"
ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"  # their data: the validation split and the statistics of the starts
K = 10                                               # hypotheses per pose
BATCH = 500
SEED = 10
METHODS = ("dposer", "dflow", "inpaint")
OBJECTIVES = {  # name: (the values of a trial, their directions)
    "mean": (("mean",), ("minimize",)),
    "pareto": (("mean", "best_of_k"), ("minimize", "minimize")),
    "mean_apd": (("mean", "apd"), ("minimize", "maximize")),
}
FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL)
RUN_KEYS = ("lr", "running_weight", "steps", "ode_steps")  # dflow: arguments of run(), the others go to the prior
KEYS = ("model", "method", "objective", "poses", "n_trials", "params", "fixed", "first", "batch", "import")


def load_config(path: Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    missing = set(KEYS[:6]) - set(config)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}.")
    unknown = set(config) - set(KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}.")
    if config["method"] not in METHODS or config["objective"] not in OBJECTIVES:
        raise ValueError(f"{path}: method must be one of {METHODS} and objective one of {list(OBJECTIVES)}.")
    config.setdefault("fixed", {})
    config.setdefault("first", [])
    for name, spec in config["params"].items():
        if not (isinstance(spec, list) or (isinstance(spec, dict) and {"low", "high"} <= set(spec))):
            raise ValueError(f"{path}: param '{name}' must be [choices] or {{low, high[, log][, int]}}, got {spec}.")
    return config


def suggest(trial, name: str, spec):
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if spec.get("int"):
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


def distribution(spec):
    """The Optuna distribution of a param of the YAML file (as suggest draws it)."""
    if isinstance(spec, list):
        return optuna.distributions.CategoricalDistribution(spec)
    if spec.get("int"):
        return optuna.distributions.IntDistribution(spec["low"], spec["high"], log=spec.get("log", False))
    return optuna.distributions.FloatDistribution(spec["low"], spec["high"], log=spec.get("log", False))


def import_trials(study, config: dict, study_dir: Path) -> None:
    """Copy the trials config["import"] names from another study into the new `study`: their results, with the params
    of this study (the other's values, or those of import.params) and the attribute imported_from. Refused unless the
    other study has the same model, method, poses, K, batch and validation set, every trial finished, its values of
    this study's fixed settings the same and its params inside this study's space."""
    spec = config["import"]
    path = study_dir / f"{spec['study']}.db"
    if not path.is_file():  # optuna.load_study would create an empty file
        raise SystemExit(f"import: no study file {path}")
    other = open_study(spec["study"], study_dir)
    theirs = other.user_attrs.get("config") or {}
    ours = dict(config, k=K, batch=config.get("batch", BATCH), rotations=ROTATIONS)
    for key in ("model", "method", "poses", "k", "batch", "rotations", "objective"):
        if theirs.get(key) != ours.get(key):
            raise SystemExit(f"import: {spec['study']} differs in {key}: {theirs.get(key)} != {ours.get(key)}")
    distributions = {name: distribution(p) for name, p in config["params"].items()}
    by_number = {t.number: t for t in other.trials}
    for number in spec["trials"]:
        trial = by_number.get(number)
        if trial is None or trial.state != optuna.trial.TrialState.COMPLETE:
            raise SystemExit(f"import: trial {number} of {spec['study']} is not a finished trial")
        values = {**(theirs.get("fixed") or {}), **trial.params, **(spec.get("params") or {})}
        for key, value in config["fixed"].items():
            if values.get(key) != value:
                raise SystemExit(f"import: trial {number} has {key}={values.get(key)}, this study fixes {value}")
        params = {name: values[name] for name in distributions}
        for name, value in params.items():
            if not distributions[name]._contains(distributions[name].to_internal_repr(value)):
                raise SystemExit(f"import: trial {number} has {name}={value} outside {config['params'][name]}")
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=distributions, values=trial.values,
            user_attrs={**trial.user_attrs, "imported_from": f"{spec['study']}:{number}"}))
    print(f"imported trials {spec['trials']} of {spec['study']}", flush=True)


def values_of(config: dict, trial) -> dict:
    """The options of a trial: the fixed values and the suggested ones, with t_min_fraction and scale resolved."""
    values = dict(config["fixed"])
    values.update({name: suggest(trial, name, spec) for name, spec in config["params"].items()})
    if "t_min_fraction" in values:
        values["t_min"] = values["t_max"] * values.pop("t_min_fraction")
    if values.get("scale") == "none":
        values.pop("scale")
    return values


def model_spec(model: str) -> dict:
    """The spec of build() for the model of a study: a .ckpt file is their pretrained PoseD-Flow, else a run folder."""
    return {"posedflow": model} if str(model).endswith(".ckpt") else {"run": model}


def objective_of(config: dict, model, gt, body, commit: str, device: str, tf32: bool = False):
    names = OBJECTIVES[config["objective"]][0]
    pareto = len(names) > 1
    batch = config.get("batch", BATCH)

    def objective(trial):
        values = values_of(config, trial)
        trial.set_user_attr("git", commit)
        trial.set_user_attr("host", platform.node().split(".")[0])
        trial.set_user_attr("device", device)
        trial.set_user_attr("tf32", tf32)
        start, values_of_occlusions = time.perf_counter(), {"mean": [], "best_of_k": [], "apd": []}
        for occlusion in OCCLUSIONS:
            if config["method"] in ("dposer", "inpaint"):
                options = dict(values)
                if config["method"] == "dposer" and (options.get("start", "rest") != "rest" or "scale" in options):
                    options.setdefault("stats", ROTATIONS)
                pred = run(model, gt, occlusion, K, batch=batch, together=True, progress=False,
                           method=config["method"], prior=options)
            else:
                kwargs = {key: values[key] for key in RUN_KEYS if key in values}
                prior = {key: value for key, value in values.items() if key not in RUN_KEYS}
                pred = run(model, gt, occlusion, K, batch=batch, together=True, progress=False, prior=prior, **kwargs)
            s = summary(metrics(pred, gt, occlusion, body))
            for key in ("mpvpe_mean", "mpvpe_min", "apd"):
                trial.set_user_attr(f"{occlusion}_{key}", s[key])
            for name, key in (("mean", "mpvpe_mean"), ("best_of_k", "mpvpe_min"), ("apd", "apd")):
                values_of_occlusions[name].append(s[key])
            if not pareto:  # intermediate values: single-objective studies only
                trial.report(float(np.mean(values_of_occlusions["mean"])), len(values_of_occlusions["mean"]) - 1)
        trial.set_user_attr("minutes", (time.perf_counter() - start) / 60)
        result = tuple(float(np.mean(values_of_occlusions[name])) for name in names)
        return result if pareto else result[0]
    return objective


def git_commit() -> str:
    """The commit of the repository of this script, with "+dirty" if it has uncommitted changes."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"
    return (commit or "unknown") + ("+dirty" if dirty else "")


def open_study(name: str, study_dir: Path, directions=None, attempts: int = 5):
    """The study in its own SQLite file, study_dir / <name>.db; created with `directions` if new (None: it must
    exist). Two processes that create the same file at the same time race on Optuna's tables; the loser waits and
    tries again (the tables are there then)."""
    study_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            storage = optuna.storages.RDBStorage(f"sqlite:///{study_dir / f'{name}.db'}",
                                                 engine_kwargs={"connect_args": {"timeout": 120}})
            if directions is None:
                return optuna.load_study(study_name=name, storage=storage)
            return optuna.create_study(study_name=name, storage=storage, load_if_exists=True, directions=directions)
        except Exception as error:  # sqlalchemy's OperationalError ("table ... already exists") of the race
            if attempt == attempts - 1 or "already exists" not in str(error):
                raise
            time.sleep(2 + attempt)


def check_or_record(study, config: dict, commit: str) -> None:
    """Record the configuration in a new study; refuse to continue a study made with another one (n_trials aside)."""
    made = {key: value for key, value in config.items() if key != "n_trials"}
    made.update(k=K, batch=config.get("batch", BATCH), rotations=ROTATIONS)
    if "config" not in study.user_attrs:
        if study.trials:
            raise SystemExit(f"study {study.study_name} has trials but no recorded configuration (made by an older "
                             "version of this script): continue it with that version, or name the file differently.")
        study.set_user_attr("config", made)
        study.set_user_attr("git", commit)
        return
    recorded = study.user_attrs["config"]
    if json.dumps(recorded, sort_keys=True) != json.dumps(made, sort_keys=True):
        changed = sorted(key for key in set(recorded) | set(made) if recorded.get(key) != made.get(key))
        raise SystemExit(f"study {study.study_name} was made with another configuration (differs in {changed}):\n"
                         f"  recorded: {json.dumps({k: recorded.get(k) for k in changed})}\n"
                         f"  file:     {json.dumps({k: made.get(k) for k in changed})}\n"
                         "Name the file differently for a new study.")


def report(study, study_dir: Path) -> None:
    pareto = len(study.directions) > 1
    frame = study.trials_dataframe(attrs=("number", "values" if pareto else "value", "params", "state"))
    if frame.empty:
        print("no trials yet")
        return
    print(frame.sort_values("values_0" if pareto else "value").head(15).to_string(index=False))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return
    out = {"study": study.study_name, "trials": len(done), "config": study.user_attrs.get("config"),
           "git": study.user_attrs.get("git")}
    if pareto:
        names = OBJECTIVES[(study.user_attrs.get("config") or {}).get("objective", "pareto")][0]
        front = sorted(study.best_trials, key=lambda t: t.values[0])
        out["front"] = [{"number": t.number, **dict(zip(names, t.values)), **averages(t), "params": t.params,
                         "attrs": t.user_attrs} for t in front]
        print(f"\nPareto front ({', '.join(names)}); all: mean / best of K (mm) / APD (cm):")
        for t in front:
            a = averages(t)
            print(f"  trial {t.number}: {a['mean']:.2f} / {a['best_of_k']:.2f} / {a['apd']:.2f}, {t.params}")
    else:
        best = study.best_trial
        out.update(value=best.value, params=best.params, attrs=best.user_attrs)
        print(f"\nbest: trial {best.number}, {best.value:.2f} mm, {best.params}")
    (study_dir / f"{study.study_name}.json").write_text(json.dumps(out, indent=1))


def averages(trial) -> dict:
    """The means over the occlusions of MPVPE mean, best of K and APD of a trial, from its user attributes."""
    return {name: float(np.mean([trial.user_attrs.get(f"{occlusion}_{key}", np.nan) for occlusion in OCCLUSIONS]))
            for name, key in (("mean", "mpvpe_mean"), ("best_of_k", "mpvpe_min"), ("apd", "apd"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="*",
                        help="the YAML file(s) of the studies (config/completion_tuning/*.yaml), run one after another")
    parser.add_argument("--study", default=None, help="with --summary: a study by name, without its file")
    parser.add_argument("--study-dir", default=STUDY_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary", action="store_true", help="only print the trials so far")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="finished trials to reach, instead of n_trials of the file (raises it, e.g. overnight)")
    parser.add_argument("--enqueue", nargs="+", default=[], metavar="JSON",
                        help='trials to run next, before the sampler, e.g. \'{"lr": 0.05, "steps": 100}\' (the values '
                             'missing are drawn); skipped if the study has them already')
    parser.add_argument("--tf32", action="store_true",
                        help="TF32 matrix products on cuda (faster, lower precision; recorded per trial)")
    args = parser.parse_args()
    study_dir = Path(args.study_dir).expanduser()
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if not args.config:
        if not (args.summary and args.study):
            parser.error("give the YAML file of the study, or --summary --study <name>")
        report(open_study(args.study, study_dir), study_dir)
        return 0
    for path in args.config:
        tune(Path(path), study_dir, args)
    return 0


def tune(path: Path, study_dir: Path, args) -> None:
    """Run (or continue, or with --summary only report) the study of one YAML file."""
    config, name, commit = load_config(path), path.stem, git_commit()
    if args.n_trials is not None:
        config["n_trials"] = args.n_trials
    study = open_study(name, study_dir, list(OBJECTIVES[config["objective"]][1]))
    check_or_record(study, config, commit)
    if args.summary:
        report(study, study_dir)
        return
    if not study.trials:
        if config.get("import"):
            import_trials(study, config, study_dir)
        for params in config["first"]:
            study.enqueue_trial(params)
    for params in args.enqueue:
        study.enqueue_trial(json.loads(params), skip_if_exists=True)
    # seeded by the trials so far: a restarted study does not draw the random trials of its first start again
    study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials), multivariate=True, n_startup_trials=15)
    finished = sum(t.state.is_finished() for t in study.trials)
    if finished < config["n_trials"]:
        torch.manual_seed(SEED)
        gt, body = valid_poses(ROTATIONS, config["poses"]), Body()
        model = build(model_spec(config["model"]), args.device)
        print(f"study {name} ({commit}): {finished} of {config['n_trials']} trials done, {len(gt)} validation poses",
              flush=True)
        # several processes may work on one study (e.g. one per GPU): every one stops when the study has n_trials
        # finished trials in all; start them a little apart, as the sampler is seeded by the trials so far
        stop = optuna.study.MaxTrialsCallback(config["n_trials"], states=FINISHED)
        study.optimize(objective_of(config, model, gt, body, commit, args.device, args.tf32),
                       n_trials=config["n_trials"] - finished, callbacks=[stop])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report(study, study_dir)


if __name__ == "__main__":
    raise SystemExit(main())
