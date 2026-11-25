"""Choose the sampling weights of the training poses with Optuna: generation against pose completion (VALIDATION only).

    python scripts/tune_weighting.py config/weighting_tuning/rdw_inpaint.yaml --device cuda:0
    python scripts/tune_weighting.py config/weighting_tuning/rdw_inpaint.yaml --summary
    python scripts/tune_weighting.py config/weighting_tuning/rdw_inpaint.yaml --summary \\
        --objectives generation.fid_val inpaint.hv_mean     # the front of the same trials in other metrics
    python scripts/tune_weighting.py config/weighting_tuning/rdw_inpaint.yaml --device cpu --smoke     # tiny sizes

A trial trains one model (the network fact_RDW, seed `seed`) on poses drawn with the weights

    w_i = clip((r_i / median r)^beta, w_min, w_max)

(r_i the rarity of explore/evaluation/data_weighting/rarity.py, read from ~/data/experiments/data_weighting/rarity.pt),
evaluates its generation and its completion and returns the values of the two (or more) `objectives` of the YAML file.
All metrics of both are kept as user attributes of the trial whatever the objectives are, so another choice of metrics
needs no new trials (--summary --objectives); only the sampler (TPE) was then guided by the old ones.

THE METRICS. An objective is <group>.<metric>, the direction is given by the metric:
  generation.*   (the harness of the factorial study: 20 sets of 500 samples and 20,000 samples against the validation
                 poses; own sampler of the model, or Heun on the EDM grid with `sampler: heun`)
                 wasserstein, recall, precision (deg, min), fid_val, fid_own (min), log_likelihood (nats, max)
  <method>.*     one group per entry of `completion` (inpaint, dposer = SDS, dflow); at its ready settings `points`, on
                 `poses` validation poses, K = 10, the means over `occlusions`
                 hv_best, hv_mean (hp.evaluation.fronts.hypervolume over APD 4.0-8.5 cm, max), band_mean (mm, min),
                 best, mean (the means over the points of best of 10 and of MPVPE mean, mm, min)

The YAML file (its name is the name of the study):
    train: {config: rdw, max_iterations: 30000}      # a config of scripts/train_rsgm_posedflow.py (without weights)
    seed: 10
    sampler: own                                     # generation: own | heun
    completion:
      inpaint: {poses: 500, points: [{steps: 400, guidance: 0.05, resample: 5, replace: false, schedule: edm}, ...]}
    objectives: [generation.wasserstein, inpaint.hv_best]
    n_trials: 24                                     # finished trials in all
    params: {beta: {low: 0, high: 3.5}, w_min: {low: 0.1, high: 1, log: true}, w_max: {low: 2, high: 10, log: true}}
    known:                                           # trained already: evaluated, not trained
      - {beta: 0.6, w_min: 0.1, w_max: 10, run: ~/data/experiments/data_weighting/rsgm_posedflow_rdw_b0.6_s10/run_0001}
Trials of `known` run first; a run folder that is missing is trained. Run again, the study continues. The run of a trial
keeps its EMA weights, configuration and model.json; the checkpoints are deleted.
"""
import argparse
import gc
import importlib.util
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the code of this checkout, before an editable install of another one

from hp.evaluation.completion import OCCLUSIONS, Body, build, metrics, run as complete, summary, valid_poses  # noqa: E402
from hp.evaluation.data import load_reference  # noqa: E402
from hp.evaluation.evaluate import EvalConfig, evaluate  # noqa: E402
from hp.evaluation.fronts import APD_RANGE, HV_REFERENCE, hypervolume  # noqa: E402
from hp.evaluation.posedflow_protocol import SmplSkeleton, apd, confidence, fid  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.training import cli  # noqa: E402

STUDY_DIR = "~/data/experiments/data_weighting/optuna"
RARITY = "~/data/experiments/data_weighting/rarity.pt"
ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"
SPLIT = "posedflow"
SKELETON = "~/data/body_models/smpl/smpl_neutral_skeleton.npz"
K = 10
BATCH = 500
SEED = 10            # of Optuna's sampler, of the completion and of the training (train seed: `seed` of the file)
N_STARTUP = 10
SETS, SET_SIZE, N_CONTRAST = 20, 500, 100_000  # as scripts/evaluate_factorial.py of the factorial study
ODE_SAMPLER = dict(sampler="heun", schedule="edm", sample_steps=50)
GENERATION = {  # metric: direction
    "wasserstein": "minimize", "recall": "minimize", "precision": "minimize", "fid_val": "minimize",
    "fid_own": "minimize", "log_likelihood": "maximize"}
COMPLETION = {"hv_best": "maximize", "hv_mean": "maximize", "band_mean": "minimize", "best": "minimize",
              "mean": "minimize"}
METHODS = ("inpaint", "dposer", "dflow")
RUN_KEYS = ("lr", "running_weight", "steps", "ode_steps")  # dflow: arguments of run(), the others go to the prior
KEYS = {"train", "seed", "sampler", "completion", "objectives", "n_trials", "params", "known"}
BAND = APD_RANGE[1] - APD_RANGE[0]


# ---- pure parts (tested) -----------------------------------------------------------------

def weights_of(r: torch.Tensor, beta: float, w_min: float, w_max: float) -> torch.Tensor:
    """The sampling weights of the poses, as rarity.weights with the floor and the cap as arguments."""
    return (r / r.median()).pow(beta).clamp(w_min, w_max)


def parse_objective(name: str):
    """("generation" or a method, metric, direction) of "generation.wasserstein", "inpaint.hv_best", ..."""
    group, _, metric = name.partition(".")
    table = GENERATION if group == "generation" else COMPLETION if group in METHODS else None
    if table is None or metric not in table:
        raise ValueError(f"unknown objective {name!r}: generation.<{'|'.join(GENERATION)}> or "
                         f"<{'|'.join(METHODS)}>.<{'|'.join(COMPLETION)}>.")
    return group, metric, table[metric]


def load_config(path: Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    missing = {"completion", "objectives", "n_trials", "params"} - set(config)
    if missing or set(config) - KEYS:
        raise ValueError(f"{path}: missing {sorted(missing)}, unknown {sorted(set(config) - KEYS)}.")
    config = {"train": {"config": "rdw"}, "seed": SEED, "sampler": "own", "known": [], **config}
    if config["sampler"] not in ("own", "heun"):
        raise ValueError(f"{path}: sampler must be own or heun.")
    if set(config["params"]) != {"beta", "w_min", "w_max"}:
        raise ValueError(f"{path}: params must be beta, w_min and w_max (a fixed one: low = high).")
    for method, part in config["completion"].items():
        if method not in METHODS or not part.get("points"):
            raise ValueError(f"{path}: completion.{method} must be one of {METHODS} with points.")
    for name in config["objectives"]:
        group = parse_objective(name)[0]
        if group != "generation" and group not in config["completion"]:
            raise ValueError(f"{path}: objective {name} needs completion.{group}.")
    if len(config["objectives"]) < 1:
        raise ValueError(f"{path}: give at least one objective.")
    return config


def completion_metrics(rows: list) -> dict:
    """The metrics of COMPLETION of the points of one method, rows {mean, best_of_k, apd}. A front that covers only a
    part of the APD band loses the rest of its area, so hv and band_mean compare only with the same points."""
    hv_best, hv_mean = hypervolume(rows, "best_of_k"), hypervolume(rows, "mean")
    return {"hv_best": hv_best, "hv_mean": hv_mean, "band_mean": HV_REFERENCE["mean"] - hv_mean / BAND,
            "best": float(np.mean([r["best_of_k"] for r in rows])), "mean": float(np.mean([r["mean"] for r in rows]))}


def known_run(config: dict, params: dict):
    """The run folder of `known` with these weights, or None; beta = 0 means the same weights whatever the limits."""
    for entry in config["known"]:
        same_beta = abs(entry["beta"] - params["beta"]) < 1e-9
        same = same_beta and (params["beta"] == 0 or all(abs(entry[k] - params[k]) < 1e-9 for k in ("w_min", "w_max")))
        if same:
            return Path(entry["run"]).expanduser()
    return None


def values_of(objectives, attrs: dict):
    """The values of the objectives from the user attributes of a trial (a missing metric is an error)."""
    out = []
    for name in objectives:
        group, metric, _ = parse_objective(name)
        out.append(float(attrs[f"{group}.{metric}"]))
    return out


def pareto(rows: list, objectives) -> list:
    """The trials on the Pareto front of the objectives, rows (number, attrs); sorted by the first objective."""
    signs = [1.0 if parse_objective(o)[2] == "minimize" else -1.0 for o in objectives]
    points = [(n, a, [s * v for s, v in zip(signs, values_of(objectives, a))]) for n, a in rows]
    front = [(n, a, v) for n, a, v in points
             if not any(all(x <= y for x, y in zip(w, v)) and any(x < y for x, y in zip(w, v)) for _, _, w in points)]
    return [(n, a) for n, a, v in sorted(front, key=lambda t: t[2][0])]


# ---- one trial ---------------------------------------------------------------------------

def train_script():
    """scripts/train_rsgm_posedflow.py as a module (its settings COMMON and CONFIGS)."""
    spec = importlib.util.spec_from_file_location("train_rsgm_posedflow", ROOT / "scripts" / "train_rsgm_posedflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(config: dict, weights_file: Path, run_root: Path, name: str, device: str, steps=None) -> Path:
    """Train a run with the weights of a file, in <run_root>/<name>/run_0001. Returns its folder."""
    module = train_script()
    iterations = steps or config["train"].get("max_iterations")
    argv = ["model=rsgm", f"run_root={run_root}", f"run_name={name}", f"seed={config['seed']}",
            f"data.rotations={ROTATIONS}", f"data.split={SPLIT}", "data.load_dtype=float32", f"train.device={device}",
            *module.COMMON, *module.CONFIGS[config["train"]["config"]], f"train.sample_weights={weights_file}"]
    if iterations:
        argv += [f"train.max_iterations={iterations}"]
    run = Path(cli.main(argv))
    if cli.exit_code(run) != 0:
        raise RuntimeError(f"{run} did not finish")
    for checkpoint in run.glob("checkpoint_*"):
        shutil.rmtree(checkpoint)
    for leftover in ("train_state.pt", "weights.pt"):
        (run / leftover).unlink(missing_ok=True)
    return run


class Data:
    """The reference poses of the harness and the validation poses of the completion, loaded once per process."""

    def __init__(self, poses: int):
        self.valid = load_reference("valid", ROTATIONS, SPLIT)
        self.train = load_reference("train", ROTATIONS, SPLIT)
        self.skeleton = SmplSkeleton(SKELETON)
        self.stats = {}
        for name, data in (("valid", self.valid), ("train", self.train)):
            j = self.skeleton.joints(data).reshape(len(data), -1).numpy().astype(np.float64)
            self.stats[name] = (j.mean(0), np.cov(j, rowvar=False))
        torch.manual_seed(SEED)
        self.gt, self.body = valid_poses(ROTATIONS, poses), Body()


def generation_metrics(run: Path, config: dict, data: Data, device: str, sizes: dict) -> dict:
    """The harness of scripts/evaluate_factorial.py (part `generation`, or generation_ode with sampler heun), without
    the nearest training pose (dNN): the same numbers for the same samples."""
    model = load_model(run, device=device, weights="weights_ema").eval()
    if config["sampler"] == "heun":
        for key, value in ODE_SAMPLER.items():
            setattr(model, key, value)
    values = {"fid_val": [], "fid_own": [], "apd": []}
    torch.manual_seed(0)
    for _ in range(sizes["sets"]):
        joints = data.skeleton.joints(model.sample(sizes["set_size"]).float().cpu())
        values["fid_val"].append(fid(joints, *data.stats["valid"]))
        values["fid_own"].append(fid(joints, *data.stats["train"]))
        values["apd"].append(apd(joints))
    pick = np.random.default_rng(0).choice(len(data.train), sizes["contrast"], replace=False)
    if config["sampler"] == "heun":
        model.has_log_prob = False  # this instance only: the likelihood does not depend on the sampler
    report = evaluate(model, data.valid, sizes["harness"], contrast=data.train[torch.from_numpy(pick)])
    likelihood = report["likelihood"]
    return {"wasserstein": report["wasserstein"]["generated"]["joint_mean_deg"]["mean"],
            "recall": report["recall"]["mean_deg"]["mean"], "precision": report["precision"]["mean_deg"]["mean"],
            "fid_val": confidence(values["fid_val"])["mean"], "fid_own": confidence(values["fid_own"])["mean"],
            "sample_apd": confidence(values["apd"])["mean"],
            "log_likelihood": float("nan") if not likelihood else likelihood["mean"]}


def completion_rows(run: Path, method: str, part: dict, data: Data, device: str, occlusions) -> list:
    """The points of a method: {mean, best_of_k, apd} averaged over the occlusions, at the ready settings."""
    model, rows = build({"run": str(run)}, device), []
    gt = data.gt[:part["poses"]]
    for options in part["points"]:
        values = {"mean": [], "best_of_k": [], "apd": []}
        for occlusion in occlusions:
            torch.manual_seed(SEED)
            if method == "dflow":
                kwargs = {key: options[key] for key in RUN_KEYS if key in options}
                prior = {key: value for key, value in options.items() if key not in RUN_KEYS}
                pred = complete(model, gt, occlusion, K, batch=BATCH, together=True, progress=False, prior=prior, **kwargs)
            else:
                prior = dict(options)
                if method == "dposer" and (prior.get("start", "rest") != "rest" or "scale" in prior):
                    prior.setdefault("stats", ROTATIONS)
                pred = complete(model, gt, occlusion, K, batch=BATCH, together=True, progress=False, method=method,
                                prior=prior)
            s = summary(metrics(pred, gt, occlusion, data.body))
            for name, key in (("mean", "mpvpe_mean"), ("best_of_k", "mpvpe_min"), ("apd", "apd")):
                values[name].append(s[key])
        rows.append({name: float(np.mean(v)) for name, v in values.items()})
    return rows


def sizes_of(smoke: bool) -> dict:
    if smoke:
        return dict(sets=2, set_size=100, contrast=2_000, train_steps=20,
                    harness=EvalConfig(n_wasserstein=100, n_queries=100, n_generated=500, n_likelihood=20, repeats=2))
    return dict(sets=SETS, set_size=SET_SIZE, contrast=N_CONTRAST, train_steps=None, harness=EvalConfig(n_likelihood=1000))


def objective_of(config: dict, data: Data, rarity: torch.Tensor, work: Path, device: str, smoke: bool):
    sizes = sizes_of(smoke)

    def objective(trial):
        p = config["params"]
        params = {name: trial.suggest_float(name, p[name]["low"], p[name]["high"], log=p[name].get("log", False))
                  for name in ("beta", "w_min", "w_max")}
        trial.set_user_attr("host", platform.node().split(".")[0])
        trial.set_user_attr("device", device)
        started = time.perf_counter()
        run = known_run(config, params)
        if run is not None and (run / "weights_ema.pt").is_file():
            trial.set_user_attr("run", str(run))
            trial.set_user_attr("trained", False)
        else:
            if run is not None:
                print(f"trial {trial.number}: no {run}/weights_ema.pt, trained instead", flush=True)
            weights_file = work / f"trial_{trial.number:04d}_weights.pt"
            torch.save(weights_of(rarity, **params), weights_file)
            run = train(config, weights_file, work, f"trial_{trial.number:04d}", device, sizes["train_steps"])
            weights_file.unlink()
            trial.set_user_attr("run", str(run))
            trial.set_user_attr("trained", True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        trained = time.perf_counter()
        trial.set_user_attr("ess", float(ess(weights_of(rarity, **params))))
        for name, value in generation_metrics(run, config, data, device, sizes).items():
            trial.set_user_attr(f"generation.{name}", value)
        generated = time.perf_counter()
        for method, part in config["completion"].items():
            if smoke:
                part = {**part, "poses": 20, "points": part["points"][:2]}
            occlusions = part.get("occlusions", list(OCCLUSIONS))
            rows = completion_rows(run, method, part, data, device, occlusions)
            trial.set_user_attr(f"{method}.rows", rows)
            for name, value in completion_metrics(rows).items():
                trial.set_user_attr(f"{method}.{name}", value)
        done = time.perf_counter()
        trial.set_user_attr("minutes", {"train": (trained - started) / 60, "generation": (generated - trained) / 60,
                                        "completion": (done - generated) / 60})
        attrs = trial.user_attrs
        print(f"trial {trial.number} ({(done - started) / 60:.1f} min): {params} -> " +
              ", ".join(f"{o} {v:.4g}" for o, v in zip(config["objectives"], values_of(config["objectives"], attrs))),
              flush=True)
        return tuple(values_of(config["objectives"], attrs))
    return objective


def ess(w: torch.Tensor) -> float:
    w = w.double()
    return float(w.sum() ** 2 / (w ** 2).sum() / len(w))


# ---- the study ---------------------------------------------------------------------------

def open_study(name: str, study_dir: Path, objectives, create: bool):
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{study_dir / f'{name}.db'}",
                                         engine_kwargs={"connect_args": {"timeout": 120}})
    if not create:
        return optuna.load_study(study_name=name, storage=storage)
    directions = [parse_objective(o)[2] for o in objectives]
    return optuna.create_study(study_name=name, storage=storage, load_if_exists=True, directions=directions)


def settings_of(config: dict) -> dict:
    """What must not change between the sessions of a study (n_trials may)."""
    return {key: value for key, value in config.items() if key != "n_trials"}


def check_or_record(study, config: dict) -> None:
    made = json.loads(json.dumps(settings_of(config)))
    if "config" not in study.user_attrs:
        study.set_user_attr("config", made)
        return
    if study.user_attrs["config"] != made:
        raise SystemExit(f"study {study.study_name} was made with another configuration; name the file differently.")


def report(study, objectives) -> None:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    states = {s.name: sum(t.state == s for t in study.trials) for s in optuna.trial.TrialState}
    print(f"study {study.study_name}: {states}")
    if not done:
        return
    rows = [(t.number, t.user_attrs) for t in done]
    front = {n for n, _ in pareto(rows, objectives)}
    names = list(objectives)
    others = [f"generation.{m}" for m in GENERATION] + [f"{m}.{c}" for m in METHODS for c in COMPLETION]
    others = [o for o in others if o not in names and all(o in t.user_attrs for t in done)]
    print("front | trial | beta | w_min | w_max | " + " | ".join(names + others) + " | min")
    for t in sorted(done, key=lambda t: t.user_attrs.get(names[0], 0)):
        minutes = sum((t.user_attrs.get("minutes") or {}).values())
        print(f"{'*' if t.number in front else ' '} | {t.number} | {t.params['beta']:.3f} | {t.params['w_min']:.3f} | "
              f"{t.params['w_max']:.2f} | " + " | ".join(f"{t.user_attrs[o]:.4g}" for o in names + others) +
              f" | {minutes:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--study-dir", default=STUDY_DIR)
    parser.add_argument("--summary", action="store_true", help="only print the trials so far")
    parser.add_argument("--objectives", nargs="+", default=None, metavar="GROUP.METRIC",
                        help="with --summary: the front of these metrics instead of those of the file")
    parser.add_argument("--n-trials", type=int, default=None, help="finished trials to reach (raises n_trials of the file)")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes in a study of its own, to test the script")
    args = parser.parse_args()
    if args.objectives and not args.summary:
        parser.error("--objectives only with --summary: the objectives of a study are those of its file")
    path = Path(args.config)
    config = load_config(path)
    name = path.stem + ("_smoke" if args.smoke else "")
    study_dir = Path(args.study_dir).expanduser()
    for o in args.objectives or ():
        parse_objective(o)
    study = open_study(name, study_dir, config["objectives"], create=not args.summary)
    if args.summary:
        report(study, args.objectives or config["objectives"])
        return 0
    if args.smoke:
        config = {**config, "n_trials": min(config["n_trials"], 2), "known": config["known"][:1]}
    if args.n_trials is not None:
        config["n_trials"] = args.n_trials
    check_or_record(study, config)
    if not study.trials:
        for entry in config["known"]:
            study.enqueue_trial({k: entry[k] for k in ("beta", "w_min", "w_max")})
    study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials), multivariate=True, constant_liar=True,
                                               n_startup_trials=N_STARTUP)
    finished = sum(t.state.is_finished() for t in study.trials)
    if finished < config["n_trials"]:
        rarity = torch.load(Path(RARITY).expanduser(), weights_only=True)
        data = Data(max(part["poses"] for part in config["completion"].values()))
        work = study_dir / "runs" / name
        work.mkdir(parents=True, exist_ok=True)
        print(f"study {name}: {finished} of {config['n_trials']} trials done, objectives {config['objectives']}", flush=True)
        stop = optuna.study.MaxTrialsCallback(config["n_trials"], states=(
            optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL))
        study.optimize(objective_of(config, data, rarity, work, args.device, args.smoke),
                       n_trials=config["n_trials"] - finished, callbacks=[stop], catch=(torch.cuda.OutOfMemoryError,))
    report(study, config["objectives"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
