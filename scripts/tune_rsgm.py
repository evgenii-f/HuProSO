"""Hyperparameter search of RSGM with Optuna, from the command line, with several trials in parallel.

    python scripts/tune_rsgm.py                          # the settings below
    python scripts/tune_rsgm.py --workers 3 --device cuda:0
    python scripts/tune_rsgm.py --summary                # only the table of the trials so far

The objective of a trial is its BEST validation log-likelihood over the evaluations at EVAL_AT (early stopping inside
the trial): a configuration that peaks at 20k steps and degrades later is scored at its peak, and the step of the peak
is recorded (`best_step`), so that a final training can stop there. Every evaluation is made the same way (EMA weights,
the stratified validation poses, OBJECTIVE_STEPS ODE steps), and the pruner sees the same values.

Workers are separate processes that share the Optuna database (SQLite): each takes the next trial, TPE's constant liar
keeps them from proposing the same configuration, and they stop together when N_TRIALS trials are finished (complete
or pruned). Every worker loads the data itself. A trial that runs out of GPU memory is marked pruned (`oom`).

The study can be stopped (Ctrl-C) and the script run again: it continues where it stopped.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import optuna
import pandas as pd
import torch

from hp.data.amass import AMASS
from hp.data.amass_cfg import ORIGINAL_AMASS_SPLITS, AmassJoints4Learning
from hp.models import build_model
from hp.models.io import save_model
from hp.training import TrainConfig, Trainer

# ---- settings --------------------------------------------------------------------------
ROTATIONS = "~/data/AMASS_rotations_c20"     # training and validation data: every 20th frame
SPLIT = "original"                           # the official AMASS split
STUDY_DIR = "~/data/experiments/rsgm_hpo"    # the Optuna database and the weights of the trials
STUDY_NAME = "rsgm_c20_best"                 # a new name for a new space or objective
N_TRIALS = 40                                # finished (complete or pruned) trials of the study
WORKERS = 1                                  # trials trained at the same time (processes on the same device)
DEVICE = "cuda:0"                            # "cpu", "mps" or "cuda:0"
SEED = 10
TRIAL_ITERATIONS = 30_000                    # training steps of a trial
EVAL_AT = (10_000, 15_000, 20_000, 25_000, 30_000)  # evaluations; the objective is the best of them
N_PER_DATASET = 500                          # validation poses per valid dataset (4 datasets)
OBJECTIVE_STEPS = 200                        # ODE steps of the log-likelihood
OBJECTIVE_STAT = "mean"                      # of the per-pose log-likelihoods: "mean", "median" or "trimmed" (1%)
WARMUP_STEPS = 500                           # linear warm-up of the cosine schedule
N_STARTUP = 15                               # random trials before TPE models the space
# ----------------------------------------------------------------------------------------

# the reference: trial 34 of the first search (the best stable RSGM so far), as the first trial
REFERENCE = dict(lr=0.0014830328437267638, batch_size=512, width=1024, depth=4, beta_f=8.077840624757041,
                 loss_t_min=0.0017130586845802236, ema_decay=0.999, scheduler="step", weighting="std",
                 weight_decay=1e-6)

States = optuna.trial.TrialState


def suggest(trial):
    """The search space -> (arguments of the model, fields of TrainConfig). Activation fixed at silu."""
    p = dict(
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
        width=trial.suggest_categorical("width", [512, 1024, 1536, 2048, 3072]),
        depth=trial.suggest_int("depth", 3, 8),
        beta_f=trial.suggest_float("beta_f", 5.0, 12.0),
        loss_t_min=trial.suggest_float("loss_t_min", 1e-3, 1e-2, log=True),
        ema_decay=trial.suggest_categorical("ema_decay", [0.999, 0.9995]),
        scheduler=trial.suggest_categorical("scheduler", ["step", "cosine"]),
        weighting=trial.suggest_categorical("weighting", ["std", "likelihood"]),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),  # AdamW
    )
    return split_params(p)


def split_params(p):
    model_args = dict(hidden=[p["width"]] * p["depth"], activation="silu", beta_f=p["beta_f"], loss_t_min=p["loss_t_min"],
                      weighting=p["weighting"], likelihood_steps=OBJECTIVE_STEPS)
    train = dict(lr=p["lr"], batch_size=p["batch_size"], optimizer="adamw", weight_decay=p["weight_decay"],
                 ema_decay=p["ema_decay"], scheduler=p["scheduler"], warmup_steps=WARMUP_STEPS if p["scheduler"] == "cosine" else 0)
    return model_args, train


def as_overrides(p):
    """The parameters of a trial as overrides of scripts/train_model.py and rsgm_train.ipynb."""
    model_args, train = split_params(p)
    return [f"model.args.hidden={json.dumps(model_args['hidden'])}", "model.args.activation=silu",
            f"model.args.beta_f={p['beta_f']}", f"model.args.loss_t_min={p['loss_t_min']}", f"model.args.weighting={p['weighting']}",
            f"train.lr={p['lr']}", f"train.batch_size={p['batch_size']}", "train.optimizer=adamw",
            f"train.weight_decay={p['weight_decay']}", f"train.ema_decay={p['ema_decay']}", f"train.scheduler={p['scheduler']}",
            f"train.warmup_steps={train['warmup_steps']}"]


# ---- data and evaluation ---------------------------------------------------------------

def load_data(rotations):
    with mock.patch.dict(os.environ, {"HUPROSO_AMASS_SPLIT": SPLIT}):
        train_set = AMASS(mode="train", path_to_dataset=rotations, rotation_representation="matrix",
                          selected_joints=AmassJoints4Learning, load_dtype="float32")
    rng = np.random.default_rng(SEED)
    parts = []
    for name in ORIGINAL_AMASS_SPLITS["valid"]:
        poses = AMASS(datasets=[name], path_to_dataset=rotations, rotation_representation="matrix",
                      selected_joints=AmassJoints4Learning).data.reshape(-1, len(AmassJoints4Learning), 3, 3)
        parts.append(poses[rng.choice(len(poses), min(N_PER_DATASET, len(poses)), replace=False)])
    return train_set, torch.cat(parts).float()


@torch.no_grad()
def log_likelihood(model, poses, steps, batch=250):
    old, model.likelihood_steps = model.likelihood_steps, steps
    try:
        return torch.cat([model.log_prob(poses[i:i + batch].to(model.device)).float().cpu() for i in range(0, len(poses), batch)])
    finally:
        model.likelihood_steps = old


def statistic(ll):
    """(objective value, details) of per-pose log-likelihoods; non-finite values make it -inf."""
    ll = ll.double().numpy()
    finite = ll[np.isfinite(ll)]
    details = dict(mean=float(finite.mean()) if len(finite) else -np.inf, median=float(np.median(finite)) if len(finite) else -np.inf,
                   nonfinite=float(1 - len(finite) / len(ll)), implausible=float(((finite < 0) | (finite > 250)).sum() / len(ll)))
    if len(finite) < len(ll):
        return -np.inf, details
    lo, hi = np.percentile(finite, [1, 99])
    details["trimmed"] = float(finite[(finite >= lo) & (finite <= hi)].mean())
    return details[OBJECTIVE_STAT], details


# ---- the objective -----------------------------------------------------------------------

def fmt(p):
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def make_objective(train_set, valid_poses, device, weights_dir, tag):
    def objective(trial):
        model_args, train = suggest(trial)
        print(f"{tag} trial {trial.number:3d} START  {fmt(trial.params)}", flush=True)
        torch.manual_seed(SEED + trial.number)
        model = build_model("rsgm", num_joints=len(AmassJoints4Learning), **model_args)
        config = TrainConfig(max_iterations=TRIAL_ITERATIONS, valid_interval=int(np.gcd.reduce(EVAL_AT)), checkpoint_every=0,
                             device=device, progress=False, **train)
        trainer = Trainer(model, train_set, None, config)
        evals, best = {}, (-np.inf, None)
        clock = dict(start=time.perf_counter(), eval=0.0)

        def check(step, _):
            nonlocal best
            if not np.isfinite(trainer.history.loss[-1]):
                raise optuna.TrialPruned(f"non-finite loss at step {step}")
            if step not in EVAL_AT:
                return
            t0 = time.perf_counter()
            value, details = statistic(log_likelihood(trainer.ema.model, valid_poses, OBJECTIVE_STEPS))
            clock["eval"] += time.perf_counter() - t0
            evals[step] = details
            mark = ""
            if value > best[0]:
                best = (value, step)
                save_model(trainer.ema.model, weights_dir / f"trial_{trial.number:04d}")  # the EMA weights of the best step
                mark = " *"
            print(f"{tag} trial {trial.number:3d} step {step:6,}: log-likelihood {value:8.2f} (median {details['median']:.2f}){mark}", flush=True)
            trial.report(value, step)
            if not np.isfinite(value) or trial.should_prune():
                raise optuna.TrialPruned(f"log-likelihood {value:.2f} at step {step}")

        try:
            trainer.fit(callback=check)
            trial.set_user_attr("best_step", best[1])
            return best[0]
        except torch.OutOfMemoryError:
            trial.set_user_attr("oom", True)
            raise optuna.TrialPruned("out of GPU memory")
        finally:
            seconds = time.perf_counter() - clock["start"]
            trial.set_user_attr("parameters", model.num_parameters())
            trial.set_user_attr("seconds", seconds)
            trial.set_user_attr("eval_seconds", clock["eval"])
            trial.set_user_attr("evals", {str(k): v for k, v in evals.items()})
            if best[1] is not None:
                trial.set_user_attr("best_step", best[1])
            del trainer, model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
    return objective


def report(tag, n_trials):
    def callback(study, trial):
        trials = study.get_trials(deepcopy=False)
        done = sum(t.state in (States.COMPLETE, States.PRUNED) for t in trials)
        running = sum(t.state == States.RUNNING for t in trials)
        complete = [t for t in trials if t.state == States.COMPLETE]
        best = max(complete, key=lambda t: t.value) if complete else None
        a = trial.user_attrs
        value = f"{trial.value:.3f}" if trial.value is not None else "-"
        step = f" at {a['best_step']:,}" if a.get("best_step") else ""
        print(f"{tag} trial {trial.number:3d} {trial.state.name:8s} {value}{step}; {a.get('seconds', 0) / 60:.1f} min "
              f"(evaluations {a.get('eval_seconds', 0) / 60:.1f} min); {done}/{n_trials} done, {running} running, "
              f"{max(0, n_trials - done)} to go; best {f'{best.value:.3f} (trial {best.number})' if best else '-'}", flush=True)
    return callback


# ---- study, workers, summary ---------------------------------------------------------

def storage(study_dir):
    return optuna.storages.RDBStorage(f"sqlite:///{study_dir / 'optuna.db'}", engine_kwargs={"connect_args": {"timeout": 120}})


def create_study(args, worker=0):
    return optuna.create_study(
        study_name=args.study, storage=storage(args.study_dir), load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED + worker, multivariate=True, constant_liar=True, n_startup_trials=N_STARTUP),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
    )


def run_worker(args):
    tag = f"[w{args.worker}]"
    study = create_study(args, args.worker)
    train_set, valid_poses = load_data(args.rotations)
    print(f"{tag} {args.study}: {len(train_set):,} training poses, {len(valid_poses)} validation poses, device {args.device}", flush=True)
    weights_dir = args.study_dir / f"trials_{args.study}"
    stop = optuna.study.MaxTrialsCallback(args.n_trials, states=(States.COMPLETE, States.PRUNED))
    study.optimize(make_objective(train_set, valid_poses, args.device, weights_dir, tag), callbacks=[report(tag, args.n_trials), stop],
                   catch=(RuntimeError,))


def summary(args):
    study = create_study(args)
    rows = []
    for t in study.get_trials(deepcopy=False):
        a = t.user_attrs
        rows.append(dict(number=t.number, state=t.state.name, value=t.value, best_step=a.get("best_step"), **t.params,
                         M=(a.get("parameters") or np.nan) / 1e6, minutes=a.get("seconds", np.nan) / 60, oom=a.get("oom", False)))
    if not rows:
        print("no trials yet")
        return
    df = pd.DataFrame(rows).sort_values("value", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.head(15).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nstates: {df.state.value_counts().to_dict()}")
    complete = [t for t in study.trials if t.state == States.COMPLETE]
    if complete:
        best = max(complete, key=lambda t: t.value)
        print(f"\nbest: trial {best.number}, log-likelihood {best.value:.3f} at step {best.user_attrs.get('best_step')}")
        print("overrides (scripts/train_model.py, rsgm_train.ipynb), with train.max_iterations at the best step:")
        print(json.dumps(as_overrides(best.params) + [f"train.max_iterations={best.user_attrs.get('best_step')}"], indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rotations", default=ROTATIONS)
    parser.add_argument("--study", default=STUDY_NAME)
    parser.add_argument("--study-dir", default=STUDY_DIR)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--summary", action="store_true", help="only print the trials so far")
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)  # set for the worker processes
    args = parser.parse_args()
    args.study_dir = Path(args.study_dir).expanduser()
    args.study_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)

    if args.summary:
        summary(args)
        return
    if args.worker is not None:
        run_worker(args)
        return

    study = create_study(args)
    if not study.trials:
        study.enqueue_trial(REFERENCE)
    done = sum(t.state in (States.COMPLETE, States.PRUNED) for t in study.trials)
    print(f"{args.study}: {done}/{args.n_trials} trials done; {args.workers} worker(s) on {args.device}", flush=True)
    start = time.perf_counter()
    base = [sys.executable, os.path.abspath(__file__), "--rotations", args.rotations, "--study", args.study,
            "--study-dir", str(args.study_dir), "--n-trials", str(args.n_trials), "--device", args.device]
    workers = [subprocess.Popen(base + ["--worker", str(k)]) for k in range(args.workers)]
    try:
        for w in workers:
            w.wait()
    except KeyboardInterrupt:  # Ctrl-C reaches the workers too; wait for them to stop
        for w in workers:
            w.wait()
    print(f"\n{(time.perf_counter() - start) / 3600:.1f} h in this run\n")
    summary(args)


if __name__ == "__main__":
    main()
