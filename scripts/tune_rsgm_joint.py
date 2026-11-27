"""Search of the training of the RSGM score network for completion and generation together (Optuna, two objectives).

    python scripts/tune_rsgm_joint.py --device cuda:0             # one worker; start more on other GPUs, same study
    python scripts/tune_rsgm_joint.py --summary                   # the trials so far, the Pareto front, the choice
    python scripts/tune_rsgm_joint.py --dry-run                   # the reference trial's plan, nothing trained
    python scripts/tune_rsgm_joint.py --smoke --device mps        # one tiny trial end to end, in <study dir>/smoke

A trial trains `scripts/train_rsgm_posedflow.py --config rdw` (timeres residual network, PoseD-Flow's AMASS c10,
seed 10) with the parameters of the trial, then evaluates its checkpoint with the lowest validation loss
(`hp.training.best_checkpoint`, a checkpoint every 1000 steps) with `scripts/evaluate.py` on the validation:

- objective 1, minimised: inpainting mean MPVPE (mm) at the fixed sampler point POINT (that of fact_RDW s10),
  1000 validation poses, K = 10, mean over the 4 occlusions;
- objective 2, minimised: Wasserstein of generation (deg per joint), the model's own sampler, 8601 validation poses;
- constraint: inpainting APD > APD_MIN cm (a trial below is infeasible for the sampler, not dropped);
- every trial sees the same number of training poses (POSES = 30k steps x 4096): the steps follow the batch;
- networks above MAX_PARAMS parameters are pruned before training and do not count towards --n-trials.

The first trial is fact_RDW (REFERENCE). The choice (--summary), fixed before the study: among the feasible trials
of the Pareto front whose W is not above the reference's, the one with the lowest inpainting mean; it and the next
two go to three seeds on the validation, then to one test.

Workers are processes that share the SQLite database of the study (same machine). A stopped worker leaves its trial
RUNNING; the same command continues with new trials, and the evaluation caches (labels "<study> trial <n>") make a
re-evaluated trial cheap. The training of a stopped trial resumes from its last checkpoint if its number is enqueued
again (not done automatically).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# ---- settings, fixed before the study -------------------------------------------------
STUDY_DIR = "~/data/experiments/rsgm_joint"
STUDY_NAME = "rsgm_joint_inpaint_w"
N_TRIALS = 40                       # COMPLETE trials (pruned and failed ones do not count)
N_STARTUP = 10                      # random trials before TPE models the space
SEED = 10                           # the training seed of every trial (the sampler's seed is SEED + worker)
CONFIG = "rdw"                      # of scripts/train_rsgm_posedflow.py: COMMON + residual timeres, data on the device
POSES = 30_000 * 4096               # training poses of a trial
CHECKPOINT_EVERY = 1000
MAX_PARAMS = 30e6
APD_MIN = 5.0                       # cm
POINT = "RDW b0 factorial/rsgm_posedflow_fact_RDW_s10, rsgm_inpaint_mean_apd_rdw_b0 t99"
SPACE = dict(net_width=[512, 1024, 1536, 2048], net_blocks=(2, 6), dropout=[0.0, 0.05, 0.1, 0.2],
             lr=(5e-4, 3e-3), batch_size=[2048, 4096, 8192], weight_decay=(1e-6, 1e-3))
REFERENCE = dict(net_width=1024, net_blocks=2, dropout=0.1, lr=2e-3, batch_size=4096, weight_decay=1e-5)
# ----------------------------------------------------------------------------------------


def suggest(trial) -> dict:
    s = SPACE
    return dict(net_width=trial.suggest_categorical("net_width", s["net_width"]),
                net_blocks=trial.suggest_int("net_blocks", *s["net_blocks"]),
                dropout=trial.suggest_categorical("dropout", s["dropout"]),
                lr=trial.suggest_float("lr", *s["lr"], log=True),
                batch_size=trial.suggest_categorical("batch_size", s["batch_size"]),
                weight_decay=trial.suggest_float("weight_decay", *s["weight_decay"], log=True))


def steps_of(batch_size: int) -> int:
    return POSES // batch_size


def overrides(p: dict, smoke: bool = False) -> list:
    """The parameters of a trial as key=value settings of scripts/train_rsgm_posedflow.py (after --config CONFIG)."""
    tiny = ["train.max_iterations=200", "train.checkpoint_every=100", "train.valid_interval=100"] if smoke else []
    return [f"model.args.net_width={p['net_width']}", f"model.args.net_blocks={p['net_blocks']}",
            f"model.args.dropout={p['dropout']}", f"train.lr={p['lr']}", f"train.batch_size={p['batch_size']}",
            f"train.weight_decay={p['weight_decay']}", f"train.max_iterations={steps_of(p['batch_size'])}",
            f"train.checkpoint_every={CHECKPOINT_EVERY}", "train.keep_best=1", "train.keep_last=1", *tiny]


def parameters(p: dict) -> int:
    """The number of parameters of the network of a trial, built as the training builds it."""
    import train_rsgm_posedflow as T
    from omegaconf import OmegaConf

    from hp.data.amass_cfg import AmassJoints4Learning
    from hp.models import build_model
    from hp.training.config import load_config

    cfg = load_config(["model=rsgm", *T.COMMON, *T.CONFIGS[CONFIG], *overrides(p)])
    model = build_model(cfg.model.name, num_joints=len(AmassJoints4Learning), **OmegaConf.to_container(cfg.model.args))
    return sum(t.numel() for t in model.parameters())


def inpainting_numbers(parts: dict) -> dict:
    """Mean over the occlusions of mean, best of k (mm) and APD (cm), from {occlusion: per-pose metrics}."""
    from hp.evaluation.completion import summary

    rows = [summary(m) for m in parts.values()]
    return {key: sum(r[name] for r in rows) / len(rows)
            for key, name in (("mean", "mpvpe_mean"), ("best_of_k", "mpvpe_min"), ("apd", "apd"))}


def choose(trials: list, reference_w: float) -> list:
    """The rule of the study: feasible trials of the Pareto front with W <= the reference's, by inpainting mean.
    trials: dicts with number, mean, w, apd."""
    feasible = [t for t in trials if t["apd"] > APD_MIN]
    front = [t for t in feasible if not any(o["mean"] <= t["mean"] and o["w"] <= t["w"] and
                                            (o["mean"] < t["mean"] or o["w"] < t["w"]) for o in feasible)]
    return sorted((t for t in front if t["w"] <= reference_w), key=lambda t: t["mean"])


# ---- a trial ----------------------------------------------------------------------------

def evaluate(argv: list) -> dict:
    import evaluate as E

    code, got = E.call(None, argv, E.build_parser())
    if code != 0 or not got:
        raise RuntimeError(f"evaluate.py {' '.join(argv[:3])} ... ended with code {code}")
    return got


def make_objective(args, tag: str):
    import optuna

    from hp.training import best_checkpoint

    def objective(trial):
        p = suggest(trial)
        n = parameters(p)
        trial.set_user_attr("parameters", n)
        trial.set_user_attr("steps", steps_of(p["batch_size"]))
        print(f"{tag} trial {trial.number:3d} START {p}, {n / 1e6:.1f}M parameters", flush=True)
        if n > MAX_PARAMS:
            trial.set_user_attr("too_large", True)
            raise optuna.TrialPruned(f"{n / 1e6:.1f}M parameters > {MAX_PARAMS / 1e6:.0f}M")
        start = time.perf_counter()
        root = args.study_dir / "trials" / f"trial_{trial.number:04d}"
        root.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(ROOT / "scripts" / "train_rsgm_posedflow.py"), "--config", CONFIG,
                   "--seeds", str(SEED), "--device", args.device, "--run-root", str(root), "--check", "0",
                   *overrides(p, args.smoke)]
        with open(root / "train.log", "a") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        run = root / f"rsgm_posedflow_{CONFIG}_s{SEED}" / "run_0001"
        if code != 0 or not (run / "weights_ema.pt").is_file():
            raise RuntimeError(f"training ended with code {code}, see {root / 'train.log'}")
        best = best_checkpoint(run) or run
        trial.set_user_attr("checkpoint", str(best))
        trial.set_user_attr("train_minutes", (time.perf_counter() - start) / 60)
        label = f"{args.study} trial {trial.number}"
        print(f"{tag} trial {trial.number:3d} trained in {(time.perf_counter() - start) / 60:.1f} min; "
              f"evaluating {best.name}", flush=True)

        _, row = evaluate(["generation", "--run", f"{label}={best}", "--like", "own", "--split", "valid", "--quick",
                           "--device", args.device])[label]
        w = row["wasserstein"]
        for key in ("wasserstein", "recall", "precision", "FID", "APD"):
            if key in row:
                trial.set_user_attr(key, row[key])
        inp = f"{label} inp"
        _, parts = evaluate(["completion", "--run", f"{inp}={best}", "--like", POINT, "--split", "valid",
                             "--device", args.device, *(["--poses", "20"] if args.smoke else [])])[inp]
        numbers = inpainting_numbers(parts)
        for key, value in numbers.items():
            trial.set_user_attr(f"inpaint_{key}", value)
        trial.set_user_attr("constraint", [APD_MIN - numbers["apd"]])
        trial.set_user_attr("minutes", (time.perf_counter() - start) / 60)
        print(f"{tag} trial {trial.number:3d} DONE inpainting {numbers['mean']:.2f} / {numbers['best_of_k']:.2f} mm, "
              f"APD {numbers['apd']:.2f} cm; W {w:.2f} deg; {(time.perf_counter() - start) / 60:.1f} min", flush=True)
        return numbers["mean"], w

    return objective


# ---- study --------------------------------------------------------------------------------

def create_study(args, worker: int = 0):
    import optuna

    sampler = optuna.samplers.TPESampler(seed=SEED + worker, multivariate=True, constant_liar=True,
                                         n_startup_trials=N_STARTUP,
                                         constraints_func=lambda t: t.user_attrs.get("constraint", [0.0]))
    storage = optuna.storages.RDBStorage(f"sqlite:///{args.study_dir / 'optuna.db'}",
                                         engine_kwargs={"connect_args": {"timeout": 120}})
    study = optuna.create_study(study_name=args.study, storage=storage, load_if_exists=True,
                                directions=["minimize", "minimize"], sampler=sampler)
    if not study.trials:
        study.enqueue_trial(REFERENCE)
    return study


def rows_of(study) -> list:
    import optuna

    rows = []
    for t in study.get_trials(deepcopy=False):
        a = t.user_attrs
        row = dict(number=t.number, state=t.state.name, **t.params, M=a.get("parameters", 0) / 1e6)
        if t.state == optuna.trial.TrialState.COMPLETE:
            row.update(mean=t.values[0], w=t.values[1], best_of_k=a.get("inpaint_best_of_k"),
                       apd=a.get("inpaint_apd"), recall=a.get("recall"), precision=a.get("precision"),
                       minutes=a.get("minutes"))
        rows.append(row)
    return rows


def summary(args) -> None:
    import pandas as pd

    study = create_study(args)
    rows = rows_of(study)
    if not rows:
        print("no trials yet")
        return
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    done = [r for r in rows if r["state"] == "COMPLETE"]
    reference = next((r for r in done if r["number"] == 0), None)
    if reference is None:
        print("\nthe reference (trial 0) is not complete yet: no choice")
        return
    chosen = choose(done, reference["w"])
    print(f"\nreference: inpainting {reference['mean']:.2f} mm, APD {reference['apd']:.2f} cm, W {reference['w']:.2f}")
    print("chosen by the rule (feasible Pareto trials with W <= the reference's, by inpainting mean): "
          + (", ".join(f"trial {t['number']} ({t['mean']:.2f} mm, W {t['w']:.2f})" for t in chosen[:3]) or "none"))
    for t in chosen[:3]:
        p = {k: t[k] for k in REFERENCE}
        print(f"  trial {t['number']}: {CONFIG} with " + " ".join(overrides(p)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default=STUDY_NAME)
    parser.add_argument("--study-dir", default=STUDY_DIR)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--worker", type=int, default=0, help="the sampler's seed offset of this process")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="one trial of 200 steps, inpainting on 20 poses")
    args = parser.parse_args()
    args.study_dir = Path(args.study_dir).expanduser() / ("smoke" if args.smoke else "")
    if args.smoke:  # its own study name: the evaluation caches are keyed by the label "<study> trial <n>"
        args.n_trials, args.study = 1, args.study + "_smoke"

    if args.dry_run:
        for name, p in (("reference", REFERENCE), ("largest allowed width x blocks", dict(REFERENCE, net_width=2048,
                                                                                         net_blocks=3))):
            print(f"{name}: {p}, {parameters(p) / 1e6:.1f}M parameters, {steps_of(p['batch_size']):,} steps")
            print("  train_rsgm_posedflow.py --config", CONFIG, " ".join(overrides(p)))
        print(f"objectives: inpainting mean at {POINT!r}, W of generation (own sampler); constraint APD > {APD_MIN}")
        return 0
    if args.summary:
        summary(args)
        return 0

    import optuna

    args.study_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[w{args.worker}]"
    study = create_study(args, args.worker)
    stop = optuna.study.MaxTrialsCallback(args.n_trials, states=(optuna.trial.TrialState.COMPLETE,))
    print(f"{tag} {args.study} in {args.study_dir}, device {args.device}, {args.n_trials} complete trials wanted",
          flush=True)
    study.optimize(make_objective(args, tag), callbacks=[stop], catch=(RuntimeError,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
