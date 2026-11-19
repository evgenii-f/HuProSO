"""Choose the settings of motion denoising with Optuna, on the VALIDATION sequences (never the test sequences).

    python scripts/tune_denoising.py config/denoising_tuning/<name>.yaml --device cuda:0
    python scripts/tune_denoising.py config/denoising_tuning/<name>.yaml --summary

A study is a YAML file in config/denoising_tuning, named as the file:

    model: ~/data/experiments/rsgm_posedflow_small_s10/run_0001   # a run folder, or posedflow: <checkpoint>
    method: dflow            # D-Flow (hp.models.motion_denoising.dflow_denoise) or dposer (DPoser's optimization)
    sigma: 0.04              # the noise (m)
    sequences: 100           # validation sequences (hp.evaluation.denoising.sequences("valid", n_valid=...))
    n_trials: 30
    params:                  # searched: {low, high, log} a float, {low, high, int: true} an integer, [a, b] a choice
      lr: {low: 0.02, high: 1.0, log: true}
    fixed: {sampler: heun, schedule: edm, ode_steps: 25}           # the same in every trial
    first: [{lr: 0.25}]                                            # trials enqueued into a new study

For dflow, steps, lr, temp_weight, running_weight, alpha and optimizer are settings of the denoising and the other
values options of the D-Flow prior (sampler, schedule, ode_steps, invert). For dposer, t_min_fraction gives
t_min = t_min_fraction * t_max (the window stays ordered). The objective is the MPJPE (mm) of the
validation sequences; every trial sees the same noise and seeds. The study is in STUDY_DIR / <name>.db with its
configuration and git commit; the best settings are written to <name>.json next to it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import optuna
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hp.evaluation.denoising import Body, gt_joints, metrics, noisy_joints, run, sequences, summary  # noqa: E402
from tune_completion import git_commit, open_study, suggest  # noqa: E402

STUDY_DIR = "~/data/experiments/denoising/optuna"
SETTING_KEYS = ("steps", "lr", "temp_weight", "running_weight", "alpha", "optimizer",  # dflow
                "iterations", "steps_per_iter", "t_max", "t_min", "sds_weight")      # dposer
KEYS = ("model", "method", "sigma", "sequences", "n_trials", "params", "fixed", "first")
BATCH = 10


def load_config(path: Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    missing, unknown = set(KEYS[:6]) - set(config), set(config) - set(KEYS)
    if missing or unknown:
        raise ValueError(f"{path}: missing {sorted(missing)}, unknown {sorted(unknown)}.")
    if config["method"] not in ("dflow", "dposer"):
        raise ValueError(f"{path}: method must be dflow or dposer.")
    config.setdefault("fixed", {})
    config.setdefault("first", [])
    return config


def spec_of(model: str) -> dict:
    if model == "none":
        return {}
    return {"posedflow": model} if model.endswith(".ckpt") else {"run": model}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    config, name = load_config(args.config), args.config.stem
    study_dir = Path(STUDY_DIR).expanduser()
    commit = git_commit()
    study = open_study(name, study_dir, ["minimize"])
    made = {k: v for k, v in config.items() if k != "n_trials"}
    if "config" not in study.user_attrs:
        study.set_user_attr("config", made)
        study.set_user_attr("git", commit)
        for values in config["first"]:
            study.enqueue_trial(values)
    elif json.dumps(study.user_attrs["config"], sort_keys=True) != json.dumps(made, sort_keys=True):
        raise SystemExit(f"study {name} was made with another configuration; name the file differently.")

    if not args.summary:
        from hp.evaluation.denoising import build

        body = Body(device=args.device)
        gt, _ = sequences("valid", n_valid=config["sequences"])
        noisy = noisy_joints(gt_joints(gt, body), config["sigma"])
        model = build(spec_of(config["model"]), args.device)

        def objective(trial):
            values = dict(config["fixed"])
            values.update({key: suggest(trial, key, spec) for key, spec in config["params"].items()})
            if "t_min_fraction" in values:
                values["t_min"] = values.pop("t_min_fraction") * values["t_max"]
            settings = {k: v for k, v in values.items() if k in SETTING_KEYS}
            prior = {k: v for k, v in values.items() if k not in SETTING_KEYS}
            trial.set_user_attr("git", commit)
            start = time.perf_counter()
            pred = run(model, gt, noisy, config["sigma"], body, config["method"], BATCH, progress=False,
                       prior=prior or None, **settings)
            s = summary(metrics(pred, gt, noisy, body))
            trial.set_user_attr("mpvpe", s["mpvpe"])
            trial.set_user_attr("minutes", (time.perf_counter() - start) / 60)
            return s["mpjpe"]

        remaining = config["n_trials"] - len([t for t in study.trials if t.state.is_finished()])
        if remaining > 0:
            study.optimize(objective, n_trials=remaining)

    frame = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    if frame.empty:
        print("no trials yet")
        return 0
    print(frame.sort_values("value").head(15).to_string(index=False))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        best = study.best_trial
        print(f"\nbest: trial {best.number}, {best.value:.2f} mm, {best.params}")
        (study_dir / f"{name}.json").write_text(json.dumps(
            {"study": name, "trials": len(done), "config": made, "value": best.value, "params": best.params,
             "attrs": best.user_attrs, "git": study.user_attrs.get("git")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
