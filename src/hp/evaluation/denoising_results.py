"""The results of motion denoising (hp/evaluation/denoising.py) in one form, for the notebooks motion_denoising.ipynb
and denoising_fronts.ipynb: the rows of the caches (scripts/evaluate_denoising.py, explore/evaluation/denoising_aniso)
and the trials of the Optuna studies (scripts/tune_denoising.py). Nothing is recomputed.

A record is a dict: source, split ("valid" / "test"), sigma, label (the row of the cache, or the study and trial),
model (the run folder or checkpoint), method ("dflow" / "dposer"), settings (the full settings of the method), steps
(optimizer steps), nfe (network calls: one per SDS step, steps x ODE steps for D-Flow, 0 without a prior), mpjpe,
mpvpe, input_mpjpe (mm, means over the frames), and for cache rows names and per_sequence (dict of [N] arrays).

    rows = load_cache("~/data/experiments/denoising/cache", split="valid")
    trials = load_study("~/data/experiments/denoising/optuna/rsgm_sds_40mm.db")
"""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from hp.evaluation.denoising import DPOSER_SETTINGS, METRICS_VERSION, POSEDFLOW_SETTINGS

METHODS = {"dflow": "D-Flow", "dposer": "DPoser's optimization (with an RSGM: score distillation)"}


def model_of(spec: dict) -> str:
    """The weights a spec uses: the run folder, the checkpoint, or "none" (DPoser's optimization without a prior)."""
    return spec.get("run") or spec.get("posedflow") or "none"


def budget(spec: dict, sigma: float, settings: dict) -> Dict[str, float]:
    """Optimizer steps and network calls of a spec at a noise level, from the defaults of its method and `settings`.
    D-Flow runs its ODE (default 100 Euler steps) at every step; SDS calls the network once per step."""
    method = spec.get("method", "dflow")
    if method == "dflow":
        conf = {**POSEDFLOW_SETTINGS[sigma], **settings}
        ode = dict(spec.get("prior") or {}).get("ode_steps", 100)
        return {"steps": conf["steps"], "nfe": conf["steps"] * ode}
    conf = {**DPOSER_SETTINGS[sigma], **settings}
    steps = conf["iterations"] * conf["steps_per_iter"]
    return {"steps": steps, "nfe": 0 if model_of(spec) == "none" else steps}


def full_settings(spec: dict, sigma: float, settings: dict) -> dict:
    """The settings a row ran with: the defaults of its method updated by its own."""
    defaults = POSEDFLOW_SETTINGS if spec.get("method", "dflow") == "dflow" else DPOSER_SETTINGS
    return {**defaults[sigma], **settings}


def folder_of(label: str) -> str:
    """The folder of a row in the cache (hp.evaluation.denoising.cache_path)."""
    return label.replace("/", "_").replace(" ", "_")


def _plain(x):
    return json.loads(json.dumps(x))


def load_cache(cache: Union[str, Path], split: Optional[str] = None, source: Optional[str] = None,
               models: Optional[Dict[str, dict]] = None) -> List[dict]:
    """The rows of a cache (cache/<label>/<split>_<sigma>mm.pt), one record per file of the current METRICS_VERSION;
    `split` None: both. The label is the one of `models` ({label: spec}, e.g. evaluate_denoising.MODELS) with that
    folder, else the folder name with "_" read as " " (the folder name loses "/" and "_"). A file made with another
    spec than models[label] (e.g. settings tuned after it) is labelled "<label> [other settings]"."""
    cache = Path(cache).expanduser()
    models = models or {}
    named = {folder_of(label): label for label in models}
    out = []
    for path in sorted(cache.glob("*/*mm.pt")):
        saved = torch.load(path, weights_only=True)
        key = json.loads(saved["key"])
        if key.get("version") != METRICS_VERSION or (split is not None and key["split"] != split):
            continue
        spec = key["model"]["spec"]
        sigma = float(key["sigma"])
        label = named.get(path.parent.name, path.parent.name.replace("_", " "))
        if label in models and _plain(models[label]) != spec:
            label += " [other settings]"
        metrics = {k: v.double().numpy() for k, v in saved["metrics"].items()}
        per_sequence = {k: np.nanmean(v, axis=1) for k, v in metrics.items()}
        out.append({
            "source": source or cache.name, "split": key["split"], "sigma": sigma,
            "label": label, "folder": path.parent.name,
            "file": str(path),
            "model": model_of(spec), "method": spec.get("method", "dflow"), "spec": spec,
            "settings": full_settings(spec, sigma, key["settings"]), **budget(spec, sigma, key["settings"]),
            **{k: float(np.nanmean(v)) for k, v in metrics.items()},
            "names": list(key["names"]), "per_sequence": per_sequence,
        })
    return out


def load_study(path: Union[str, Path]) -> List[dict]:
    """The complete trials of every Optuna study in an .db file (scripts/tune_denoising.py): validation records with
    the model, the settings of the trial (t_min = t_min_fraction * t_max), MPJPE (the objective), MPVPE, the
    minutes of the trial, and study, trial (its number) and params (the values Optuna drew)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{Path(path).expanduser()}"
    out = []
    for summary in optuna.get_all_study_summaries(storage):
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        config = study.user_attrs.get("config", {})
        sigma = float(config["sigma"])
        spec = {"run": config["model"]} if not str(config["model"]).endswith(".ckpt") else {
            "posedflow": config["model"]}
        spec["method"] = config.get("method", "dposer")
        for trial in study.trials:
            if trial.value is None or trial.state.name != "COMPLETE":
                continue
            values = {**config.get("fixed", {}), **trial.params}
            if "t_min_fraction" in values:
                values["t_min"] = values.pop("t_min_fraction") * values["t_max"]
            out.append({
                "source": f"{summary.study_name}:{trial.number}", "split": "valid", "sigma": sigma,
                "label": f"{summary.study_name}, trial {trial.number}", "model": model_of(spec),
                "method": spec["method"], "spec": spec, "settings": full_settings(spec, sigma, values),
                **budget(spec, sigma, values), "mpjpe": float(trial.value),
                "mpvpe": float(trial.user_attrs.get("mpvpe", np.nan)), "input_mpjpe": np.nan,
                "minutes": float(trial.user_attrs.get("minutes", np.nan)), "sequences": config.get("sequences"),
                "study": summary.study_name, "trial": trial.number, "params": dict(trial.params),
            })
    return out


def load(caches: Iterable[Union[str, Path]] = (), studies: Iterable[Union[str, Path]] = (),
         split: Optional[str] = None, models: Optional[Dict[str, dict]] = None) -> List[dict]:
    """The records of several caches and study files; the trials only for split None or "valid". A row (label,
    split, noise level) found in several caches (e.g. copies from the servers) is taken from the first."""
    out, seen = [], set()
    for c in caches:
        for r in load_cache(c, split, models=models):
            if (r["label"], r["split"], r["sigma"]) not in seen:
                seen.add((r["label"], r["split"], r["sigma"]))
                out.append(r)
    if split in (None, "valid"):
        out += [r for s in studies for r in load_study(s)]
    return out


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> Dict[str, float]:
    """a - b over the same sequences ([N] errors of two methods): the mean difference, its bootstrap interval over
    the sequences (percentile, `level`), and the fraction of sequences where a is lower."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.shape[0] == 0:
        raise ValueError("no sequences to compare.")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "a_better": float((d < 0).mean()),
            "n": int(len(d))}


def dataset_of(name: str) -> str:
    """The group of a sequence name: the dataset for validation (HDM05/...), the subject for HumanEva (S1/...)."""
    parts = name.split("/")
    return parts[1] if parts[0] == "HumanEva" and len(parts) > 2 else parts[0]


def by_group(record: dict, metric: str = "mpjpe") -> Dict[str, float]:
    """The mean of a metric of a cache record over the sequences of each group (dataset_of)."""
    groups: Dict[str, List[float]] = {}
    for name, value in zip(record["names"], record["per_sequence"][metric]):
        groups.setdefault(dataset_of(name), []).append(float(value))
    return {g: float(np.mean(v)) for g, v in sorted(groups.items())}


def find(records: Sequence[dict], label: str, sigma: float, split: str) -> Optional[dict]:
    """The cache record of a row (label as in the cache folder, with spaces) at a noise level and split."""
    hits = [r for r in records if r["label"] == label and r["sigma"] == sigma and r["split"] == split
            and "per_sequence" in r]
    return hits[0] if hits else None
