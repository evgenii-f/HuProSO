"""Results of pose completion on the VALIDATION poses in one form, for comparing methods and models on their fronts
(completion_fronts.ipynb, hp.evaluation.fronts): the JSON lines of the scripts of
explore/evaluation/completion_diagnostics (dflow_valid.py, inpaint_valid.py) and the trials of the Optuna studies of
scripts/tune_completion.py.

A record (dict) is one setting of one method on one model, evaluated on the first `poses` of the 1000 fixed validation
poses, K = 10, all four occlusions:

    source      the file (and study) it comes from
    model       a name (that of dflow_valid.MODELS) or, for a study, its run folder
    run         the run folder of the model, if known
    method      "inpaint", "dflow" or "dposer" (DPoser's optimization; with an RSGM, score distillation)
    setting     a short label of the settings (the variant, the ODE, or "trial <number>")
    options     the settings
    poses       the number of validation poses
    mean, best_of_k, apd      the means over the occlusions of MPVPE mean (mm), best of K (mm), APD (cm)
    occlusions  {occlusion: {"mean", "best_of_k", "apd"}}
    seconds     the time of all four occlusions (None if not known)
    device      where it ran ("cuda:0", "mps", ...; with the host if known; None if not known)

Only records of the same poses compare: the numbers on 100 poses are not those on 1000.
"""
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

from hp.evaluation.completion import OCCLUSIONS

METHODS = {"inpaint": "inpainting", "dflow": "D-Flow", "dposer": "DPoser's optimization (with an RSGM: score distillation)"}
_KEYS = (("mean", "mpvpe_mean"), ("best_of_k", "mpvpe_min"), ("apd", "apd"))
# studies made before tune_completion.py recorded its configuration: completion_<method>_<run folder>_v<poses>
_OLD_STUDY = re.compile(r"completion_(dflow|dposer|inpaint)_(.+)_v(\d+)$")


def _record(source, model, run, method, setting, options, poses, cells, seconds, device) -> dict:
    occlusions = {o: {name: float(cells[o][key]) for name, key in _KEYS} for o in OCCLUSIONS}
    return dict(source=source, model=model, run=run, method=method, setting=setting, options=options, poses=poses,
                **{name: float(np.mean([occlusions[o][name] for o in OCCLUSIONS])) for name, _ in _KEYS},
                occlusions=occlusions, seconds=seconds, device=device)


def load_jsonl(path: Union[str, Path]) -> List[dict]:
    """The records of a .jsonl file of dflow_valid.py (method dflow, setting its ODE), inpaint_valid.py (method
    inpaint, setting its variant), or of another script whose lines name their method ("method", optional "setting"
    and "options"; e.g. the baselines with their own models)."""
    path = Path(path).expanduser()
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "variant" in row:
            method, setting, options = "inpaint", row["variant"], row.get("options", {})
        elif "ode" in row:
            method, setting, options = "dflow", row["ode"], {"ode": row["ode"]}
        elif row.get("method") in METHODS:  # any other script: the method named, its settings optional
            method, options = row["method"], row.get("options") or {}
            setting = row.get("setting") or "default"
        else:
            raise ValueError(f"{path}: a line without a known method (variant of inpaint_valid.py, ode of "
                             f"dflow_valid.py, or method one of {list(METHODS)}): {line[:120]}")
        run = (row.get("spec") or {}).get("run")
        out.append(_record(path.name, row["model"], run, method, setting, options, row["poses"], row["occlusions"],
                           row.get("seconds"), row.get("device")))
    return out


def load_study(path: Union[str, Path], name: Optional[str] = None) -> List[dict]:
    """The records of the finished trials of an Optuna study of tune_completion.py in its SQLite file (name: the study
    in it, default the name of the file). The model, method and poses come from the configuration the study recorded,
    or, for older studies, from their name (completion_<method>_<run folder>_v<poses>)."""
    import optuna  # only here: the rest of this module does not need it

    path = Path(path).expanduser()
    name = name or path.stem
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.load_study(study_name=name, storage=f"sqlite:///{path}")
    config = study.user_attrs.get("config")
    if config:
        run, method, poses, fixed = config["model"], config["method"], config["poses"], config.get("fixed") or {}
    else:
        match = _OLD_STUDY.match(name)
        if not match:
            raise ValueError(f"study {name} has no recorded configuration and a name not of the form "
                             "completion_<method>_<run folder>_v<poses>")
        method, folder, poses = match.group(1), match.group(2), int(match.group(3))
        run, fixed = f"~/data/experiments/{folder}/run_0001", {}
    out = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        attrs = trial.user_attrs
        if not all(f"{o}_{key}" in attrs for o in OCCLUSIONS for _, key in _KEYS):
            continue
        cells = {o: {key: attrs[f"{o}_{key}"] for _, key in _KEYS} for o in OCCLUSIONS}
        minutes = attrs.get("minutes")
        device = attrs.get("device")
        if device and attrs.get("host"):
            device = f"{attrs['host']} {device}"
        out.append(_record(f"{path.name}:{name}", run, run, method, f"trial {trial.number}",
                           {**fixed, **trial.params}, poses, cells, None if minutes is None else 60 * minutes, device))
    return out


def load(paths: Iterable[Union[str, Path]]) -> List[dict]:
    """The records of every file of `paths`: .jsonl of the validation scripts, .db of Optuna studies (each study in
    it); files that do not exist are skipped."""
    out = []
    for path in paths:
        path = Path(path).expanduser()
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            out += load_jsonl(path)
        elif path.suffix == ".db":
            import optuna
            for summary in optuna.get_all_study_summaries(f"sqlite:///{path}"):
                if summary.n_trials:
                    out += load_study(path, summary.study_name)
        else:
            raise ValueError(f"{path}: neither .jsonl nor .db")
    return out


def at_occlusion(record: dict, occlusion: Optional[str]) -> dict:
    """The record with mean, best_of_k and apd of one occlusion (None: the means over the occlusions, as it is)."""
    if occlusion is None:
        return record
    return {**record, **record["occlusions"][occlusion]}
