"""The settings of the completion for `scripts/evaluate.py all`: ready (those of a reference, one setting), a grid (10
or 20 settings) or an Optuna study (60-120 settings). Which one is said on the command line, never taken silently; here
are the plan of a grid or study, its announcement and the rule that picks a point (the run is tune() of
scripts/tune_completion.py)."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from hp.evaluation.protocol_cli import UsageError, short_hash

ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = ROOT / "config" / "completion_tuning"
MODES = ("ready", "grid", "study")
MENU = """choose how the completion of this model is set (no default):
  --like                 ready: the settings of the reference of its architecture, ONE setting per procedure (--methods)
  --tune grid            inpaint only: 10 guidance values (20 for an unknown architecture), then the best
  --tune study           an Optuna study: 120 settings (inpaint), 60 (dflow), hours; then the best
  --tune inpaint=grid dflow=ready      per procedure"""


@dataclass
class Plan:
    """A grid or a study of one model: the configuration of tune_completion.py under a name of its own."""
    method: str
    mode: str
    name: str
    config: dict
    floor: float
    minutes: Optional[float] = None  # of one setting, recorded for the reference of the same architecture
    hours: Optional[float] = None    # of the whole study of that reference
    grid: List[dict] = field(default_factory=list)  # the settings of a grid


def modes(tune: Optional[Iterable[str]], methods: Iterable[str]) -> Dict[str, str]:
    """{method: mode} from --tune [METHOD=]MODE ...: a mode without a method is that of all the others."""
    methods, chosen, rest = list(methods), {}, None
    for item in tune or []:
        method, sep, mode = item.rpartition("=")
        if mode not in MODES or (sep and method not in methods):
            raise UsageError(f"--tune {item}: [METHOD=]{'|'.join(MODES)}, METHOD of --methods ({', '.join(methods)})")
        if sep:
            chosen[method] = mode
        else:
            rest = mode
    result = {m: chosen.get(m, rest) for m in methods}
    if missing := [m for m, mode in result.items() if mode is None]:
        raise UsageError(f"--tune says nothing about {', '.join(missing)}\n{MENU}")
    if result.get("dflow") == "grid":
        raise UsageError("there is no grid for D-Flow (its best points needed 40-60 trials): "
                         "--tune inpaint=grid dflow=ready (or dflow=study), or --methods inpaint")
    return result


def make_plan(references: dict, method: str, mode: str, run: str, row: dict, own: bool,
              n_trials: Optional[int] = None, batch: Optional[int] = None) -> Plan:
    """The grid or study of the model `run`; row: the reference set of its method, `own`: of its architecture. The
    grid has the steps of that set (an unknown architecture: 400 and 200)."""
    template = TEMPLATES / f"rsgm_{method}_{mode}_template.yaml"
    config = {"model": run, **yaml.safe_load(template.read_text())}
    if batch is not None:
        config["batch"] = batch
    grid = []
    if mode == "grid":
        steps = [row["settings"].get("steps", 400)] if own else [400, 200]
        config["params"] = {"steps": steps, **config["params"]}
        config["fixed"] = {k: v for k, v in config["fixed"].items() if k != "steps"}
        grid = config["first"] = [{"steps": s, **first} for s in steps for first in config["first"]]
        config["n_trials"] = len(grid)
    else:
        start = {k: row["settings"][k] for k in config["params"] if k in row["settings"]}
        if set(start) == set(config["params"]):  # the settings of the reference, the first trial
            config["first"] = [start, *config["first"]]
        if n_trials is not None:
            config["n_trials"] = n_trials
    made = {k: v for k, v in config.items() if k != "n_trials"}
    folder = Path(run).parts[-2:] if Path(run).name.startswith("run_") else Path(run).parts[-1:]
    name = f"all_{mode}_{method}_{'_'.join(folder)}_{short_hash(made)}"
    return Plan(method, mode, name, config, references["apd_floor"], row.get("minutes") if own else None,
                row.get("study_hours") if own else None, grid)


def announce(plan: Plan, done: int, device: str) -> str:
    """What is about to run, before it runs."""
    total = plan.config["n_trials"]
    kind = "OPTUNA STUDY" if plan.mode == "study" else "GRID"
    text = [f"{kind} of completion {plan.method}: {total} settings x {plan.config['poses']} validation poses "
            f"x 4 occlusions on {device}, NOT one setting; {done} done, {max(total - done, 0)} to run"]
    space = {k: v for k, v in plan.config["params"].items()}
    text.append(f"  study {plan.name}, searched: {space}, fixed: {plan.config['fixed']}")
    text.append("  then one full evaluation at the best point with APD >= "
                f"{plan.floor:.2f} (the rule of the references)")
    if plan.mode == "grid" and plan.minutes:
        text.append(f"  a setting took {plan.minutes:g} min in the study of the reference of this architecture (other "
                    f"hosts differ): about {plan.minutes * max(total - done, 0) / 60:.1f} h")
    elif plan.mode == "study" and plan.hours:
        text.append(f"  the study of the reference of this architecture took {plan.hours:g} h (other hosts differ)")
    else:
        text.append("  time unknown for this architecture")
    return "\n".join(text)


def missing(grid: List[dict], trials) -> List[dict]:
    """The settings of a grid that no finished or waiting trial has (a stopped or killed trial is run again)."""
    have = [t["params"] for t in trials if t["state"] in ("COMPLETE", "WAITING")]
    return [g for g in grid if not any(all(h.get(k) == v for k, v in g.items()) for h in have)]


def choose(points: List[dict], floor: float) -> Tuple[dict, str]:
    """The point (mean, apd, params, number) the rule takes and a note: the lowest mean with APD >= floor; without
    any the highest APD."""
    allowed = [p for p in points if p["apd"] >= floor]
    if not allowed:
        return max(points, key=lambda p: p["apd"]), f"no point with APD >= {floor:.2f}, the highest APD taken"
    return min(allowed, key=lambda p: p["mean"]), ""


def at_end(point: dict, grid: List[dict]) -> bool:
    """Whether the point is at the lowest or highest guidance of its part of the grid."""
    part = [g["guidance"] for g in grid if g["steps"] == point["params"]["steps"]]
    return point["params"]["guidance"] in (min(part), max(part))
