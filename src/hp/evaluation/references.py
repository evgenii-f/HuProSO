"""The references of PoseD-Flow's protocol (config/references) and the comparison of a model with them: what
scripts/evaluate.py all and references use.

posedflow.yaml holds the fixed reference rows: per completion method and architecture (layers and width) the settings
chosen on the validation and the numbers recorded there, the front of the studies and the targets (DPoser,
PoseD-Flow). The numbers of the reference networks on the validation, all three protocols, are in
posedflow_valid.jsonl next to it (written by `evaluate.py references`), their numbers per pose (completion) and per
sequence (denoising) in ~/data, for paired differences and for checks on the first n poses.

An entry: {"protocol": "generation" | "completion" | "denoising", "group": the method of the completion or "",
"network", "architecture", "label", "spec", "n": poses, "numbers": {column: value}, "arrays": {column: [n]}}.
"""
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "references" / "posedflow.yaml"
NUMBERS = ROOT / "config" / "references" / "posedflow_valid.jsonl"
ARRAYS = "~/data/experiments/references/posedflow_valid.pt"
METHODS = ("inpaint", "dflow")  # of the completion, the procedures with references
PROTOCOLS = ("generation", "completion", "denoising")
FULL = {"completion": 1000, "denoising": 100}  # poses / sequences of the validation draws (the studies)

# columns of the tables: (key, title, decimals, direction: -1 lower is better, 1 higher, 0 neither)
COLUMNS = {
    "generation": (("FID_val", "FID_val", 4, -1), ("FID_own", "FID own", 4, -1), ("APD", "APD", 2, 0),
                   ("dNN", "dNN", 4, 0), ("wasserstein", "W", 2, -1), ("recall", "recall", 2, -1),
                   ("precision", "precision", 2, -1), ("log_likelihood", "log-lik.", 2, 1)),
    "completion": (("mean", "mean", 2, -1), ("best", "best of 10", 2, -1), ("apd", "APD", 2, 0),
                   ("to_front", "to the front", 2, -1)),
    "denoising": (("mpjpe_40", "MPJPE 40 mm", 2, -1), ("mpvpe_40", "MPVPE 40 mm", 2, -1),
                  ("mpjpe_100", "MPJPE 100 mm", 2, -1), ("mpvpe_100", "MPVPE 100 mm", 2, -1)),
}
# generation: the spread of the same model between runs (seeds, devices: 6 runs of fact_RDW, 2026-11-23) is larger than
# the std within a run, so a difference is noise up to this much at least
SPREAD = {"wasserstein": 0.1, "recall": 0.15, "precision": 0.25}


def load(path=CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    for row in config["sets"]:
        row["key"] = f"{row['method']} {row['architecture']}"
    return config


def resolve_run(path: str) -> str:
    """The folder of a run with model.json: `path` itself or its only run below; with ~ for the home folder, so that
    the caches are the same on every machine and however the path was typed."""
    folder = Path(path).expanduser()
    folder = folder if folder.is_absolute() else Path.cwd() / folder
    if not (folder / "model.json").is_file():
        runs = sorted(p.parent for p in folder.glob("run_*/model.json"))
        if len(runs) != 1:
            raise ValueError(f"{path}: {len(runs)} runs with a model.json" + (
                f" ({', '.join(p.name for p in runs)}), name one" if runs else ""))
        folder = runs[0]
    home = Path.home()
    return f"~/{folder.relative_to(home)}" if folder.is_relative_to(home) else str(folder)


def resolve_or_none(path: str) -> Optional[str]:
    try:
        return resolve_run(path)
    except ValueError:  # not on this machine
        return None


def architecture(build_args: dict) -> str:
    """The layers and width of an RSGM: "timeres 512x2" or "MLP 512x4"."""
    if build_args.get("name") != "rsgm":
        raise ValueError(f"the references are those of RSGM, not of {build_args.get('name')!r}")
    if build_args.get("net", "mlp") == "timeres":
        return f"timeres {build_args['net_width']}x{build_args['net_blocks']}"
    if build_args.get("net", "mlp") != "mlp":
        return build_args["net"]  # another network: no reference of its own
    hidden = build_args["hidden"]
    return f"MLP {hidden[0]}x{len(hidden)}" if len(set(hidden)) == 1 else f"MLP {'-'.join(map(str, hidden))}"


def architecture_of(run: str) -> str:
    return architecture(json.loads((Path(run).expanduser() / "model.json").read_text())["build_args"])


def set_for(config: dict, method: str, arch: str) -> Tuple[dict, bool]:
    """(the reference row whose settings a model of this architecture gets, whether it is of this architecture): the
    row of `unknown_architecture` if the architecture has none with this method."""
    rows = {row["architecture"]: row for row in config["sets"] if row["method"] == method}
    return (rows[arch], True) if arch in rows else (rows[config["unknown_architecture"]], False)


def front_mean(front: Iterable[dict], apd: float) -> Optional[float]:
    """The mean MPVPE of the front at this APD, interpolated; below its lowest APD its lowest mean, above its highest
    APD None."""
    points = sorted((p["apd"], p["mean"]) for p in front)
    if not points or apd > points[-1][0]:
        return None
    return float(np.interp(apd, [a for a, _ in points], [m for _, m in points]))


# ---- entries --------------------------------------------------------------------------------------------------------
def completion_entry(parts: Dict[str, dict], **fields) -> dict:
    """The entry of a model from the per-pose metrics {occlusion: metrics()} of all the occlusions: means over the
    occlusions, per pose."""
    arrays = {"mean": np.mean([m["mpvpe"].mean(1) for m in parts.values()], axis=0),
              "best": np.mean([m["mpvpe"].min(1) for m in parts.values()], axis=0),
              "apd": np.mean([m["apd"] for m in parts.values()], axis=0)}
    return _of_arrays(arrays, protocol="completion", **fields)


def denoising_entry(parts: Dict[float, dict], **fields) -> dict:
    """The entry of a model from the per-frame metrics {noise level (m): metrics()}, per sequence."""
    arrays = {f"{key}_{round(sigma * 1000)}": np.nanmean(m[key], axis=1) for sigma, m in parts.items()
              for key in ("mpjpe", "mpvpe")}
    return _of_arrays(arrays, protocol="denoising", group="", **fields)


def generation_entry(numbers: dict, n: int, **fields) -> dict:
    return dict(protocol="generation", group="", n=n, numbers=numbers, arrays={}, **fields)


def _of_arrays(arrays: Dict[str, np.ndarray], **fields) -> dict:
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    return dict(n=len(next(iter(arrays.values()))), arrays=arrays, numbers=_means(arrays), **fields)


def _means(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {key: float(np.nanmean(value)) for key, value in arrays.items()}


def entry_key(entry: dict) -> str:
    return "|".join((entry["protocol"], entry["group"], entry["network"]))


def save(entries: List[dict], numbers=NUMBERS, arrays=ARRAYS) -> None:
    """The entries of the references, replacing those with the same key: the numbers in `numbers`, per pose in
    `arrays`."""
    arrays = Path(arrays).expanduser()
    kept = {}
    for entry in read(numbers, arrays) + list(entries):
        kept[entry_key(entry)] = entry
    Path(numbers).parent.mkdir(parents=True, exist_ok=True)
    arrays.parent.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in entry.items() if k != "arrays"} for entry in kept.values()]
    Path(numbers).write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary = arrays.with_suffix(".tmp")
    torch.save({key: {name: torch.as_tensor(value, dtype=torch.float32) for name, value in entry["arrays"].items()}
                for key, entry in kept.items() if entry["arrays"]}, temporary)
    temporary.replace(arrays)


def read(numbers=NUMBERS, arrays=ARRAYS) -> List[dict]:
    """The entries written by save(); without the arrays on this machine, or with arrays that are not those of the
    numbers, they have none."""
    numbers, arrays = Path(numbers), Path(arrays).expanduser()
    if not numbers.is_file():
        return []
    per_pose = torch.load(arrays, weights_only=True) if arrays.is_file() else {}
    entries = []
    for line in numbers.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        own = {k: v.double().numpy() for k, v in per_pose.get(entry_key(entry), {}).items()}
        same = own and all(math.isclose(float(np.nanmean(own.get(k, np.nan))), v, rel_tol=1e-5)
                           for k, v in entry["numbers"].items())
        entries.append({**entry, "arrays": own if same else {}})
    return entries


def recorded_entries(config: dict, n: int = FULL["completion"]) -> List[dict]:
    """The completion numbers recorded with the sets, as entries without arrays: what there is to compare with before
    `evaluate.py references` has run."""
    return [dict(protocol="completion", group=row["method"], network=row["network"], architecture=row["architecture"],
                 label=row["source"], spec=None, n=n, numbers=dict(row["validation"]), arrays={})
            for row in config["sets"]]


def first(entry: dict, n: int) -> Optional[dict]:
    """The entry on the first n poses; None if it has no numbers for them."""
    if entry["n"] == n:
        return entry
    if entry["protocol"] == "generation" or not entry["arrays"] or entry["n"] < n:
        return None
    arrays = {key: value[:n] for key, value in entry["arrays"].items()}
    return {**entry, "n": n, "arrays": arrays, "numbers": _means(arrays)}


# ---- the comparison -------------------------------------------------------------------------------------------------
def tolerance(model: dict, other: dict, key: str) -> Optional[float]:
    """Up to what difference of two entries a column is noise (of the poses and the draws, not of the training):
    two standard errors of the paired difference; for the generation the spread between runs and the intervals of the
    two. None if unknown."""
    if key in model["arrays"] and key in other["arrays"]:
        d = model["arrays"][key] - other["arrays"][key]
        d = d[~np.isnan(d)]
        return 2 * float(d.std(ddof=1)) / math.sqrt(len(d)) if len(d) > 1 else None
    if model["protocol"] != "generation":
        return None
    ours, theirs = model["numbers"], other["numbers"]
    for suffix, factor in (("_ci", 1.0), ("_std", 2.0)):
        if key + suffix in ours and key + suffix in theirs:
            return max(SPREAD.get(key, 0.0), factor * math.hypot(ours[key + suffix], theirs[key + suffix]))
    return None


def difference(model: dict, other: dict, key: str, digits: int) -> str:
    """"-0.40" (the model minus the other), with "~" when it is within the tolerance, "?" when that is unknown."""
    if model["numbers"].get(key) is None or other["numbers"].get(key) is None:
        return ""
    delta = model["numbers"][key] - other["numbers"][key]
    noise = tolerance(model, other, key)
    return f"{delta:+.{digits}f}" + ("?" if noise is None else "~" if abs(delta) <= noise else "")


def with_front(entry: dict, config: dict) -> dict:
    """A completion entry with its distance to the front at its APD (mm; < 0: beyond the front), on the validation
    poses of the front only."""
    if entry["protocol"] != "completion" or entry["n"] != FULL["completion"]:
        return entry
    mean = front_mean(config["front"], entry["numbers"]["apd"])
    value = None if mean is None else entry["numbers"]["mean"] - mean
    return {**entry, "numbers": {**entry["numbers"], "to_front": value}}


def compare(model: dict, references: List[dict], config: dict, targets: bool = True) -> str:
    """The table of a model against the references of its protocol and group: the model first, then the references
    with the differences (the model minus the reference), the rank of the model per column, the targets last."""
    columns = list(COLUMNS[model["protocol"]])
    same = [first(r, model["n"]) for r in references
            if r["protocol"] == model["protocol"] and r["group"] == model["group"]]
    same = [with_front(r, config) for r in same if r is not None]
    model = with_front(model, config)
    columns = [c for c in columns if model["numbers"].get(c[0]) is not None]
    lines = ["| model | " + " | ".join(title for _, title, _, _ in columns) + " |", "|" + "---|" * (len(columns) + 1)]
    value = lambda entry, key, digits: (  # noqa: E731
        "-" if entry.get(key) is None else f"{entry[key]:.{digits}f}")
    lines.append("| " + " | ".join([f"**{model['label']}**", *(f"**{value(model['numbers'], k, d)}**"
                                                                for k, _, d, _ in columns)]) + " |")
    for r in same:
        own = " (same architecture)" if r["architecture"] == model["architecture"] else ""
        cells = []
        for key, _, digits, _ in columns:
            delta = "" if key == "to_front" else difference(model, r, key, digits)
            cells.append(value(r["numbers"], key, digits) + (f" ({delta})" if delta else ""))
        lines.append("| " + " | ".join([f"{r['architecture']}: {r['network']}{own}", *cells]) + " |")
    if same:
        ranks = []
        for key, _, _, direction in columns:
            others = [r["numbers"][key] for r in same if r["numbers"].get(key) is not None]
            better = sum(direction * (o - model["numbers"][key]) > 0 for o in others)
            ranks.append(f"{better + 1} of {len(others) + 1}" if direction and others else "")
        lines.append("| rank of the model | " + " | ".join(ranks) + " |")
    if targets and model["n"] == FULL.get(model["protocol"]):  # the poses of the targets
        for t in config["targets"].get(model["protocol"], []):
            lines.append("| " + " | ".join([f"target: {t['label']}", *(value(t, k, d) for k, _, d, _ in columns)]) + " |")
    if not same and references:
        lines.append(f"\nno references for n = {model['n']} (run: scripts/evaluate.py references)")
    return "\n".join(lines)
