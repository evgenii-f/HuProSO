"""The command line shared by the test protocols of scripts/evaluate.py: how a model is named (a run folder or
checkpoint, a row of a registry, a point of a front), which poses (--split, --poses), where the results go (--out),
and the settings of a method (--set key=value, --settings JSON).

A protocol (completion; later generation, runtime) adds its own options to these and gets {label: spec} from
models(), the specs of hp.evaluation.completion.build: {"run": <folder>} or {"posedflow": <.ckpt>}, with optional
"weights", "method", "prior" and "dflow".
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

SPLITS = ("valid", "test")


class UsageError(ValueError):
    """A wrong combination of flags or an unknown model: scripts/evaluate.py reports it as a usage error."""

DFLOW_KEYS = ("lr", "running_weight", "steps")  # D-Flow: arguments of completion.run(); the other keys go to the prior


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("models (any mix, at least one)")
    group.add_argument("--run", nargs="+", default=[], metavar="[LABEL=]PATH",
                       help="a run or checkpoint folder (model.json and weights), or a PoseD-Flow .ckpt; the label "
                            "defaults to the path below ~/data/experiments, the method and a hash of the settings")
    group.add_argument("--weights", default=None, help="with --run: weights_ema (default) or weights")
    group.add_argument("--models", nargs="+", default=[], metavar="LABEL",
                       help="rows of the registry (all: every row)")
    group.add_argument("--points", default=None,
                       help="a YAML file of scripts/select_test_points.py: its points (all, or --fronts)")
    group.add_argument("--fronts", nargs="+", default=None, help='with --points: only these fronts, e.g. "RSGM dflow"')
    group.add_argument("--method", default=None, help="with --run: the method of the protocol (default its own)")
    group.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", dest="assignments",
                       help="with --run: settings of the method, values read as YAML (0.05, true, [1, 2], edm)")
    group.add_argument("--settings", default=None, metavar="JSON",
                       help="with --run: settings of the method as a JSON object; --set overrides its keys")


def add_common_arguments(parser: argparse.ArgumentParser, out: str, quick: Optional[int] = None) -> None:
    """quick: the number of validation poses of --quick for this protocol (no --quick without it)."""
    group = parser.add_argument_group("common")
    group.add_argument("--split", choices=SPLITS + ("all",), default=None,
                       help="the poses: valid (the fixed validation draw), test (default), or all (valid, then test)")
    group.add_argument("--poses", type=int, default=None, help="the first n poses of the split (a quick check, "
                                                               "kept in a folder of its own)")
    if quick is not None:
        group.add_argument("--quick", action="store_const", const=quick, default=None,
                           help=f"a quick check on the validation: --split valid --poses {quick} (--poses overrides "
                                f"the number); the first poses of the validation draw, so a subset of the full one")
    group.add_argument("--out", default=out, help=f"the results root (default {out}); see split_dir")
    group.add_argument("--device", default="cuda:0", help="cuda:N | mps | cpu")
    group.add_argument("--rotations", default=None, help="the data folder (default: that of the protocol)")
    group.add_argument("--recompute", action="store_true",
                       help="replace results made with other settings (otherwise they are reported and kept)")
    group.add_argument("--dry-run", action="store_true",
                       help="print what would run and what is already there, run nothing")


def resolve_poses(args: argparse.Namespace) -> None:
    """--split and --poses after --quick: --quick means the validation and its number of poses (unless --poses);
    without it the split defaults to the test. Raises UsageError on --quick with another split."""
    if getattr(args, "quick", None) is None:
        args.split = args.split or "test"
        return
    if args.split not in (None, "valid"):
        raise UsageError(f"--quick is a check on the validation, not on --split {args.split}")
    args.split = "valid"
    args.poses = args.poses if args.poses is not None else args.quick


def splits(split: str) -> List[str]:
    return list(SPLITS) if split == "all" else [split]


def split_dir(out, split: str, poses: Optional[int] = None) -> Path:
    """Where the results of a split go: the test in <out> itself (as the caches made before splits existed), the
    validation in <out>/valid; a quick check of the first n poses in <out>/<split>_first<n>."""
    out = Path(out).expanduser()
    if poses is not None:
        return out / f"{split}_first{poses}"
    return out if split == "test" else out / split


def parse_value(text: str):
    return yaml.safe_load(text) if text != "" else ""


def method_settings(settings: Optional[str] = None, assignments: Iterable[str] = ()) -> dict:
    """The settings of --settings (a JSON object) updated by --set key=value."""
    values = {}
    if settings:
        values = json.loads(settings)
        if not isinstance(values, dict):
            raise UsageError(f"--settings must be a JSON object, not {settings!r}")
    for item in assignments:
        key, sep, text = item.partition("=")
        if not sep or not key:
            raise UsageError(f"--set takes key=value, not {item!r}")
        values[key] = parse_value(text)
    return values


def run_spec(path: str, weights: Optional[str] = None, method: Optional[str] = None,
             settings: Optional[dict] = None, default_method: str = "dflow") -> dict:
    """The spec of a run folder or a PoseD-Flow .ckpt with a method and its settings, as the rows of the registry
    hold them: for D-Flow lr, running_weight and steps under "dflow" and the rest under "prior"; for the other methods
    all under "prior"."""
    spec = {"posedflow": path} if str(path).endswith(".ckpt") else {"run": path}
    if weights and "run" in spec:
        spec["weights"] = weights
    method = method or default_method
    settings = dict(settings or {})
    if method == "dflow":
        dflow = {key: settings.pop(key) for key in DFLOW_KEYS if key in settings}
        if dflow:
            spec["dflow"] = dflow
    else:
        spec["method"] = method
    if settings:
        spec["prior"] = settings
    return spec


def default_label(path: str, spec: dict) -> str:
    """The path below ~/data/experiments (or the file name of a .ckpt), the method unless D-Flow, and a hash of the
    settings when there are any: other settings never share a label (and so a cache entry)."""
    p = Path(path).expanduser()
    root = Path("~/data/experiments").expanduser()
    name = str(p.relative_to(root)) if p.is_relative_to(root) else (p.name if p.suffix == ".ckpt" else str(path))
    parts = [name]
    if spec.get("method"):
        parts.append(spec["method"])
    settings = {key: spec[key] for key in ("weights", "prior", "dflow") if key in spec}
    if settings:
        parts.append(hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8])
    return ", ".join(parts)


def models(args: argparse.Namespace, registry: Dict[str, dict], default_method: str = "dflow") -> Dict[str, dict]:
    """{label: spec} of --run, --models and --points, in that order; labels of points among --models restrict the
    points to them. Raises UsageError on an unknown label, on --set / --settings / --method / --weights without --run,
    and on one label with two specs."""
    out: Dict[str, dict] = {}

    def add(label: str, spec: dict, where: str) -> None:
        if label in out and out[label] != spec:
            raise UsageError(f"the label {label!r} of {where} has another spec than before")
        if label in registry and registry[label] != spec:
            raise UsageError(f"the label {label!r} of {where} is a row of the registry with another spec")
        out[label] = spec

    settings = method_settings(args.settings, args.assignments)
    if not args.run and (settings or args.method or args.weights):
        raise UsageError("--set, --settings, --method and --weights apply to --run models")
    for item in args.run:
        label, sep, path = item.rpartition("=")
        path = path if sep else item
        spec = run_spec(path, args.weights, args.method, settings, default_method)
        add(label if sep else default_label(path, spec), spec, "--run")
    points = {}
    if args.points:
        fronts = yaml.safe_load(Path(args.points).read_text())
        missing = set(args.fronts or []) - set(fronts)
        if missing:
            raise UsageError(f"no fronts {sorted(missing)} in {args.points}; there are {list(fronts)}")
        points = {p["label"]: p["spec"] for name, ps in fronts.items() if not args.fronts or name in args.fronts
                  for p in ps}
    elif args.fronts:
        raise UsageError("--fronts needs --points")
    names = list(registry) if args.models == ["all"] else args.models
    unknown = [name for name in names if name not in registry and name not in points]
    if unknown:
        raise UsageError(f"not in the registry{' nor in ' + args.points if args.points else ''}: {unknown}")
    named_points = [name for name in names if name in points]
    for name in names:
        if name in registry:
            add(name, registry[name], "--models")
    for label, spec in points.items():  # --models naming points: only those
        if not named_points or label in named_points:
            add(label, spec, args.points)
    if not out:
        raise UsageError("no models: give --run, --models or --points")
    return out


def recorded(points: Optional[str]) -> Dict[str, dict]:
    """{label: the validation numbers recorded with the point} of a --points file (empty without one)."""
    if not points:
        return {}
    fronts = yaml.safe_load(Path(points).read_text())
    return {p["label"]: p["validation"] for ps in fronts.values() for p in ps if "validation" in p}
