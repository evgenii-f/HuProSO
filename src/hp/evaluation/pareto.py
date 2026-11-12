"""The Pareto comparison of pose priors: sample quality against the cost of sampling.

Shared by explore/evaluation/pareto_posedflow_vs_rsgm.ipynb and scripts/evaluate_pareto.py, so that both evaluate the
same models with the same settings and read and write the same cache:

    <CACHE>/<slug of label>/<stage>.json = {"name", "stage", "settings", "result"}

with the stages
    harness                    our harness (Wasserstein, recall, precision, log-likelihood) on their test split;
    paper                      the protocol of the PoseD-Flow paper (FID, FID own, APD, dNN over 20 sets of 500);
    runtime_<chip>_<device>    the wall-clock time of model.sample(n), one file per chip and device type, with the
                               times of every n measured so far (`measure_runtime`).

A result is reused only if it was computed with the same settings. Settings are compared with the home folder as ~ and
file times in whole seconds (`portable`), so results computed under /home/<user> are found under /Users/<user> after
`rsync -a` or `tar` (both keep the file times). The data and evaluation settings are those of posedflow_vs_rsgm.ipynb:
change them together, or the cache does not match.
"""
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from hp.evaluation.evaluate import EvalConfig, evaluate
from hp.evaluation.posedflow_protocol import SmplSkeleton, apd, confidence, fid, nearest_neighbour_distance
from hp.models.io import MANIFEST, load_model

# ---- settings (part of the cache keys) ------------------------------------------------------------------------------
ROTATIONS = "~/data/AMASS_rotations_posedflow_c10"
CHECKPOINT = "~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt"
SKELETON = "~/data/body_models/smpl/smpl_neutral_skeleton.npz"
GT_STATS = "~/Documents/PoseDFlow/examples/gt_mean_var_v2.npz"
HARNESS = EvalConfig()  # 20k samples, 2000 queries, 3 repeats
N_CONTRAST = 100_000
SETS, SET_SIZE = 20, 500
CACHE = Path("~/data/experiments/posedflow_comparison/cache").expanduser()
BATCHES = (4096, 1)  # the n of model.sample(n) timed by default; more can be added to the cache later
REPEATS = 3

# The models: the specs (and names, which key the cache) of posedflow_vs_rsgm.ipynb, plus the HuProSO3 flow.
# "short": the label in the plots. posedflow=dict(...) for their checkpoint; run=<folder or list of folders (seeds)>;
# weights= (default weights_ema) and the overrides sample_steps=, likelihood_steps=, eps=, and of RSGM sampler=,
# schedule=, schedule_rho=; likelihood=False: the harness without the log-likelihood.
MODELS = {
    "PoseD-Flow: their checkpoint, midpoint 100 (paper)": dict(short="PoseD-Flow, mid 100",
        posedflow=dict(weights="ema", method="midpoint", sample_steps=100)),
    # "PoseD-Flow: their checkpoint, Euler 1000 raw (their script)": dict(short="PoseD-Flow, Euler 1000",
    #     posedflow=dict(weights="raw", method="euler", sample_steps=1000)),
    # "PoseD-Flow retrained: our AMASS, midpoint 100": dict(short="PoseD-Flow†, mid 100", run="~/data/experiments/posedflow_amass_s10"),
    "DPoser: their full, 1000 steps": dict(short="DPoser full, 1000", run="~/data/experiments/full/dposer_posedflow_s10"),
    "DPoser: their full, 100 steps": dict(short="DPoser full, 100", run="~/data/experiments/full/dposer_posedflow_s10", sample_steps=100),
    "DPoser: their c10 (RSGM budget), 1000 steps": dict(short="DPoser c10, 1000", run="~/data/experiments/dposer_posedflow_rsgm_budget_s10"),
    "DPoser: their c10 (RSGM budget), 100 steps": dict(short="DPoser c10, 100", run="~/data/experiments/dposer_posedflow_rsgm_budget_s10", sample_steps=100),
    # "RSGM 1024x3 t23: our AMASS": dict(short="RSGM 1024x3†", run="~/data/experiments/rsgm_tuned_t23"),
    "RSGM 512x4: their full": dict(short="RSGM 512x4 full", run="~/data/experiments/full/rsgm_posedflow_small_s10"),
    "RSGM 512x4: their c10 (3 seeds)": dict(short="RSGM 512x4", run=[f"~/data/experiments/rsgm_posedflow_small_s{seed}" for seed in (10, 11, 12)]),
    "RSGM 1536x6: their c10": dict(short="RSGM 1536x6", run="~/data/experiments/rsgm_posedflow_s10"),
    "RSGM 4096x7: their c10, final (30k)": dict(short="RSGM 4096x7", run="~/data/experiments/rsgm_posedflow_large_s10"),
    # "RSGM 4096x7: their c10, best (8k)": dict(short="RSGM 4096x7 @8k", run="~/data/experiments/rsgm_posedflow_large_s10/run_0001/checkpoint_008000"),
    "RSGM 512x4: their c10, 300 steps": dict(short="RSGM 512x4, 300", run="~/data/experiments/rsgm_posedflow_small_s10", sample_steps=300),
    "RSGM 512x4: their c10, 1000 steps": dict(short="RSGM 512x4, 1000", run="~/data/experiments/rsgm_posedflow_small_s10", sample_steps=1000),
    # "RSGM 512x4: their c10, eps 2e-3": dict(short="RSGM 512x4, eps 2e-3", run="~/data/experiments/rsgm_posedflow_small_s10", eps=2e-3),
    # "RSGM 512x4: their c10, eps 1e-2": dict(short="RSGM 512x4, eps 1e-2", run="~/data/experiments/rsgm_posedflow_small_s10", eps=1e-2),
    # "RSGM 512x4: their c10, raw weights": dict(short="RSGM 512x4, raw", run="~/data/experiments/rsgm_posedflow_small_s10", weights="weights"),
    # ---- the HuProSO3 flow of the paper's settings, trained on our AMASS (all frames, seed 11), final weights
    "HuProSO3 flow: our AMASS": dict(short="HuProSO3 flow†", run="~/data/experiments/huproso3_full_seed11", weights="weights"),
    # the same flow and training (20k steps of 2000, no EMA) on their c10; the final weights, as in the HuProSO3 paper
    "HuProSO3 flow: their c10": dict(short="HuProSO3 flow", run="~/data/experiments/huproso3_posedflow_s10", weights="weights"),
    # the best validation NLL (step 3k): the training NLL keeps falling, the validation NLL rises after it
    "HuProSO3 flow: their c10, best (3k)": dict(short="HuProSO3 flow @3k", run="~/data/experiments/huproso3_posedflow_s10/run_0001/checkpoint_003000", weights="weights"),
    # ---- VPoser (V02_05 network, loss and training), retrained; no EMA. The best checkpoint by their validation v2v,
    # as their release (epoch 13): after the extra loss terms end (216k steps) the latent space partly collapses.
    "VPoser: their full, best (196k)": dict(short="VPoser full", run="~/data/experiments/full/vposer_posedflow_s10/run_0001/checkpoint_196000", weights="weights"),
    "VPoser: their full, final (270k)": dict(short="VPoser full @270k", run="~/data/experiments/full/vposer_posedflow_s10", weights="weights"),
    "VPoser: their c30, best (168k)": dict(short="VPoser c30", run="~/data/experiments/vposer_posedflow_c30_s10/run_0001/checkpoint_168000", weights="weights"),
    "VPoser: our AMASS, best (168k)": dict(short="VPoser†", run="~/data/experiments/vposer_amass_s10/run_0001/checkpoint_168000", weights="weights"),
    "VPoser: our AMASS c30, best (84k)": dict(short="VPoser† c30", run="~/data/experiments/vposer_amass_c30_s10/run_0001/checkpoint_084000", weights="weights"),
}
# PoseD-Flow and DPoser with fewer steps of their own samplers, towards the fast end of the fronts. Network calls:
# midpoint 2 per step, Euler (PoseD-Flow) and Euler-Maruyama (DPoser) 1.
for _steps in (50, 25, 10):
    MODELS[f"PoseD-Flow: their checkpoint, midpoint {_steps}"] = dict(
        short=f"PoseD-Flow, mid {_steps}", posedflow=dict(weights="ema", method="midpoint", sample_steps=_steps))
for _steps in (50, 20):
    MODELS[f"PoseD-Flow: their checkpoint, Euler {_steps}"] = dict(
        short=f"PoseD-Flow, Euler {_steps}", posedflow=dict(weights="ema", method="euler", sample_steps=_steps))
for _steps in (50, 20):
    MODELS[f"DPoser: their full, {_steps} steps"] = dict(
        short=f"DPoser full, {_steps}", run="~/data/experiments/full/dposer_posedflow_s10", sample_steps=_steps)
    MODELS[f"DPoser: their c10 (RSGM budget), {_steps} steps"] = dict(
        short=f"DPoser c10, {_steps}", run="~/data/experiments/dposer_posedflow_rsgm_budget_s10", sample_steps=_steps)
# The RSGM models on the Pareto fronts with the samplers chosen in explore/evaluation/rsgm_samplers.ipynb, at 50 network
# calls: the probability flow ODE on the uniform grid (Euler) and with the EDM schedule (Heun, 2 calls per step).
RSGM_SAMPLERS = {
    "ODE Euler 50": dict(sampler="euler", sample_steps=50),
    "Heun EDM 50 NFE": dict(sampler="heun", schedule="edm", sample_steps=25),
}
RSGM_FRONT = {
    "RSGM 512x4: their c10 (3 seeds)": MODELS["RSGM 512x4: their c10 (3 seeds)"],
    "RSGM 512x4: their full": MODELS["RSGM 512x4: their full"],
    "RSGM 1536x6: their c10": MODELS["RSGM 1536x6: their c10"],
}
for _base, _spec in RSGM_FRONT.items():
    for _label, _sampler in RSGM_SAMPLERS.items():
        # no log-likelihood: it does not depend on the sampler (that of the base model) and is the slow part of the harness
        MODELS[f"{_base}, {_label}"] = {**_spec, "short": f"{_spec['short']}, {_label}", **_sampler, "likelihood": False}
# The baselines (the models other than RSGM) in the settings of their papers: their sampler and steps, their training
# data, the checkpoint they release. The others are our variants (fewer steps, other data, other checkpoints).
PUBLISHED = {
    "PoseD-Flow: their checkpoint, midpoint 100 (paper)",  # their Tab. 1
    "DPoser: their full, 1000 steps",  # their N = 1000
    "VPoser: their full, best (196k)",  # the best validation v2v, as their release
    "VPoser: our AMASS, best (168k)",  # on the data of the released V02_05 (†)
    "HuProSO3 flow: our AMASS",  # the paper's flow on the paper's data (†)
    "HuProSO3 flow: their c10",  # the final weights, as in the paper
}
assert PUBLISHED <= set(MODELS), sorted(PUBLISHED - set(MODELS))
# ---------------------------------------------------------------------------------------------------------------------


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def select(patterns: Optional[List[str]], models: Dict[str, dict] = MODELS) -> Dict[str, dict]:
    """The models whose name or short label equals a pattern, or contains it (case-insensitive); None: all."""
    if not patterns:
        return dict(models)
    chosen = {}
    for pattern in patterns:
        exact = [n for n, s in models.items() if pattern in (n, s.get("short"))]
        found = exact or [n for n, s in models.items() if pattern.lower() in f"{n}\n{s.get('short', '')}".lower()]
        if not found:
            raise KeyError(f"no model matches {pattern!r}")
        chosen.update({n: models[n] for n in found})
    return {n: models[n] for n in models if n in chosen}  # in the order of MODELS


# ---- the models -----------------------------------------------------------------------------------------------------
OVERRIDES = ("sample_steps", "likelihood_steps", "eps", "sampler", "schedule", "schedule_rho")


def run_folder(path):
    path = Path(path).expanduser()
    if (path / MANIFEST).is_file():
        return path
    runs = sorted(p for p in path.glob("run_[0-9][0-9][0-9][0-9]") if (p / "history.json").is_file() and (p / MANIFEST).is_file())
    return runs[-1] if runs else None


def tag(path):
    p = Path(path).expanduser()
    return p.parent.name if re.fullmatch(r"run_\d{4}", p.name) else p.name


def members(name, spec):
    """(label, path) of each run of a model: a model with several runs (seeds) is evaluated per run."""
    if "posedflow" in spec:
        return [(name, None)]
    paths = spec["run"] if isinstance(spec["run"], (list, tuple)) else [spec["run"]]
    return [(name, paths[0])] if len(paths) == 1 else [(f"{name} / {tag(p)}", p) for p in paths]


def resolve(spec, path, device: str):
    """(builder, identity) of one run: how to build it, and what its results depend on (for the cache)."""
    if "posedflow" in spec:
        from hp.models.posedflow import PoseDFlowPretrained
        args = dict(spec["posedflow"])
        checkpoint = Path(CHECKPOINT).expanduser()
        if not checkpoint.is_file():
            return None, None
        identity = {"kind": "posedflow", "args": args, "checkpoint": str(checkpoint), "mtime": checkpoint.stat().st_mtime}
        return (lambda: PoseDFlowPretrained(checkpoint, **args).to(device)), identity
    run = run_folder(path)
    if run is None:
        return None, None
    weights = spec.get("weights", "weights_ema")
    file = run / f"{weights}.pt"
    overrides = {k: spec[k] for k in OVERRIDES if k in spec}
    identity = {"kind": "run", "run": str(run), "weights": weights, "size": file.stat().st_size,
                "mtime": file.stat().st_mtime, "overrides": overrides}

    def build():
        model = load_model(run, device=device, weights=weights)
        for k, v in overrides.items():
            setattr(model, k, v)
        if not spec.get("likelihood", True):
            model.has_log_prob = False
        return model
    return build, identity


class Zoo:
    """The runs of some models, resolved on this machine: builders[label], identities[label], groups[name] (its
    labels), group_of[label] (its name). A run that is not on this machine is skipped, with a message."""

    def __init__(self, models: Dict[str, dict], device: str, log: Callable = print):
        self.models, self.device = models, device
        self.builders, self.identities, self.groups, self.group_of = {}, {}, {}, {}
        for name, spec in models.items():
            for label, path in members(name, spec):
                build, identity = resolve(spec, path, device)
                if build is None:
                    log(f"{label}: skipped, not on this machine ({path or CHECKPOINT})")
                    continue
                self.builders[label], self.identities[label], self.group_of[label] = build, identity, name
                self.groups.setdefault(name, []).append(label)

    def forced(self, label: str, recompute) -> bool:
        """Whether `recompute` (True, or a set of names of models or labels of runs) asks to compute `label` again."""
        return recompute is True or label in recompute or self.group_of.get(label) in recompute


# ---- the cache ------------------------------------------------------------------------------------------------------
def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def portable(value):
    """Settings as compared across machines: a home folder as ~, file times in whole seconds."""
    if isinstance(value, dict):
        return {k: (int(v) if k == "mtime" else portable(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return re.sub(r"^/(home|Users)/[^/]+(?=/|$)", "~", value) if isinstance(value, str) else value


def same_elsewhere(stage, settings, cache: Path = CACHE):
    for other in sorted(cache.glob(f"*/{stage}.json")):
        saved = json.loads(other.read_text())
        if portable(saved.get("settings")) == portable(settings):
            return saved["result"], other.parent.name
    return None, None


def cached(label, stage, settings, compute, force: bool = False, cache: Path = CACHE, log: Callable = print):
    """The result of `stage` for `label` from <cache>/<label>/<stage>.json if it was computed with these settings (or
    from another folder with the same settings); otherwise (or with `force`) computed and saved."""
    path = cache / slug(label) / f"{stage}.json"
    settings = json.loads(json.dumps(settings))
    result = None
    if not force:
        if path.is_file():
            saved = json.loads(path.read_text())
            if portable(saved.get("settings")) == portable(settings):
                return saved["result"]
            log(f"  {label} / {stage}: the cached result is of other settings")
        result, source = same_elsewhere(stage, settings, cache)
        if result is not None:
            log(f"  {label} / {stage}: the same evaluation is cached as {source}, reused")
    if result is None:
        log(f"  {label} / {stage}: computing")
        result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": label, "stage": stage, "settings": settings, "result": result}, indent=1))
    return result


# ---- quality: our harness and the protocol of PoseD-Flow -------------------------------------------------------------
def harness_settings(test, harness: EvalConfig = HARNESS, n_contrast: int = N_CONTRAST) -> dict:
    return {"config": harness.__dict__, "rotations": ROTATIONS, "test": len(test), "contrast": n_contrast}


def paper_settings(train, sets: int = SETS, set_size: int = SET_SIZE) -> dict:
    return {"sets": sets, "size": set_size, "rotations": ROTATIONS, "train": len(train), "skeleton": SKELETON,
            "gt_stats": GT_STATS}


def contrast_poses(train, n_contrast: int = N_CONTRAST):
    """The training poses that the harness compares with the test poses (the yardstick of Wasserstein)."""
    return train[torch.from_numpy(np.random.default_rng(0).choice(len(train), n_contrast, replace=False))]


class Protocol:
    """FID (their statistics and of our train subset), APD and dNN over `sets` sets of `set_size` poses.
    `skeleton` and `gt_stats`: where the files are on this machine (the settings keep SKELETON and GT_STATS)."""

    def __init__(self, train, device: str, sets: int = SETS, set_size: int = SET_SIZE,
                 skeleton: str = SKELETON, gt_stats: str = GT_STATS):
        self.train, self.device, self.sets, self.set_size = train, device, sets, set_size
        self.skeleton_path, self.gt_stats_path = skeleton, gt_stats
        self._data = None

    def data(self):
        if self._data is None:
            skeleton = SmplSkeleton(self.skeleton_path)
            gt = np.load(Path(self.gt_stats_path).expanduser())
            own = skeleton.joints(self.train).reshape(len(self.train), -1).numpy()
            self._data = (skeleton, gt["muj"].astype(np.float64), gt["covj"].astype(np.float64), own.mean(0),
                          np.cov(own, rowvar=False), self.train.to(self.device))
        return self._data

    def __call__(self, make_set: Callable[[int], torch.Tensor]) -> dict:
        skeleton, muj, covj, mu_own, cov_own, train_dev = self.data()
        out = {"FID": [], "FID own": [], "APD": [], "dNN": []}
        for k in range(self.sets):
            poses = make_set(k)
            joints = skeleton.joints(poses)
            out["FID"].append(fid(joints, muj, covj))
            out["FID own"].append(fid(joints, mu_own, cov_own))
            out["APD"].append(apd(joints))
            out["dNN"].append(nearest_neighbour_distance(poses, train_dev).mean().item())
        return {name: confidence(values) for name, values in out.items()}

    def generated(self, build) -> Callable[[], dict]:
        def compute():
            model = build().eval()
            torch.manual_seed(0)
            return self(lambda k: model.sample(self.set_size).float().cpu())
        return compute

    def real(self, poses, seed: int) -> Callable[[], dict]:
        def compute():
            rng = np.random.default_rng(seed)
            return self(lambda k: poses[torch.from_numpy(rng.choice(len(poses), self.set_size, replace=False))])
        return compute


def evaluate_quality(zoo: Zoo, test, train, recompute=frozenset(), harness: EvalConfig = HARNESS,
                     n_contrast: int = N_CONTRAST, protocol: Optional[Protocol] = None,
                     stages=("harness", "paper"), build_wrapper: Callable = lambda m: m,
                     cache: Path = CACHE, log: Callable = print) -> Tuple[dict, dict]:
    """harness[label] and paper[label] of every run of `zoo` (from the cache, or computed), and the paper metrics of
    the real train and test poses (the yardsticks). `build_wrapper` changes a built model (e.g. chunked_log_prob)."""
    protocol = protocol or Protocol(train, zoo.device)
    h_settings = harness_settings(test, harness, n_contrast)
    p_settings = paper_settings(train, protocol.sets, protocol.set_size)
    contrast = None
    harness_results, paper_results = {}, {}
    for label, build in zoo.builders.items():
        force = zoo.forced(label, recompute)
        wrapped = (lambda build=build: build_wrapper(build()))
        if "harness" in stages:
            if contrast is None:
                contrast = contrast_poses(train, n_contrast)
            harness_results[label] = cached(
                label, "harness", {"model": zoo.identities[label], **h_settings},
                lambda wrapped=wrapped: evaluate(wrapped(), test, harness, contrast=contrast, progress=True),
                force=force, cache=cache, log=log)
        if "paper" in stages:
            paper_results[label] = cached(label, "paper", {"model": zoo.identities[label], **p_settings},
                                          protocol.generated(wrapped), force=force, cache=cache, log=log)
    if "paper" in stages:
        paper_results["real train poses"] = cached("real train poses", "paper", p_settings, protocol.real(train, 2),
                                            cache=cache, log=log)
        paper_results["real test poses"] = cached("real test poses", "paper", {"test": len(test), **p_settings},
                                                  protocol.real(test, 1), cache=cache, log=log)
    return harness_results, paper_results


def chunked_log_prob(model, chunk: int):
    """model.log_prob in pieces of `chunk` poses: the exact divergence of a wide network needs memory per pose.
    The result is the same; the harness's sample_batch (part of the settings) stays as it is."""
    full = model.log_prob

    def log_prob(pose, cond=None):
        return torch.cat([full(pose[i:i + chunk], cond) for i in range(0, len(pose), chunk)])
    model.log_prob = log_prob
    return model


# ---- cost: the time of model.sample(n) ------------------------------------------------------------------------------
def device_type(device: str) -> str:
    return torch.device(device).type


def sync(device: str):
    if device_type(device) == "mps":
        torch.mps.synchronize()
    elif device_type(device) == "cuda":
        torch.cuda.synchronize(device)


def chip_name(device: str) -> str:
    """The GPU on CUDA, otherwise the CPU (on Apple silicon the same chip as the GPU of mps)."""
    if device_type(device) == "cuda":
        return torch.cuda.get_device_name(torch.device(device))
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    found = re.search(r"model name\s*:\s*(.+)", cpuinfo.read_text()) if cpuinfo.is_file() else None
    return found.group(1).strip() if found else platform.processor() or platform.machine()


def runtime_stage(device: str) -> str:
    """One file per chip and device type: another machine does not overwrite the times of this one."""
    return f"runtime_{slug(chip_name(device))}_{device_type(device)}"


def runtime_settings(device: str) -> dict:
    """What the times of one file depend on, besides the model. The n and the repeats are per entry of the file."""
    return {"chip": chip_name(device), "device": device_type(device), "platform": platform.platform(),
            "torch": torch.__version__}


def _conditions(device: str) -> dict:
    """What else ran on the machine while timing (not part of the settings: for reading the times later)."""
    out = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "load_average": list(os.getloadavg())}
    if device_type(device) == "cuda":
        try:
            index = torch.device(device).index or 0
            query = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            gpu = query.stdout.strip()
            apps = subprocess.run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.split()
            out["gpu"] = {"utilization_memory_mib": gpu, "processes": len(apps)}
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def time_sampling(model, n: int, device: str, repeats: int = REPEATS) -> dict:
    """The median wall-clock time of model.sample(n) over `repeats` runs after a warm-up, the device synchronized."""
    torch.manual_seed(0)
    with torch.no_grad():
        model.sample(n)  # warm-up
        sync(device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.sample(n)
            sync(device)
            times.append(time.perf_counter() - start)
    seconds = float(np.median(times))
    return {"seconds": seconds, "us_per_pose": seconds / n * 1e6, "all_seconds": times, "repeats": repeats,
            "conditions": _conditions(device)}


def upgrade_runtime(saved: dict) -> dict:
    """A runtime file of the first format (the n and the repeats in the settings, one set of conditions) in the
    current one (the n as keys of result["batches"], the repeats and conditions in each entry). The times stay."""
    settings = dict(saved["settings"])
    if "batches" not in settings:
        return saved
    settings.pop("batches")
    repeats = settings.pop("repeats")
    result = dict(saved["result"])
    conditions = result.pop("conditions", None)
    result["batches"] = {n: {**entry, "repeats": repeats, **({"conditions": conditions} if conditions else {})}
                         for n, entry in result["batches"].items()}
    return {**saved, "settings": settings, "result": result}


def _set_aside(path: Path, saved: dict, settings: dict, log: Callable) -> Path:
    """Rename a runtime file of other settings, so its times are kept: <stage>.torch_<version>.json if torch changed,
    else <stage>.<time of the file>.json."""
    old = saved["settings"]
    if old.get("torch") != settings.get("torch"):
        suffix = slug(f"torch {old.get('torch')}")
    else:
        suffix = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    target, k = path.with_name(f"{path.stem}.{suffix}.json"), 2
    while target.exists():
        target, k = path.with_name(f"{path.stem}.{suffix}_{k}.json"), k + 1
    path.rename(target)
    changed = sorted(k for k in set(old) | set(settings) if portable(old.get(k)) != portable(settings.get(k)))
    log(f"  {path.parent.name} / {path.stem}: of other settings ({', '.join(changed)}), set aside as {target.name}")
    return target


def cached_runtime(label, stage, settings, build, batches=BATCHES, repeats: int = REPEATS, force: bool = False,
                   device: str = "cpu", cache: Path = CACHE, log: Callable = print) -> dict:
    """The times of model.sample(n) for every n of `batches` (and those measured before), from
    <cache>/<label>/<stage>.json: an n not there yet, or timed with other repeats, is measured and added; with `force`
    all of `batches` are measured again. A file of other settings (another torch, other weights) is set aside, not
    overwritten; a file of another label with the same settings is used as the start."""
    path = cache / slug(label) / f"{stage}.json"
    settings = json.loads(json.dumps(settings))
    result, upgraded = None, False
    if path.is_file():
        raw = json.loads(path.read_text())
        saved = upgrade_runtime(raw)
        if portable(saved["settings"]) == portable(settings):
            result, upgraded = saved["result"], saved is not raw
        else:
            _set_aside(path, saved, settings, log)
    if result is None:
        for other in sorted(cache.glob(f"*/{stage}.json")):
            saved = upgrade_runtime(json.loads(other.read_text()))
            if portable(saved["settings"]) == portable(settings):
                result, upgraded = saved["result"], True
                log(f"  {label} / {stage}: the same model is timed as {other.parent.name}, reused")
                break
    result = result or {"batches": {}}
    todo = [n for n in batches if force or str(n) not in result["batches"]
            or result["batches"][str(n)].get("repeats") != repeats]
    if todo:
        log(f"  {label} / {stage}: timing n = {', '.join(map(str, todo))}")
        model = build().eval()
        result["parameters"] = model.num_parameters()
        result["sample_steps"] = getattr(model, "sample_steps", None)
        result["sample_nfe"] = getattr(model, "sample_nfe", None)
        for n in todo:
            entry = time_sampling(model, n, device, repeats)
            result["batches"][str(n)] = entry
            log(f"    n={n}: {entry['seconds']:.3f} s, {entry['us_per_pose']:,.1f} us/pose")
        del model
        if device_type(device) == "mps":
            torch.mps.empty_cache()
        elif device_type(device) == "cuda":
            torch.cuda.empty_cache()
    result["batches"] = dict(sorted(result["batches"].items(), key=lambda item: -int(item[0])))
    if todo or upgraded or not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": label, "stage": stage, "settings": settings, "result": result}, indent=1))
    return result


def measure_runtime(zoo: Zoo, recompute=frozenset(), batches=BATCHES, repeats: int = REPEATS,
                    cache: Path = CACHE, log: Callable = print) -> dict:
    """runtime[name] of every model of `zoo`, with result["batches"][str(n)] for every n measured so far: one run per
    model is timed (its seeds have the same network and sampler)."""
    stage, settings = runtime_stage(zoo.device), runtime_settings(zoo.device)
    runtime = {}
    for name, labels in zoo.groups.items():
        label = labels[0]
        runtime[name] = cached_runtime(label, stage, {"model": zoo.identities[label], **settings}, zoo.builders[label],
                                       batches, repeats, zoo.forced(label, recompute), zoo.device, cache, log)
    return runtime


def load_runtime(zoo: Zoo, stage: str, cache: Path = CACHE) -> Tuple[dict, dict]:
    """The times of another machine (or this one) from the cache, without timing: runtime[name] of the models of
    `zoo` whose <stage>.json is of the same weights, and what the times were measured on and how: chip, device,
    platform, torch, repeats and the dates of measuring (several values if the files differ)."""
    runtime, settings, entries = {}, [], []
    for name, labels in zoo.groups.items():
        path = cache / slug(labels[0]) / f"{stage}.json"
        if not path.is_file():
            continue
        saved = upgrade_runtime(json.loads(path.read_text()))
        if portable(saved["settings"].get("model")) != portable(zoo.identities[labels[0]]):
            continue
        runtime[name] = saved["result"]
        settings.append(saved["settings"])
        entries += saved["result"]["batches"].values()

    def values(get):
        return sorted({str(v) for v in map(get, settings + entries) if v is not None})
    dates = values(lambda s: s.get("conditions", {}).get("date"))
    info = {key: values(lambda s, key=key: s.get(key)) for key in ("chip", "device", "platform", "torch")}
    info.update(repeats=values(lambda s: s.get("repeats")), dates=[d[:10] for d in dates[:1] + dates[-1:]])
    return runtime, info


def runtime_stages(zoo: Zoo, cache: Path = CACHE) -> Dict[str, Tuple[dict, dict]]:
    """Every runtime measurement in the cache of the models of `zoo`: {stage: load_runtime(zoo, stage)}, the stages
    with the most models first. Files set aside (<stage>.<suffix>.json) are not among them."""
    stages = {p.stem for labels in zoo.groups.values() for p in (cache / slug(labels[0])).glob("runtime_*.json")
              if "." not in p.stem}
    found = {stage: load_runtime(zoo, stage, cache) for stage in sorted(stages)}
    return dict(sorted(((s, v) for s, v in found.items() if v[0]), key=lambda item: -len(item[1][0])))
