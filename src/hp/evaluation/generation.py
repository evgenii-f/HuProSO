"""Unconditional generation in the protocol of PoseD-Flow, on the validation or the test poses of their split (the
third protocol of scripts/evaluate.py).

The test is that of the Pareto comparison (hp.evaluation.pareto: stages "harness" and "paper", the same settings and
the same cache), so a row computed there is found here and the other way round. The validation has its own stages, in
<cache>/valid:

    generation   FID_val (the joint statistics of the validation poses in place of theirs), FID own (the statistics of
                 the training poses), APD and dNN over 20 sets of 500 samples
    harness      Wasserstein, recall and precision (deg per joint) against the validation poses, no log-likelihood
    likelihood   the log-likelihood of validation poses, if the model has one (not in the quick check)

Their FID, FID own, APD and dNN of the test do not look at the test poses at all (their statistics and the training
poses do not depend on the split), so a model chosen by them would be chosen on the numbers of the test. That is why
the validation has FID_val and does not compute their FID.
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from hp.evaluation import pareto as P
from hp.evaluation.data import load_reference
from hp.evaluation.evaluate import EvalConfig, evaluate, log_likelihood
from hp.evaluation.posedflow_protocol import SmplSkeleton

VALID_HARNESS = EvalConfig(n_likelihood=0)  # the sizes of the test; the likelihood is a stage of its own
VALID_LIKELIHOOD = EvalConfig(n_likelihood=1000)  # poses per repeat, 3 repeats
LL_CHUNK = 250  # poses per log_prob call (memory only)

STAGES = {"valid": ("generation", "harness", "likelihood"), "test": ("paper", "harness")}
# columns of the table and keys of a row: (key, title, decimals)
COLUMNS = {
    "valid": (("FID_val", "FID_val", 4), ("FID_own", "FID own", 4), ("APD", "APD", 2), ("dNN", "dNN", 4),
              ("wasserstein", "W", 2), ("recall", "recall", 2), ("precision", "precision", 2),
              ("log_likelihood", "log-lik.", 2)),
    "test": (("FID", "FID", 4), ("FID_own", "FID own", 4), ("APD", "APD", 2), ("dNN", "dNN", 4),
             ("wasserstein", "W", 2), ("recall", "recall", 2), ("precision", "precision", 2),
             ("log_likelihood", "log-lik.", 2)),
}


class OtherSettings(ValueError):
    """A cache entry made with other settings, which run_stage was told not to replace."""


def zoo_spec(spec: dict) -> dict:
    """A spec of scripts/evaluate.py ({"run", "weights", "settings"}) as a spec of hp.evaluation.pareto.MODELS (the
    settings next to the run); a spec of MODELS is returned as it is."""
    out = {key: value for key, value in spec.items() if key not in ("settings", "method")}
    return {**out, **spec.get("settings", {})}


class ValidProtocol(P.Protocol):
    """The protocol of PoseD-Flow with the joint statistics of the validation poses in place of theirs."""

    def __init__(self, valid, train, device: str, **kwargs):
        super().__init__(train, device, **kwargs)
        self.valid = valid

    def reference(self, skeleton: SmplSkeleton) -> Tuple[np.ndarray, np.ndarray]:
        joints = skeleton.joints(self.valid).reshape(len(self.valid), -1).numpy()
        return joints.mean(0), np.cov(joints, rowvar=False)

    def __call__(self, make_set: Callable) -> dict:
        out = super().__call__(make_set)
        return {"FID_val": out.pop("FID"), **out}


class Arena:
    """The poses of a split, the settings of its stages and how to compute them for one model."""

    def __init__(self, split: str, device: str, skeleton: str = P.SKELETON, gt_stats: str = P.GT_STATS,
                 load: Callable = load_reference):
        self.split, self.device = split, device
        self.poses = load(split, P.ROTATIONS, "posedflow")
        self.train = load("train", P.ROTATIONS, "posedflow")
        if split == "test":
            self.protocol = P.Protocol(self.train, device, skeleton=skeleton, gt_stats=gt_stats)
        else:
            self.protocol = ValidProtocol(self.poses, self.train, device, skeleton=skeleton)
        self._contrast = None

    def contrast(self):
        if self._contrast is None:
            self._contrast = P.contrast_poses(self.train)
        return self._contrast

    def settings(self, stage: str) -> dict:
        """What a result of the stage depends on, besides the model."""
        if self.split == "test":
            return P.harness_settings(self.poses) if stage == "harness" else P.paper_settings(self.train)
        data = {"rotations": P.ROTATIONS, "valid": len(self.poses)}
        if stage == "generation":
            return {"sets": P.SETS, "size": P.SET_SIZE, "train": len(self.train), "skeleton": P.SKELETON, **data}
        if stage == "harness":
            return {"config": asdict(VALID_HARNESS), "contrast": P.N_CONTRAST, **data}
        return {"config": asdict(VALID_LIKELIHOOD), **data}

    def compute(self, stage: str, build: Callable) -> Optional[dict]:
        model = P.chunked_log_prob(build().eval(), LL_CHUNK)
        if stage == "likelihood":
            return log_likelihood(model, self.poses, VALID_LIKELIHOOD, progress=True)
        if stage == "harness":
            config = P.HARNESS
            if self.split == "valid":
                config, model.has_log_prob = VALID_HARNESS, False  # this instance only
            return evaluate(model, self.poses, config, contrast=self.contrast(), progress=True)
        return self.protocol.generated(lambda: model)()

    def real(self, cache: Path, log: Callable = print) -> Dict[str, dict]:
        """Real training poses and real poses of the split in place of the samples: the yardsticks of the protocol,
        the same entries as hp.evaluation.pareto.evaluate_quality writes for the test. The rows of the table."""
        settings = self.settings("paper" if self.split == "test" else "generation")
        name = f"real {self.split} poses"
        own = {"test": len(self.poses), **settings} if self.split == "test" else settings
        stage = "paper" if self.split == "test" else "generation"
        results = {
            "real train poses": P.cached("real train poses", stage, {**settings, **self._role("train")},
                                         self.protocol.real(self.train, 2), cache=cache, log=log),
            name: P.cached(name, stage, {**own, **self._role(self.split)},
                           self.protocol.real(self.poses, 1), cache=cache, log=log)}
        return {label: numbers({stage: result}) for label, result in results.items()}

    def _role(self, role: str) -> dict:
        """The test entries tell the two rows apart by "test" alone (as pareto); the validation says it."""
        return {} if self.split == "test" else {"real": role}


def stage_state(cache: Path, label: str, stage: str, settings: dict) -> str:
    """"cached" (made with these settings, also under another label), "other" or "missing"."""
    path = Path(cache) / P.slug(label) / f"{stage}.json"
    settings = json.loads(json.dumps(settings))
    if path.is_file():
        saved = json.loads(path.read_text())
        return "cached" if P.portable(saved.get("settings")) == P.portable(settings) else "other"
    return "cached" if P.same_elsewhere(stage, settings, Path(cache))[0] is not None else "missing"


def run_stage(arena: Arena, label: str, stage: str, build: Callable, identity: dict, cache: Path,
              overwrite: bool = False, force: bool = False, log: Callable = print) -> Optional[dict]:
    """The result of a stage, from the cache or computed. An entry made with other settings is replaced only with
    overwrite=True; otherwise OtherSettings."""
    settings = {"model": identity, **arena.settings(stage)}
    if stage_state(cache, label, stage, settings) == "other" and not (overwrite or force):
        raise OtherSettings(f"{label} / {stage} was made with other settings; not replaced")
    return P.cached(label, stage, settings, lambda: arena.compute(stage, build), force=force, cache=Path(cache), log=log)


def numbers(results: Dict[str, Optional[dict]]) -> dict:
    """The flat numbers of a row from the results of its stages; the half-width of the 95% interval over the sets as
    <key>_ci, the standard deviation over the repeats of the harness as <key>_std."""
    out = {}
    for name, value in (results.get("generation") or results.get("paper") or {}).items():
        key = name.replace(" ", "_")
        out[key], out[key + "_ci"] = value["mean"], value["ci95"]
    harness = results.get("harness")
    if harness:
        for key, part in (("wasserstein", harness["wasserstein"]["generated"]["joint_mean_deg"]),
                          ("recall", harness["recall"]["mean_deg"]), ("precision", harness["precision"]["mean_deg"])):
            out[key], out[key + "_std"] = part["mean"], part["std"]
        for key, part in (("wasserstein", harness["wasserstein"].get("contrast", {}).get("joint_mean_deg")),
                          ("recall", harness.get("contrast_recall", {}).get("mean_deg")),
                          ("precision", harness.get("contrast_precision", {}).get("mean_deg"))):
            if part:
                out[key + "_contrast"] = part["mean"]  # real training poses in place of the samples
    ll = results.get("likelihood") or (harness or {}).get("likelihood")
    if ll:
        out["log_likelihood"], out["log_likelihood_std"] = ll["mean"], ll["std"]
        if ll.get("nan_fraction"):
            out["log_likelihood_nan"] = ll["nan_fraction"]
    return out


def table(rows: List[dict], split: str) -> str:
    """One row per model: the columns of COLUMNS ("-" where a stage is missing or the model has no density)."""
    columns = COLUMNS[split]
    lines = ["| model | " + " | ".join(title for _, title, _ in columns) + " |", "|" + "---|" * (len(columns) + 1)]
    for row in rows:
        cells = [f"{row[key]:.{digits}f}" if row.get(key) is not None else "-" for key, _, digits in columns]
        lines.append("| " + " | ".join([row["label"], *cells]) + " |")
    return "\n".join(lines)
