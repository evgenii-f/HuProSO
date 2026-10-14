"""Evaluate a PoseModel: how well do its samples match real poses?

    report = evaluate(model, reference_poses)

works for every model of the zoo, because it needs only model.sample() (and model.log_prob() if there is one).
Everything is drawn without replacement from generators with a fixed seed, and the random state of the caller
is left as it was, so a report is reproducible and an evaluation does not disturb a training.

What is measured (angles are per joint, in degrees, unless it says radians):
  wasserstein   optimal transport between generated and real poses, per joint and for the whole pose, next to
                its noise floor (the distance of two disjoint samples of real poses). A ratio near 1: as good as
                real data.
  recall        distance of a real pose to the nearest generated one: are all real poses covered?
  precision     distance of a generated pose to the nearest real one: are the generated poses realistic?
  baseline      the same distance of real poses to *other real poses of the same set*.
  contrast      (if `contrast` poses are given) the same numbers for real poses of *other subjects*, e.g. the
                training poses.
  likelihood    mean log-likelihood of real poses, only for models that have one (and only comparable between
                models with the same density_space).
Every number is the mean and standard deviation over `repeats` independent draws of the poses compared.

Reading the numbers: baseline and contrast
  Poses of one recording are strongly correlated: at 120 frames per second neighbouring frames are nearly
  identical. So a real test pose has a nearly identical twin among the other real test poses, and the noise
  floor and the baseline are tiny (on the AMASS test set: Wasserstein 6 deg per joint, nearest neighbour 1.5
  deg). A model has never seen those subjects and cannot get that close, so "ratio to the floor" is not a
  score and these two numbers are not goals.
  The contrast is the yardstick that can be reached. Real poses of other subjects (the training poses) stand
  in for a generator, and the same three numbers are computed for them: Wasserstein to the test poses, test
  poses to their nearest contrast pose (like recall), contrast poses to their nearest test pose (like
  precision). A model that reproduces the distribution of the training poses exactly would score about the
  same as the contrast. On the AMASS test set that is 17.5 deg per joint (Wasserstein) and 11.8 / 13.6
  (recall / precision), against 6.4 and 1.7 for the same-set numbers. A model close to the contrast is as good
  as real unseen data; one that generalises better than its training data can even pass it.
  Caveats: the contrast is a sample of the training poses, so it is noisy by the std of the repeats, and it
  is only as representative as the training subjects are of the test subjects.
"""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from hp.evaluation.neighbours import nearest_distances, summarize
from hp.evaluation.wasserstein import wasserstein_with_floor
from hp.models.base import PoseModel
from hp.models.io import load_model


@dataclass
class EvalConfig:
    n_wasserstein: int = 2000  # poses per sample in the Wasserstein distance
    n_queries: int = 2000  # poses whose nearest neighbour is looked for in recall and precision
    n_generated: int = 20_000  # generated poses (the candidates of the recall; real candidates for precision)
    n_likelihood: int = 5000  # real poses for the likelihood
    repeats: int = 3
    seed: int = 0
    sample_batch: int = 2000  # poses drawn from the model at a time
    chunk: int = 20_000  # candidates in one block of the nearest-neighbour search


def _aggregate(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {key: {"mean": float(np.mean([r[key] for r in runs])), "std": float(np.std([r[key] for r in runs]))} for key in runs[0]}


@torch.no_grad()
def _generate(model: PoseModel, n: int, batch: int, progress: bool = False):
    start, pieces = time.perf_counter(), []
    for done in tqdm(range(0, n, batch), desc="sampling", unit="batch", disable=not progress):
        pieces.append(model.sample(min(batch, n - done)).float().cpu())
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(pieces), time.perf_counter() - start


def _wasserstein(reference, generated, cfg, rng, joints, contrast=None, progress=False):
    problems = cfg.repeats * (joints + 1) * (2 if contrast is None else 3)  # transport problems
    with tqdm(total=problems, desc="wasserstein", unit="OT", disable=not progress) as bar:
        runs = [wasserstein_with_floor(reference, generated, cfg.n_wasserstein, rng, contrast, step=bar.update)
                for _ in range(cfg.repeats)]
    out = {"n": runs[0]["n"]}
    for key in ("generated", "floor", *(["contrast"] if contrast is not None else [])):
        sums = [r[key]["joint_sum_rad"] for r in runs]
        per_joint = np.array([r[key]["per_joint_rad"] for r in runs])  # [repeats, J]
        out[key] = {
            "joint_sum_rad": {"mean": float(np.mean(sums)), "std": float(np.std(sums))},
            "joint_mean_deg": {"mean": float(np.rad2deg(np.mean(sums)) / joints), "std": float(np.rad2deg(np.std(sums)) / joints)},
            "by_joint_deg": np.rad2deg(per_joint.mean(axis=0)).tolist(),
        }
    ratios = [r["generated"]["joint_sum_rad"] / r["floor"]["joint_sum_rad"] for r in runs]
    out["ratio_to_floor"] = {"mean": float(np.mean(ratios)), "std": float(np.std(ratios))}
    return out


def _coverage(reference, generated, cfg, rng, contrast=None, progress=False):
    runs, sizes = [], {}
    for _ in tqdm(range(cfg.repeats), desc="nearest neighbours", unit="repeat", disable=not progress):
        n_q = min(cfg.n_queries, len(reference) // 2, len(generated))
        n_c = min(cfg.n_generated, len(reference) - n_q)
        order = rng.permutation(len(reference))
        real_queries, real_candidates = reference[order[:n_q]], reference[order[n_q : n_q + n_c]]
        generated_queries = generated[rng.choice(len(generated), n_q, replace=False)]
        run = {
            "recall": summarize(nearest_distances(real_queries, generated, chunk=cfg.chunk)),
            "precision": summarize(nearest_distances(generated_queries, real_candidates, chunk=cfg.chunk)),
            "baseline": summarize(nearest_distances(real_queries, real_candidates, chunk=cfg.chunk)),
        }
        sizes = {"queries": n_q, "real_candidates": n_c, "generated_candidates": len(generated)}
        if contrast is not None:  # real poses of other subjects play the part of the generator
            pool = contrast[rng.choice(len(contrast), min(cfg.n_generated, len(contrast)), replace=False)]
            queries = contrast[rng.choice(len(contrast), min(n_q, len(contrast)), replace=False)]
            run["contrast_recall"] = summarize(nearest_distances(real_queries, pool, chunk=cfg.chunk))
            run["contrast_precision"] = summarize(nearest_distances(queries, real_candidates, chunk=cfg.chunk))
            sizes["contrast_candidates"] = len(pool)
        runs.append(run)
    out = {name: _aggregate([r[name] for r in runs]) for name in runs[0]}
    out["sizes"] = sizes
    return out


@torch.no_grad()
def _likelihood(model, reference, cfg, rng, progress=False):
    if not model.has_log_prob:
        return None
    means, nan_fraction = [], []
    n = min(cfg.n_likelihood, len(reference))
    bar = tqdm(total=cfg.repeats * n, desc="log-likelihood", unit="pose", disable=not progress)
    for _ in range(cfg.repeats):
        poses = reference[rng.choice(len(reference), n, replace=False)]
        pieces = []
        for i in range(0, len(poses), cfg.sample_batch):
            pieces.append(model.log_prob(poses[i : i + cfg.sample_batch].to(model.device)).float().cpu())
            bar.update(len(pieces[-1]))
        values = torch.cat(pieces)
        nan_fraction.append(float((~torch.isfinite(values)).float().mean()))
        means.append(float(values[torch.isfinite(values)].mean()))
    bar.close()
    return {"mean": float(np.mean(means)), "std": float(np.std(means)), "nan_fraction": float(np.mean(nan_fraction)),
            "space": model.density_space, "n": min(cfg.n_likelihood, len(reference))}


NOTES = {
    "units": "Angles are geodesic angles in degrees per joint (the mean over joints), unless the key says rad.",
    "wasserstein": "Optimal transport between generated and real poses. joint_sum_rad is the HuProSO distance "
                   "(sum of the joint angles, radians); joint_mean_deg is the same divided by the joints, in degrees.",
    "floor_and_baseline": "The noise floor and the baseline compare real poses with other poses of the same set. "
                          "Neighbouring frames of a recording are nearly identical, so these are much smaller than any "
                          "model can reach. They are no goals, and ratio_to_floor is not a score.",
    "contrast": "Real poses of other subjects (e.g. the training poses) in place of the generator: the distance a "
                "model would have if it reproduced the training distribution exactly. Compare the model with this.",
}


def _plain(value):
    return json.loads(json.dumps(value, default=lambda o: OmegaConf.to_container(o) if OmegaConf.is_config(o) else str(o)))


def evaluate(model: PoseModel, reference: torch.Tensor, config: Optional[EvalConfig] = None,
             contrast: Optional[torch.Tensor] = None, progress: bool = False) -> dict:
    """Evaluate `model` against real poses [N, J, 3, 3]. Returns a report that json can write.
    `contrast`: real poses of other subjects (e.g. the training poses), for the yardstick numbers.
    `progress`: a progress bar for every stage (sampling, Wasserstein, nearest neighbours, log-likelihood)."""
    cfg = config or EvalConfig()
    reference = torch.as_tensor(reference).float().cpu()
    if contrast is not None:
        contrast = torch.as_tensor(contrast).float().cpu()
        if contrast.ndim != 4 or contrast.shape[1:] != reference.shape[1:] or len(contrast) < 2:
            raise ValueError(f"The contrast poses must have shape [N, {model.num_joints}, 3, 3], got {tuple(contrast.shape)}.")
    if reference.ndim != 4 or reference.shape[1:] != (model.num_joints, 3, 3):
        raise ValueError(f"The real poses must have shape [N, {model.num_joints}, 3, 3], got {tuple(reference.shape)}.")
    if len(reference) < 4 or not torch.isfinite(reference).all():
        raise ValueError("The real poses must be finite, and there must be at least 4 of them.")

    was_training, rng = model.training, np.random.default_rng(cfg.seed)
    model.eval()
    try:
        devices = [model.device] if model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):  # the random state of the caller is restored afterwards
            torch.manual_seed(cfg.seed)
            generated, seconds = _generate(model, cfg.n_generated, cfg.sample_batch, progress)
        if generated.shape != (cfg.n_generated, model.num_joints, 3, 3) or not torch.isfinite(generated).all():
            raise ValueError("The model produced samples of the wrong shape or with non-finite values.")
        report = {
            "model": {"name": model.name, "parameters": model.num_parameters(), "build_args": _plain(model.build_args)},
            "config": asdict(cfg),
            "notes": {k: v for k, v in NOTES.items() if contrast is not None or k != "contrast"},
            "reference": {"poses": len(reference), "joints": model.num_joints},
            "sampling": {"generated": cfg.n_generated, "seconds_per_1000": seconds / cfg.n_generated * 1000},
            "wasserstein": _wasserstein(reference, generated, cfg, rng, model.num_joints, contrast, progress),
            **_coverage(reference, generated, cfg, rng, contrast, progress),
            "likelihood": _likelihood(model, reference, cfg, rng, progress),
        }
    finally:
        model.train(was_training)
    return report


def save_report(report: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def load_report(path) -> dict:
    return json.loads(Path(path).read_text())


def evaluate_run(run_dir, reference: torch.Tensor, config: Optional[EvalConfig] = None, weights: str = "weights",
                 output=None, device: str = "cpu", contrast: Optional[torch.Tensor] = None, progress: bool = False) -> dict:
    """Load a saved run (see hp.models.io), evaluate it and write the report (default: <run_dir>/metrics.json)."""
    run_dir = Path(run_dir)
    report = evaluate(load_model(run_dir, device=device, weights=weights), reference, config, contrast, progress)
    report["run"] = {"directory": str(run_dir), "weights": weights}
    save_report(report, output or run_dir / "metrics.json")
    return report
