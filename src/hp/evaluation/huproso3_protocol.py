"""The evaluation protocol of the HuProSO3 paper (Dünkel, Salzmann, Pfaff), for any PoseModel.

    report = evaluate_paper_protocol(model, {"test": test_poses, "train": train_poses})

Unlike hp.evaluation.evaluate (the harness of the model zoo, with its own sizes, units and yardsticks), this
reproduces what the paper reports for the unconditional prior, so the numbers can be put next to its tables:

  Tab. 1   recall and precision of the summed geodesic distance (radians, the sum over the joints of
           arccos((tr(R_i^T R_j) - 1) / 2), Eq. 9), mean [median], on the test and the train split.
           Recall: 1k dataset poses against 100k poses sampled from the model, the distance of each dataset pose
           to its nearest sample. Precision: 1k sampled poses against 100k dataset poses, the distance of each
           sample to its nearest dataset pose.
  Tab. 6   the same with the MPJPE (mm) after forward kinematics of the SMPL body (the joint positions of
           hp.utils.skeleton_forward_kinematics, the code of the paper); the distance of two poses is the mean
           over the joints of the distance of the joint positions.
  Fig. 10, 11   the precision and recall curves: P(distance < eps). The distances are kept in the report.
  Tab. 7   the mean log-likelihood of dataset poses (relative to the uniform distribution on SO(3)^J), test and
           train. The paper's config draws num_eval_samples = 5000 poses.

Where the paper says nothing, the choices are the ones of the paper's code (hp.analysis.compute_metrics,
hp.core.evaluator): 19 joints (the text says J = 21, the code and the model have 19), the geodesic sum over
them, the 22 joint positions of the forward kinematics (the root is always at the origin, so it adds 0 to the
sum but counts in the mean). Two additions that the paper does not have: every number is repeated `repeats`
times with independent draws (mean and std over the repeats), and the log-likelihood comes with robust
statistics (median, trimmed mean, the share of values outside a plausible range), because a mean can be
dominated by a few broken values (docs/rsgm_rare_pose_instability.md).

The 100k model samples are drawn once and shared by the repeats (sampling a flow takes minutes); the repeats
draw new dataset poses and new 1k subsets of the samples.
"""
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from hp.data.amass_cfg import AmassJoints4Learning
from hp.evaluation.neighbours import nearest_distances
from hp.models.base import PoseModel
from hp.utils.skeleton_forward_kinematics import perform_forward_kinematics

# The numbers of the paper for its prior (HuProSO3, "Ours") and the baselines, as printed.
PAPER = {
    "geodesic_rad": {  # Tab. 1: mean [median]
        "GAN-S": {"test": {"recall": (3.76, 3.34), "precision": (4.51, 4.23)}, "train": {"recall": (3.57, 3.34), "precision": (4.38, 4.13)}},
        "6D NF": {"test": {"recall": (3.66, 3.16), "precision": (4.50, 4.00)}, "train": {"recall": (3.55, 3.32), "precision": (4.42, 4.10)}},
        "HuProSO3": {"test": {"recall": (3.44, 2.95), "precision": (4.24, 3.71)}, "train": {"recall": (2.93, 2.64), "precision": (3.90, 3.59)}},
    },
    "mpjpe_mm": {  # Tab. 6: mean [median]
        "GAN-S": {"test": {"recall": (50.6, 48.4), "precision": (68.7, 64.2)}, "train": {"recall": (39.7, 35.0), "precision": (56.2, 49.7)}},
        "6D NF": {"test": {"recall": (46.3, 39.9), "precision": (65.8, 57.8)}, "train": {"recall": (39.4, 34.2), "precision": (57.2, 49.4)}},
        "HuProSO3": {"test": {"recall": (45.4, 40.2), "precision": (61.2, 52.1)}, "train": {"recall": (34.3, 28.8), "precision": (51.3, 43.2)}},
    },
    "log_likelihood": {"HuProSO3": {"test": 137.6, "train": 184.7}},  # Tab. 7, "Prior" and "Prior (Train)"
}


@dataclass
class PaperProtocolConfig:
    n_queries: int = 1_000  # dataset poses of the recall, sampled poses of the precision
    n_pool: int = 100_000  # sampled poses of the recall, dataset poses of the precision
    n_likelihood: int = 5_000  # dataset poses per log-likelihood evaluation (num_eval_samples of the paper)
    repeats: int = 3
    seed: int = 0
    sample_batch: int = 5_000  # poses drawn from the model at a time
    likelihood_batch: int = 1_000  # poses of one log_prob call
    chunk: int = 10_000  # candidates in one block of the nearest-neighbour search
    plausible: Tuple[float, float] = (0.0, 250.0)  # per-pose log-likelihoods outside are counted as implausible
    mpjpe: bool = True  # also the MPJPE metrics (needs the 19 joints of AmassJoints4Learning)


def joint_positions(poses: torch.Tensor, batch: int = 50_000) -> torch.Tensor:
    """Joint positions [N, 22, 3] in mm of poses [N, 19, 3, 3] (the forward kinematics of the paper's code)."""
    if poses.shape[1:] != (len(AmassJoints4Learning), 3, 3):
        raise ValueError(f"Forward kinematics needs poses [N, {len(AmassJoints4Learning)}, 3, 3], got {tuple(poses.shape)}.")
    return 1000 * torch.cat([perform_forward_kinematics(poses[i : i + batch].float(), selected_joints=AmassJoints4Learning)
                             for i in range(0, len(poses), batch)])


def nearest_mpjpe(queries: torch.Tensor, candidates: torch.Tensor, chunk: int = 10_000) -> torch.Tensor:
    """MPJPE [Nq] of every query [Nq, K, 3] to its nearest candidate [Nc, K, 3]: min over the candidates of the
    mean over the K joints of the Euclidean distance. Exact, streamed over blocks of candidates."""
    best = torch.full((len(queries),), float("inf"))
    for start in range(0, len(candidates), chunk):
        block = candidates[start : start + chunk]
        total = sum(torch.cdist(queries[:, k], block[:, k]) for k in range(queries.shape[1]))
        best = torch.minimum(best, total.min(dim=1).values / queries.shape[1])
    return best


def _stats(x: np.ndarray) -> Dict[str, float]:
    return {"mean": float(np.mean(x)), "median": float(np.median(x))}


def _aggregate(runs):
    """[{metric: {stat: value}}] over the repeats -> {metric: {stat: {"mean", "std"}}}."""
    return {m: {s: {"mean": float(np.mean([r[m][s] for r in runs])), "std": float(np.std([r[m][s] for r in runs]))}
                for s in runs[0][m]} for m in runs[0]}


def precision_recall(generated: torch.Tensor, real: torch.Tensor, cfg: PaperProtocolConfig, rng: np.random.Generator,
                     rng_generated: np.random.Generator, generated_positions: Optional[torch.Tensor] = None,
                     progress: bool = False) -> dict:
    """Recall and precision of `generated` poses against `real` poses of one split, `repeats` times. `rng` draws
    the dataset poses, `rng_generated` the subsets of the generated ones, so that the dataset poses do not depend
    on how many poses were generated."""
    n_q = min(cfg.n_queries, len(real), len(generated))
    n_c = min(cfg.n_pool, len(real))
    runs, curves = [], None
    for _ in tqdm(range(cfg.repeats), desc="precision / recall", unit="repeat", disable=not progress):
        real_queries = real[rng.choice(len(real), n_q, replace=False)]
        real_candidates_idx = rng.choice(len(real), n_c, replace=False)
        gen_idx = rng_generated.choice(len(generated), n_q, replace=False)
        d = {
            "recall_geodesic": nearest_distances(real_queries, generated, "sum", cfg.chunk),
            "precision_geodesic": nearest_distances(generated[gen_idx], real[real_candidates_idx], "sum", cfg.chunk),
        }
        if generated_positions is not None:
            d["recall_mpjpe"] = nearest_mpjpe(joint_positions(real_queries), generated_positions, cfg.chunk)
            d["precision_mpjpe"] = nearest_mpjpe(generated_positions[gen_idx], joint_positions(real[real_candidates_idx]), cfg.chunk)
        runs.append({k: _stats(v.double().numpy()) for k, v in d.items()})
        if curves is None:  # the distances of the first repeat, for the curves of Fig. 10 and 11
            curves = {k: v.double().numpy().tolist() for k, v in d.items()}
    return {"sizes": {"queries": n_q, "pool_generated": len(generated), "pool_real": n_c}, **_aggregate(runs), "distances": curves}


@torch.no_grad()
def likelihood(model: PoseModel, real: torch.Tensor, cfg: PaperProtocolConfig, rng: np.random.Generator,
               progress: bool = False) -> Optional[dict]:
    """Mean log-likelihood of real poses (the number of Tab. 7) and robust statistics, `repeats` times."""
    if not model.has_log_prob:
        return None
    n = min(cfg.n_likelihood, len(real))
    lo, hi = cfg.plausible
    runs, values_first = [], None
    bar = tqdm(total=cfg.repeats * n, desc="log-likelihood", unit="pose", disable=not progress)
    for _ in range(cfg.repeats):
        poses = real[rng.choice(len(real), n, replace=False)]
        pieces = []
        for i in range(0, n, cfg.likelihood_batch):
            pieces.append(model.log_prob(poses[i : i + cfg.likelihood_batch].to(model.device)).cpu().double())  # mps has no float64
            bar.update(len(pieces[-1]))
        v = torch.cat(pieces).numpy()
        finite = v[np.isfinite(v)]
        lo_q, hi_q = np.percentile(finite, [1, 99])
        runs.append({"ll": {
            "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "trimmed_mean": float(finite[(finite >= lo_q) & (finite <= hi_q)].mean()),  # without the 1% at each end
            "min": float(finite.min()),
            "max": float(finite.max()),
            "nonfinite_share": float(1 - len(finite) / len(v)),
            "implausible_share": float(((finite < lo) | (finite > hi)).sum() / len(v)),
        }})
        if values_first is None:
            values_first = v.tolist()
    bar.close()
    return {"n": n, "space": model.density_space, **_aggregate(runs)["ll"], "values": values_first}


@torch.no_grad()
def sample_pool(model: PoseModel, n: int, batch: int, seed: int, progress: bool = False) -> Tuple[torch.Tensor, float]:
    """n poses of the model (float32, CPU) from a fixed seed, and the seconds it took. The random state of the
    caller is left as it was."""
    devices = [model.device] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        start, pieces = time.perf_counter(), []
        for done in tqdm(range(0, n, batch), desc="sampling", unit="batch", disable=not progress):
            pieces.append(model.sample(min(batch, n - done)).float().cpu())
        seconds = time.perf_counter() - start
    poses = torch.cat(pieces)
    if not torch.isfinite(poses).all():
        raise ValueError("The model produced non-finite samples.")
    return poses, seconds


def evaluate_paper_protocol(model: Optional[PoseModel], splits: Dict[str, torch.Tensor],
                            config: Optional[PaperProtocolConfig] = None, generated: Optional[torch.Tensor] = None,
                            progress: bool = False) -> dict:
    """The metrics of the HuProSO3 paper for `model` on every split of `splits` ({"test": poses, "train": poses},
    each [N, J, 3, 3]). Instead of a model, `generated` poses can be given (e.g. real poses of other subjects as a
    yardstick); then there is no log-likelihood."""
    cfg = config or PaperProtocolConfig()
    if (model is None) == (generated is None):
        raise ValueError("Give either a model or generated poses.")
    splits = {name: torch.as_tensor(p).float().cpu() for name, p in splits.items()}
    joints = model.num_joints if model is not None else generated.shape[1]
    for name, p in splits.items():
        if p.ndim != 4 or p.shape[1:] != (joints, 3, 3) or not torch.isfinite(p).all():
            raise ValueError(f"The {name} poses must be finite with shape [N, {joints}, 3, 3], got {tuple(p.shape)}.")
    mpjpe = cfg.mpjpe and joints == len(AmassJoints4Learning)

    report = {"config": asdict(cfg), "protocol": "HuProSO3 paper: Tab. 1, 6, 7 and Fig. 10, 11"}
    was_training = model.training if model is not None else None
    if model is not None:
        model.eval()
    try:
        if generated is None:
            generated, seconds = sample_pool(model, cfg.n_pool, cfg.sample_batch, cfg.seed, progress)
            report["model"] = {"name": model.name, "parameters": model.num_parameters(), "build_args": model.build_args}
            report["sampling"] = {"n": len(generated), "seconds_per_1000": seconds / len(generated) * 1000}
        else:
            generated = torch.as_tensor(generated).float().cpu()
            report["model"] = {"name": "given poses"}
            report["sampling"] = {"n": len(generated)}
        positions = joint_positions(generated) if mpjpe else None
        for i, (name, real) in enumerate(splits.items()):
            rngs = [np.random.default_rng([cfg.seed, i, k]) for k in range(3)]  # the same dataset poses for every model
            report[name] = {"poses": len(real), **precision_recall(generated, real, cfg, rngs[0], rngs[1], positions, progress)}
            report[name]["likelihood"] = likelihood(model, real, cfg, rngs[2], progress) if model is not None else None
    finally:
        if model is not None:
            model.train(was_training)
    return report
