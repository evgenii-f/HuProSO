"""Train RSGM with the rounded settings of the comparison with PoseD-Flow, on their AMASS or on ours (the script
version of explore/training/rsgm_posedflow_train.ipynb, e.g. for a cluster):

    python scripts/train_rsgm_posedflow.py --config large --seeds 10 11 --device cuda:0
    python scripts/train_rsgm_posedflow.py --config small --dry-run          # print the configuration only
    python scripts/train_rsgm_posedflow.py --config search train.time_limit_minutes=230
    python scripts/train_rsgm_posedflow.py --data amass --config small search    # our AMASS, two networks in a row

--data picks the data (DATASETS): posedflow, their AMASS converted by scripts/preprocess/convert_posedflow.py (train
every 10th pose, valid every 100th), or amass, our AMASS (SMPL-X G, the official split) compressed to every 10th frame,
the arena of the HuProSO3 comparison (explore/evaluation/rsgm_huproso_comparison.ipynb). --rotations overrides the
folder. The settings are rounded values from the plateau of the last Optuna search on our AMASS (not tuned for either
data); CONFIGS picks the network, and several configs are trained one after the other. Every config and seed is its own
run, <run root>/<name>/run_000N, with <name> rsgm_posedflow[_<config>]_s<seed> on their data and
rsgm_amass_<folder>[_<config>]_s<seed> on ours (<folder>: c10 for AMASS_rotations_c10, full for AMASS_rotations):
- a finished run is skipped (--retrain trains a new one);
- a run that stopped early (time limit, SIGTERM) is resumed from its newest checkpoint.
Further key=value arguments go to scripts/train_model.py and win over the settings here.

Progress: a header per run ([k/n], the network, the epochs), the trainer's progress bar with the elapsed and remaining
time of the run, its time when it ends, and a summary of all runs at the end.

The exit status is 0 when every run is finished and 75 when one stopped early and can be resumed (as train_model.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

# the code of this checkout, before an editable install of another one
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hp.evaluation.data import load_reference  # noqa: E402
from hp.models.io import load_model  # noqa: E402
from hp.models.rsgm import fit_joint_scale, joint_cov_power, joint_covariance, z_statistics  # noqa: E402
from hp.training import cli  # noqa: E402
from hp.training.config import load_config  # noqa: E402

# ---- parameters ----------------------------------------------------------------------
DATASETS = {  # --data: the rotation folder and the split
    "posedflow": dict(rotations="~/data/AMASS_rotations_posedflow_c10", split="posedflow"),
    "amass": dict(rotations="~/data/AMASS_rotations_c10", split="original"),
}
RUN_ROOT = "~/data/experiments"
SEEDS = [10]
DEVICE = "mps"  # "cpu", "mps" or "cuda:N"
N_CHECK = 500  # valid poses for the quick likelihood check after training (0: no check)

COMMON = [  # rounded from the plateau of the search
    "model.args.activation=silu",
    "model.args.beta_f=7.5",
    "model.args.loss_t_min=0.002",
    "model.args.weighting=std",
    "train.lr=0.002",
    "train.batch_size=4096",
    "train.ema_decay=0.999",
    "train.scheduler=step",
    "train.optimizer=adamw",
    "train.weight_decay=1e-5",
    "train.max_iterations=30000",
    "train.valid_interval=1000",
    "train.checkpoint_every=5000",
]
CONFIGS = {  # the network, and what differs from COMMON
    "search": ["model.args.hidden=[1536, 1536, 1536, 1536, 1536, 1536]"],  # 12.2M parameters
    "small": ["model.args.hidden=[512, 512, 512, 512]"],  # the size of PoseD-Flow's network
    # the same with the times of the loss drawn as the likelihood weighting (about 34% at t <= 0.025 instead of 2%):
    # the small noise where the score distillation of DPoser's completion works
    "small_lik": ["model.args.hidden=[512, 512, 512, 512]", "model.args.weighting=likelihood"],
    # and the network of "search": does the sharper score at small noise need more capacity?
    "search_lik": ["model.args.hidden=[1536, 1536, 1536, 1536, 1536, 1536]", "model.args.weighting=likelihood"],
    "large": ["model.args.hidden=[4096, 4096, 4096, 4096, 4096, 4096]", "train.lr=1e-3"],  # 84.8M parameters
    # the time in every layer (hp.models.score_nets, docs/experiments/2026-11-14_score_network_architectures.md): a residual network as
    # DPoser's, 5.8M parameters at 1024 x 2 blocks, 1.9M at 512 x 2; lr 1e-3 for the wider residual network
    "timeres": ["model.args.net=timeres", "model.args.net_width=1024", "model.args.net_blocks=2",
                "model.args.dropout=0.1", "train.lr=1e-3"],
    "timeres_small": ["model.args.net=timeres", "model.args.net_width=512", "model.args.net_blocks=2",
                      "model.args.dropout=0.1", "train.lr=1e-3"],
    "timeres_lik": ["model.args.net=timeres", "model.args.net_width=1024", "model.args.net_blocks=2",
                    "model.args.dropout=0.1", "train.lr=1e-3", "model.args.weighting=likelihood"],
}
# The ablation of timeres_small (docs/score_network_architectures.md): abl_base has every component off (a plain MLP of
# 5 layers of 512 with t at the input, lr 2e-3 as the MLPs); every other abl_<component> turns on exactly one component
# of timeres_small. All of them on is timeres_small itself.
ABL_BASE = ["model.args.net=timeres", "model.args.net_width=512", "model.args.net_blocks=2", "model.args.dropout=0.0",
            "model.args.net_level=t", "model.args.net_time_in=input", "model.args.net_time_embed=scalar",
            "model.args.net_residual=false", "model.args.net_norm=false", "model.args.net_zero_init=false",
            "train.lr=2e-3"]
ABLATION = {
    "base": [],
    "time_layers": ["model.args.net_time_in=layers"],  # the time in every layer (as a scalar, as PoseRFM's ConcatLinear)
    "sinusoidal": ["model.args.net_time_embed=sinusoidal"],  # the sinusoidal embedding of the time (at the input)
    "log_s": ["model.args.net_level=log_s"],  # log s(t) / 4 instead of t (at the input)
    "residual": ["model.args.net_residual=true"],  # the skip connections of the blocks
    "norm": ["model.args.net_norm=true"],  # LayerNorm in every layer
    "dropout": ["model.args.dropout=0.1"],  # dropout 0.1 in every layer
    "zero_init": ["model.args.net_zero_init=true"],  # the last layer starts at zero
    "lr": ["train.lr=1e-3"],  # the learning rate of the timeres configs
}
CONFIGS.update({f"abl_{name}": ABL_BASE + extra for name, extra in ABLATION.items()})
# Depth instead of width: 256 x 8 blocks (17 layers, 2.3M parameters, as many as timeres_small's 1.9M), with and
# without the skip connections, which deep plain networks need to train
CONFIGS["deep"] = ["model.args.net=timeres", "model.args.net_width=256", "model.args.net_blocks=8",
                   "model.args.dropout=0.1", "train.lr=1e-3"]
CONFIGS["deep_nores"] = CONFIGS["deep"] + ["model.args.net_residual=false"]
# The two components that carry the ablation, together and alone: a plain MLP with the skip connections and LayerNorm
# (the time a scalar t at the input, no dropout, no zero init, lr 2e-3 as abl_base)
CONFIGS["res_norm"] = ABL_BASE + ["model.args.net_residual=true", "model.args.net_norm=true"]
# The z-scored tangent coordinates next to the matrices at the input (inputs="local+z", hp.models.rsgm, as DPoser's
# input), with abl_base and with res_norm: the pairs differ by the input only. The statistics are fitted on the training
# poses once (fitted_z_statistics) and given as model.args.z_mean / z_std.
Z_INPUT = ["model.args.inputs=local+z"]
CONFIGS["abl_z"] = ABL_BASE + Z_INPUT
CONFIGS["res_norm_z"] = CONFIGS["res_norm"] + Z_INPUT
# Per-joint noise (joint_scale, hp.models.rsgm): the Brownian time c_j s(t) with c_j proportional to the spread of joint
# j (fit_joint_scale of the z statistics), as DPoser's noise in z-scored coordinates; with abl_base, the pair differs by
# the noise only. The configs in ANISO get model.args.joint_scale from the statistics.
# Where the loss is evaluated (docs: the noise levels of training), with the network of abl_residual: the
# likelihood weighting (about log-uniform in s at small s) and EDM's log-normal noise levels in s, narrow and wide
CONFIGS["abl_residual_lik"] = CONFIGS["abl_residual"] + ["model.args.weighting=likelihood"]
# share of the loss at s < 0.01 / s > 1: std 6% / 42%, likelihood 51% / 16%, ln_m3 21% / 7%, ln_m4 38% / 2%
CONFIGS["abl_residual_ln_m3"] = CONFIGS["abl_residual"] + ["model.args.weighting=lognormal",
                                                           "model.args.loss_log_s_mean=-3.0",
                                                           "model.args.loss_log_s_std=2.0"]
CONFIGS["abl_residual_ln_m4"] = CONFIGS["abl_residual"] + ["model.args.weighting=lognormal",
                                                           "model.args.loss_log_s_mean=-4.0",
                                                           "model.args.loss_log_s_std=2.0"]
CONFIGS["abl_aniso"] = list(ABL_BASE)
# and with the two components that stop the drift of the second phase, one at a time (as abl_residual and abl_norm)
CONFIGS["abl_residual_aniso"] = ABL_BASE + ABLATION["residual"]
CONFIGS["abl_norm_aniso"] = ABL_BASE + ABLATION["norm"]
ANISO = {"abl_aniso", "abl_residual_aniso", "abl_norm_aniso"}
# DPoser's network and training on RSGM: TimeFC's size (1024 x 2 blocks, a 512-dim embedding, dropout 0.1) and
# DPoser's schedule (200k steps x 5120, lr 2e-4, cosine with 5k warm-up, EMA 0.9999). For the full data, as DPoser's
# full run: --rotations ~/data/AMASS_rotations_posedflow --run-root ~/data/experiments/full
CONFIGS["dposer_like"] = ["model.args.net=timeres", "model.args.net_width=1024", "model.args.net_blocks=2",
                          "model.args.net_embed=512", "model.args.dropout=0.1", "train.lr=2e-4",
                          "train.batch_size=5120", "train.max_iterations=200000", "train.scheduler=cosine",
                          "train.warmup_steps=5000", "train.ema_decay=0.9999", "train.valid_interval=5000",
                          "train.checkpoint_every=10000"]
# and with the z-scored tangent input (Z_INPUT): the pair dposer_like / dposer_like_z differs by the input only
CONFIGS["dposer_like_z"] = CONFIGS["dposer_like"] + Z_INPUT
# and with the per-joint noise (joint_scale, as abl_aniso): the pair dposer_like / dposer_like_aniso differs by the
# noise only. For the full data, as dposer_like: --rotations ~/data/AMASS_rotations_posedflow --run-root ~/data/experiments/full
CONFIGS["dposer_like_aniso"] = list(CONFIGS["dposer_like"])
ANISO.add("dposer_like_aniso")
# The full 2^5 factorial of section 13 of docs/experiments/2026-11-14_score_network_architectures.md: abl_base with
# residual (R), LayerNorm (N), per-joint noise (A), dropout 0.1 (D) and width 1024 (W) each on or off. The name of a
# cell is fact_ followed by the letters that are on, e.g. fact_NA (fact_0: all off, = abl_base). All of them
# train with the batches sliced on the device.
FACTORS = {"R": ["model.args.net_residual=true"], "N": ["model.args.net_norm=true"], "A": [],
           "D": ["model.args.dropout=0.1"], "W": ["model.args.net_width=1024"]}
for bits in range(2 ** len(FACTORS)):
    on = [f for i, f in enumerate(FACTORS) if bits >> i & 1]
    name = "fact_" + ("".join(on) or "0")
    CONFIGS[name] = ABL_BASE + [o for f in on for o in FACTORS[f]] + ["train.data_on_device=true"]
    if "A" in on:
        ANISO.add(name)
FACTORIAL = [c for c in CONFIGS if c.startswith("fact_")]
# Depth: do the effects of the factorial hold in a deeper network? Cells of the factorial with 5 blocks (11 layers)
# instead of 2, everything else as the cell (lr 2e-3 of abl_base): with residual at both widths (R, RD, RN, RA, RAD
# and the same with W), and the control without residual DW, whose pair is RDW. The depth-2 twin of every cell is the
# cell of the same letters above. Run under their own root (~/data/experiments/factorial_depth).
DEPTH5 = ["R", "RD", "RN", "RA", "RAD", "RW", "RDW", "RNW", "RAW", "RADW", "DW"]
# Second round: the cells with LayerNorm that complete the 2^4 of N, A, D, W with residual on.
DEPTH5 += ["RND", "RNDW", "RNAD", "RNADW", "RNA", "RNAW"]
for cell in DEPTH5:
    CONFIGS[f"fact_d5_{cell}"] = CONFIGS[f"fact_{cell}"] + ["model.args.net_blocks=5"]
    if "A" in cell:
        ANISO.add(f"fact_d5_{cell}")
# The control without residual at lr 1e-3: did DW fail at depth 5 by its learning rate or by its structure?
CONFIGS["fact_d5_DW_lr0001"] = CONFIGS["fact_d5_DW"] + ["train.lr=1e-3"]
# Outside the design: fact_NA with the small noise of the wide joints trained (section 13). With joint_scale the
# smallest training noise of joint j is c_j s(loss_t_min), sqrt(c_j) times wider in angle than without it (1.4 deg for
# the knees). tmin: loss_t_min = 2.2e-4, where max_j c_j s(t) equals s(0.002) of the isotropic networks; lik: the
# likelihood weighting (more training at small t); tmin_lik: both.
CONFIGS["fact_NA_tmin"] = CONFIGS["fact_NA"] + ["model.args.loss_t_min=2.2e-4"]
CONFIGS["fact_NA_lik"] = CONFIGS["fact_NA"] + ["model.args.weighting=likelihood"]
CONFIGS["fact_NA_tmin_lik"] = CONFIGS["fact_NA"] + ["model.args.loss_t_min=2.2e-4", "model.args.weighting=likelihood"]
ANISO.update({"fact_NA_tmin", "fact_NA_lik", "fact_NA_tmin_lik"})
# Per-axis noise (joint_cov, aniso_x, hp.models.rsgm): the noise of every joint shaped by the covariance of
# log(M_j^T x_j) on its principal axes at small noise, as DPoser's z-scoring of each coordinate; with fact_NA
# (fact_NA: per-joint noise and LayerNorm, the factorial cell of abl_norm_aniso, the best hybrid of section 10) the
# pair fact_NA / fact_NA_x differs by the shape of the noise within the joints only. The configs in ANISO_X get
# model.args.joint_cov from the statistics (fitted_z_statistics, key z_cov).
CONFIGS["fact_NA_x"] = list(CONFIGS["fact_NA"])
ANISO.add("fact_NA_x")
ANISO_X = {"fact_NA_x"}
# The alpha study: the noise of fact_NA_x with c_j ~ sigma_j^(2 alpha) and the eigenvalue ratios within a joint
# lambda^alpha (fit_joint_scale, joint_cov_power). alpha = 1 is fact_NA_x (the same signal-to-noise ratio everywhere),
# alpha = 0 the isotropic noise (fact_N, with IGSO(3) instead of the wrapped normal at small noise); 0.5 is where the
# denoising loss itself would put the noise (a Gaussian estimate), 1.5 further toward anisotropy.
ALPHA = {"fact_NA_x_a05": 0.5, "fact_NA_x_a15": 1.5}
for name in ALPHA:
    CONFIGS[name] = list(CONFIGS["fact_NA_x"])
ANISO.update(ALPHA)
ANISO_X.update(ALPHA)
# Is the network the remainder of the gap to DPoser at equal noise (section 16)? The noise of fact_NA_x with a wide
# residual network: residual and dropout 0.1 as fact_RADW (the best hybrid of the factorial; without LayerNorm, which
# with residual was worse than their sum), 1536 x 3 blocks (7 layers, 14.5M parameters), lr 1e-3 as the wide residual
# configs. Meant for the full data: --rotations ~/data/AMASS_rotations_posedflow --run-root ~/data/experiments/full
CONFIGS["wide_RAD_x"] = ABL_BASE + ["model.args.net_residual=true", "model.args.dropout=0.1",
                                    "model.args.net_width=1536", "model.args.net_blocks=3", "train.lr=1e-3",
                                    "train.data_on_device=true"]
ANISO.add("wide_RAD_x")
ANISO_X.add("wide_RAD_x")
# and with the Wigner D-matrices D^2, D^3 of the joints next to the matrices (inputs="local+wigner", 1577 inputs
# instead of 171): the pair wide_RAD_x / wide_RAD_x_wigner differs by the input only. On c10, as the control of
# section 17 of the score report (the full data did not change that network).
CONFIGS["wide_RAD_x_wigner"] = CONFIGS["wide_RAD_x"] + ["model.args.inputs=local+wigner"]
ANISO.add("wide_RAD_x_wigner")
ANISO_X.add("wide_RAD_x_wigner")
# and with the two-body terms: the contracted products of the harmonics of the joints with the root and of the pairs
# of left and right and of siblings (inputs="local+wigner+pairs", 4067 inputs). The pair wide_RAD_x_wigner /
# wide_RAD_x_wigner_pairs differs by the pairs only. On the full data: the one-body harmonics overfit on c10.
CONFIGS["wide_RAD_x_wigner_pairs"] = CONFIGS["wide_RAD_x"] + ["model.args.inputs=local+wigner+pairs"]
ANISO.add("wide_RAD_x_wigner_pairs")
ANISO_X.add("wide_RAD_x_wigner_pairs")
# the control of the pairs with l = 1 only: the rotations G_j and G_i^T G_k of the same joints and pairs, without
# the higher harmonics (inputs="local+pairs", 441 inputs). The pair wide_RAD_x / wide_RAD_x_pairs_l1 differs by the
# relations of the joints only. On the full data.
CONFIGS["wide_RAD_x_pairs_l1"] = CONFIGS["wide_RAD_x"] + ["model.args.inputs=local+pairs"]
ANISO.add("wide_RAD_x_pairs_l1")
ANISO_X.add("wide_RAD_x_pairs_l1")
# The graph score network (net="mace", hp.models.graph_nets, docs/mace_lite_plan.md) with the noise and budget of
# wide_RAD_x: the joints as nodes, the tree and the pairs of local+pairs as edges with their relative rotations,
# 256 channels, 3 layers, dropout 0.1 (2.2M parameters). On the full data.
CONFIGS["mace_x"] = ABL_BASE + ["model.args.net=mace", "model.args.net_width=256", "model.args.net_blocks=3",
                                "model.args.dropout=0.1", "train.lr=1e-3", "train.data_on_device=true"]
ANISO.add("mace_x")
ANISO_X.add("mace_x")
# and with a virtual node connected to every joint (net_global): every joint sees the whole pose in one layer instead
# of one edge per layer (mace_x failed in completion, worst where the hidden joints are far from the visible ones)
CONFIGS["mace_g_x"] = CONFIGS["mace_x"] + ["model.args.net_global=true"]
ANISO.add("mace_g_x")
ANISO_X.add("mace_g_x")
# fact_RDW one step further along width, the strongest factor of the factorial for inpainting
CONFIGS["fact_RDW_w2048"] = CONFIGS["fact_RDW"] + ["model.args.net_width=2048"]
# The data-weighting study (explore/evaluation/data_weighting/rarity.py): the network of fact_RDW of the factorial
# (abl_base with residual, dropout 0.1 and width 1024, data on the device), the poses drawn in proportion to their
# rarity, softly (ESS 0.7 N) or strongly (ESS 0.4 N); the uniform baseline is fact_RDW_s10 of the factorial.
# --run-root ~/data/experiments/data_weighting
RDW = ABL_BASE + ["model.args.net_residual=true", "model.args.dropout=0.1", "model.args.net_width=1024",
                  "train.data_on_device=true"]
CONFIGS["rdw"] = list(RDW)
CONFIGS["rdw_soft"] = RDW + ["train.sample_weights=~/data/experiments/data_weighting/weights_soft.pt"]
CONFIGS["rdw_strong"] = RDW + ["train.sample_weights=~/data/experiments/data_weighting/weights_strong.pt"]
# the beta grid (rarity.py BETAS; soft and strong are beta 1.16 and 2.52)
for _beta in (0.6, 1.8, 3.3, 4.2):
    CONFIGS[f"rdw_b{_beta}"] = RDW + [f"train.sample_weights=~/data/experiments/data_weighting/weights_b{_beta}.pt"]
# ----------------------------------------------------------------------------------------


def run_name(data: str, rotations: str, config: str, seed: int) -> str:
    suffix = "" if config == "search" else "_" + config
    if data == "posedflow":
        return f"rsgm_posedflow{suffix}_s{seed}"
    folder = Path(rotations).expanduser().name.replace("AMASS_rotations", "").strip("_") or "full"
    return f"rsgm_amass_{folder}{suffix}_s{seed}"


def train_frames(rotations: str, split: str) -> int:
    """The number of training poses of the split, from the headers of the files where possible."""
    from hp.data.amass_cfg import ORIGINAL_AMASS_SPLITS, POSEDFLOW_AMASS_SPLITS, SUBSET_AMASS_SPLITS

    names = {"posedflow": POSEDFLOW_AMASS_SPLITS, "subset": SUBSET_AMASS_SPLITS}.get(split, ORIGINAL_AMASS_SPLITS)["train"]
    total = 0
    for name in names:
        path = Path(rotations).expanduser() / f"{name}.pt"
        try:
            total += torch.load(path, mmap=True, weights_only=True).shape[0]
        except RuntimeError:  # an old file format without mmap
            total += torch.load(path, weights_only=True).shape[0]
    return total


def fitted_z_statistics(rotations: str, split: str, run_root: str, frames: int) -> dict:
    """z_mean and z_std of inputs="local+z" on the training poses, from <run root>/rsgm_z_statistics.json or fitted
    now (and written there) when the file is missing or of other data."""
    path = Path(run_root).expanduser() / "rsgm_z_statistics.json"
    stats = json.loads(path.read_text()) if path.is_file() else None
    if stats is None or (stats["rotations"], stats["split"], stats["poses"]) != (rotations, split, frames) \
            or "z_cov" not in stats:
        train = load_reference("train", rotations, split)
        if stats is None or (stats["rotations"], stats["split"], stats["poses"]) != (rotations, split, frames):
            mean, std = z_statistics(train)
            stats = {"rotations": rotations, "split": split, "poses": len(train), "z_mean": mean, "z_std": std}
        stats["z_cov"] = joint_covariance(train, stats["z_mean"])  # added to a file from before it
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=1))
        print(f"z statistics fitted on {len(train):,} training poses, written to {path}", flush=True)
    return stats


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{seconds / 3600:.2f} h"


def runs_of(root: Path):
    return sorted(p for p in root.glob("run_[0-9][0-9][0-9][0-9]") if p.is_dir())


def finished(run: Path) -> bool:
    status = run / "status.json"
    return status.is_file() and json.loads(status.read_text()).get("finished", False) and (run / "weights.pt").is_file()


def resumable(run: Path) -> bool:
    return not finished(run) and any((c / "train_state.pt").is_file() for c in run.glob("checkpoint_*"))


def quick_check(run: Path, device: str, rotations: str, split: str, n: int, seed: int) -> None:
    """The EMA weights: valid rotations, and a finite log-likelihood of n valid poses."""
    model = load_model(run, device=device, weights="weights_ema")
    torch.manual_seed(seed)
    samples = model.sample(1000).cpu().double()
    orthogonality = (samples.transpose(-1, -2) @ samples - torch.eye(3, dtype=torch.float64)).abs().max()
    valid = load_reference("valid", rotations, split)
    poses = valid[torch.randperm(len(valid), generator=torch.Generator().manual_seed(seed))[:n]]
    if not model.has_log_prob:  # per-axis noise (joint_cov): no exact likelihood
        print(f"  check: max |R^T R - I| = {orthogonality:.1e} (no likelihood: the model has none)", flush=True)
        return
    with torch.no_grad():
        ll = torch.cat([model.log_prob(poses[i:i + 250].to(model.device)).cpu() for i in range(0, len(poses), 250)])
    print(f"  check: max |R^T R - I| = {orthogonality:.1e}, valid log-likelihood {ll.mean():.2f} +- "
          f"{ll.std() / len(ll) ** 0.5:.2f} (s.e.m., {len(ll)} poses), {(~torch.isfinite(ll)).sum()} non-finite", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", choices=sorted(CONFIGS), default=["search"],
                        help="one or more networks, trained one after the other")
    parser.add_argument("--data", choices=sorted(DATASETS), default="posedflow")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--rotations", default=None, help="the rotation folder (default: that of --data)")
    parser.add_argument("--run-root", default=RUN_ROOT)
    parser.add_argument("--retrain", action="store_true", help="train a new run even if a finished one exists")
    parser.add_argument("--check", type=int, default=N_CHECK, help="valid poses of the quick check (0: none)")
    parser.add_argument("--dry-run", action="store_true", help="print the configuration and the plan, train nothing")
    args, overrides = parser.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        parser.error(f"unknown arguments {bad}; settings of train_model.py are given as key=value")
    rotations = args.rotations or DATASETS[args.data]["rotations"]
    split = DATASETS[args.data]["split"]
    frames = train_frames(rotations, split)
    jobs = [(config, seed) for config in args.config for seed in args.seeds]
    print(f"{len(jobs)} run(s) on {rotations} (split {split}, {frames:,} training poses), device {args.device}", flush=True)

    status, summary, started = 0, [], time.perf_counter()
    for k, (config, seed) in enumerate(jobs, 1):
        name = run_name(args.data, rotations, config, seed)
        argv = ["model=rsgm", f"run_root={args.run_root}", f"run_name={name}", f"seed={seed}",
                f"data.rotations={rotations}", f"data.split={split}", "data.load_dtype=float32",
                f"train.device={args.device}", *COMMON, *CONFIGS[config], *overrides]
        if "model.args.inputs=local+z" in argv:
            stats = fitted_z_statistics(rotations, split, args.run_root, frames)
            argv += [f"model.args.z_mean={json.dumps(stats['z_mean'])}", f"model.args.z_std={json.dumps(stats['z_std'])}"]
        if config in ANISO:
            stats = fitted_z_statistics(rotations, split, args.run_root, frames)
            alpha = ALPHA.get(config, 1.0)
            argv += [f"model.args.joint_scale={json.dumps(fit_joint_scale(stats['z_std'], alpha))}"]
            if config in ANISO_X:
                argv += [f"model.args.joint_cov={json.dumps(joint_cov_power(stats['z_cov'], alpha))}"]
        runs = runs_of(Path(args.run_root).expanduser() / name)
        done = [r for r in runs if finished(r)]
        stopped = [r for r in runs if resumable(r)]
        head = f"[{k}/{len(jobs)}] {name}"
        if done and not args.retrain:
            print(f"{head}: finished already, {done[-1]} (--retrain trains a new run)", flush=True)
            summary.append((name, "finished already", None))
            continue
        resuming = bool(stopped) and not args.retrain
        cfg = load_config(argv)
        t = cfg.train
        args_ = cfg.model.args
        kind = args_.get("net", "mlp")
        net = (f"timeres {args_.net_width} x {args_.net_blocks} blocks" if kind == "timeres"
               else f"graph {args_.net_width} channels x {args_.net_blocks} layers" if kind == "mace"
               else f"hidden {list(args_.hidden)}")
        plan = (f"{net}, weighting {args_.weighting}, lr {t.lr}, {t.max_iterations:,} steps x batch {t.batch_size} = "
                f"{t.max_iterations * t.batch_size / frames:.1f} epochs")
        if resuming:
            # resuming takes the settings the run began with; only the resumable ones (e.g. the time limit) may change
            argv = [f"resume={stopped[-1]}", f"train.device={args.device}", *overrides]
            plan = f"resuming {stopped[-1]}; " + plan
        if args.dry_run:
            shown = [a.split("=")[0] + "=[...]" if a.startswith(("model.args.z_mean=", "model.args.z_std=",
                                                                  "model.args.joint_scale=", "model.args.joint_cov="))
                     else a for a in argv]
            print(f"{head}: {plan}\n  would run scripts/train_model.py {' '.join(shown)}", flush=True)
            continue
        print(f"\n{head}: {plan}; started {time.strftime('%H:%M')}, {minutes(time.perf_counter() - started)} into the script",
              flush=True)
        t0 = time.perf_counter()
        run = cli.main(argv)
        seconds = time.perf_counter() - t0
        times = run / "train_time.json"
        previous = json.loads(times.read_text())["seconds"] if times.is_file() else 0.0  # resumed: add up the sessions
        times.write_text(json.dumps({"seconds": previous + seconds, "device": args.device}))
        code = cli.exit_code(run)
        state = "finished" if code == 0 else "stopped early"
        print(f"{head}: {state} after {minutes(seconds)}, {run}; {len(jobs) - k} run(s) left", flush=True)
        summary.append((name, state, seconds))
        if code != 0:
            status = code
            continue
        if args.check:
            quick_check(run, args.device, rotations, split, args.check, seed)

    if not args.dry_run:
        print(f"\nall runs ({minutes(time.perf_counter() - started)}):", flush=True)
        for name, state, seconds in summary:
            print(f"  {name:40s} {state}{'' if seconds is None else ', ' + minutes(seconds)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
