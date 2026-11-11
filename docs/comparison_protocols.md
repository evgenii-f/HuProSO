# Comparing the models: data, protocols, models

A reference for ourselves, as of 2026-11-09, branch `dev`. It has no result numbers, only what is compared with what, how, and where the results are. For the numbers see the notebooks in the protocol table and [model_comparison_paper_protocol.md](model_comparison_paper_protocol.md).

## The main rule

A number depends on three things: the **training data**, the **test data** and the **protocol**. Only rows with the same test data and protocol can be compared. If the training data are the same too, the comparison is clean; if not, the difference mixes the model and the data shift.

## Data

| name | folder | split | train | valid | test | what it is |
|---|---|---|---|---|---|---|
| **our full** | `~/data/AMASS_rotations` | `original` | 14,779,964 | 1,320,126 | 117,679 | AMASS SMPL-X **G**, all frames, the official split (test: Transitions + SSM) |
| our cN | `~/data/AMASS_rotations_c10`, `_c30`, ... | `original` | about 1/N | | | our AMASS, every N-th frame; c10 has 1,482,799 training poses |
| **their full** | on the server `/home/evgenii/data/AMASS_rotations_posedflow`; sources on the Mac: `~/data/posedflow_amass` | `posedflow` | 18,611,047 | 860,080 | 54,200 | the AMASS of PoseD-Flow: **SMPL** bodies, the preprocessing of DPoser, 30 Hz, the first and last 10% of each sequence cut |
| their c10 | `~/data/AMASS_rotations_posedflow_c10` | `posedflow` | about 1.86M | every 100th | 54,200 | train every 10th frame, valid every 100th, test complete |
| the AMASS of the HuProSO3 authors | not available | | | | | most likely the SMPL-X **N** release: their checkpoint `id_0900` does not reproduce the paper's numbers on our data |

To fix: the comments in `src/hp/data/amass_cfg.py` swap valid and test (they say valid 117679, test 1320126). The numbers above are the right ones.

Joints: ours are 19 (`AmassJoints4Learning`, no toes). PoseD-Flow has 21; the toes are leaves of the tree, so its samples are cut to 19, and it has no density on 19 joints.

## Protocols

| | **Harness** | **HuProSO3 paper** | **PoseD-Flow paper** | **Occlusion** |
|---|---|---|---|---|
| code | `hp/evaluation/evaluate.py` | `hp/evaluation/huproso3_protocol.py` | `hp/evaluation/posedflow_protocol.py` | notebook, section 3 |
| metrics | Wasserstein, recall, precision (degrees per joint), log-likelihood | recall and precision (geodesic distance summed over the 19 joints, rad, Tab. 1; MPJPE in mm, Tab. 6), log-likelihood (Tab. 7) | FID (22 joints of SMPL neutral), FID own (our statistics), APD (cm), dNN | MPJPE (mm), MGEO (rad), conditional log p; for 1 sample, mean of 10, best of 10 |
| sizes | Wasserstein 2000; queries 2000; 20,000 samples; LL 5000; 3 repeats | 1k queries, pool of 100k, LL 5000, 3 repeats | 20 sets × 500, 95% CI | 1000 test poses, K = 10 (100 poses in the version with D-Flow) |
| yardstick | real train poses in place of a model | 100k real train poses | real train and test poses | the authors' flows `id_0384`, `id_0410` |
| the ± | evaluation noise over the repeats, not the spread between seeds | the same | CI over the sets | |

### Where each protocol is computed

| notebook | protocol | arena | cache |
|---|---|---|---|
| `explore/evaluation/rsgm_huproso_comparison.ipynb` | Harness + paired LL | our test, all frames; contrast: train c30 | `~/data/experiments/rsgm_vs_flow/` |
| `explore/evaluation/model_comparison_loglkhd_recall_precision.ipynb` | HuProSO3 | our test and train, all frames | `cache/model_comparison_loglkhd_recall_precision/` in the checkout (git-ignored) |
| `explore/evaluation/posedflow_vs_rsgm.ipynb` | Harness + PoseD-Flow | **their** test; contrast: 100k train | `~/data/experiments/posedflow_comparison/cache/` (with older names too: the settings count, not the name) |
| `explore/evaluation/conditional_flow_occlusion.ipynb` | Occlusion | our test, all frames | none |
| `explore/evaluation/conditional_flow_occlusion-3models.ipynb` | Occlusion + D-Flow | our test, 100 poses | none |

`scripts/evaluate_posedflow_comparison.py` computes Harness and PoseD-Flow on a server with the same settings and writes into the same cache. The constants of the script and the notebook must stay equal.

### What to watch for

- **FID** of our poses against their statistics is shifted: real poses of our AMASS score 0.022, their train poses 0.011. Models trained on another AMASS cannot be compared by FID alone.
- **dNN** here is on 19 joints against the 1.86M subset, so it is larger than in the paper's table, for all models alike.
- **The absolute numbers of the HuProSO3 paper** cannot be reached on our data: real train poses do not reach its test recall. Compare the models with each other.
- **LL** of RSGM, PoseRFM and PoseD-Flow comes from an ODE: 200 steps in the HuProSO3 and `rsgm_huproso` notebooks; the configs have 100 (about +0.3 nats). The flow's is exact.
- **The validation loss of RSGM** cannot be compared between `noise="cpu"` and `noise="device"`: other random numbers.
- **Checkpoint:** the HuProSO3 notebook takes the best by validation loss (`CHECKPOINT="best"`), `posedflow_vs_rsgm` the final weights unless a checkpoint is named.

## Models

All runs are in `~/data/experiments/`. Network: width × layers. Training: steps × batch.

### HuProSO3 (normalizing flow), SO(3)^19

| name | path | training data | training |
|---|---|---|---|
| flow full | `prior_full/run_0001` | our full | 20k × 2000, no EMA |
| flow full, seeds 11 and 12 | `huproso3_full_seed11`, `_seed12` | our full | the same |
| flow c30 | `prior_c30/id_9030`, copy in `so3_flow_c30/id_9030` | our c30 | |
| flow authors | `experiments/pretrained_models/id_0900`, copy in `so3_flow_authors/id_0900` | the authors' AMASS | their checkpoint |
| conditional flows | `experiments/pretrained_models/id_0384` (random masks), `id_0410` (all joints visible) | the authors' AMASS | their checkpoints |

There are also `prior_c10` ... `prior_c100` (data compression) and `id_0510` (not looked at).

### RSGM (ours)

| name | path | network | data | training |
|---|---|---|---|---|
| RSGM full | `rsgm` | 512×5 | our full | 100k × 512 |
| RSGM c30 | `rsgm_c30` | 512×5 | our c30 | 100k × 512 |
| t23 | `rsgm_tuned_t23` | 1024×3 | our full | 100k × 512, Optuna |
| t23 AdamW | `rsgm_tuned_r1t23_adamw` | 1024×3 | our full | the same, AdamW |
| t23 AdamW, GPU noise | `rsgm_tuned_r1t23_adamw_gpunoise` | 1024×3 | our full | the same, `noise=device` (branch `igso3_gpu`) |
| r2t12 | `rsgm_tuned_r2t12` | 1024×5 | our full | 100k × 512, Optuna |
| c10 1536×6 | `rsgm_amass_c10_s10` | 1536×6 | our c10 | 30k × 4096, rounded settings |
| c10 512×4 | `rsgm_amass_c10_small_s10` | 512×4 | our c10 | 30k × 4096, the same |
| conditional 3D | `rsgm_cond3d_t23_adamw` | 1024×3 | our full | 100k × 512, random masks |
| their c10 512×4 | `rsgm_posedflow_small_s10`, `_s11`, `_s12` | 512×4 | their c10 | 30k × 4096, 3 seeds |
| their c10 1536×6 | `rsgm_posedflow_s10` | 1536×6 | their c10 | 30k × 4096 |
| their c10 4096×7 | `rsgm_posedflow_large_s10` | 4096×7 | their c10 | 30k × 4096, lr 3e-4; best at 8k, then overfits |
| their full 512×4 | `full/rsgm_posedflow_small_s10` | 512×4 | their full | 30k × 4096 |
| their full 4096×7 soft | `full_large7_soft/rsgm_posedflow_large_s10` | 4096×7 | their full | 60k × 4096, lr 2e-4; best at 11k |

Rounded settings: silu, β_f 7.5, loss_t_min 0.002, lr 2e-3, AdamW wd 1e-5, EMA 0.999, step schedule. Taken from the plateau of the search on our AMASS, not tuned for either data.

### PoseRFM (our Riemannian flow matching)

| name | path | network | data | training |
|---|---|---|---|---|
| PoseRFM full | `poserfm` | 512×4 | our full | 50k × 4096, the settings of the PoseD-Flow paper |
| PoseRFM c10 | `poserfm-c10` | 512×4 | our c10 | 50k × 4096 |
| t24 @30k | `poserfm-full-t24/run_0001/checkpoint_030000` | 1024×6 | our full | Optuna; the run broke down at step 33.9k |

### PoseD-Flow (Nadar, Foti, Birdal, CVPR 2026)

| name | where | data | note |
|---|---|---|---|
| their checkpoint | `~/data/posedflow_ckpt/`, `PoseDFlowPretrained` | their full | 21 joints. Midpoint 100 (paper) or Euler 1000 raw (their script) |
| checkpoint + D-Flow | `dflow_21`, notebook `-3models` | their full | condition at inference; data shift |
| retrained | `posedflow_amass_s10`, class `PoseDFlow` | our full | their network, loss and settings, 50k × 4096, 19 joints |
| conditional | class `PoseDFlow` with `condition="3D"`, `scripts/train_posedflow_cond.py` | our full | **code in dev, no run yet** |

### DPoser (Lu et al., ICCV 2025), Euclidean, axis-angle

| name | path | data | training |
|---|---|---|---|
| their full | `full/dposer_posedflow_s10` | their full | 200k × 5120, RAdam (the paper's settings) |
| their c10, RSGM budget | `dposer_posedflow_rsgm_budget_s10` | their c10 | 30k × 4096 |

`dposer_posedflow_s10` at the top level is an unfinished attempt and is not compared.

## Samplers

How each model draws a pose, by default. NFE: network calls per sample. The step options are attributes of the model (`sample_steps`, `sampler`, ...), set by the overrides of a model in `hp.evaluation.pareto.MODELS`.

| model | noise | sampler | space of the steps | steps (NFE) by default | the end |
|---|---|---|---|---|---|
| HuProSO3 flow | uniform on SO(3)^19 | the inverse of the flow, one pass (exact sample and log p together) | SO(3) per joint, autoregressive over the joints | 1 pass | rotations by construction |
| RSGM | uniform on SO(3)^19 | Euler–Maruyama on the reverse SDE, t from 1 to `eps` = 2e-4 (as the JAX code) | geodesic steps in the frame of x (`exp_frame`) | 100 (100) | rotations by construction |
| PoseRFM | uniform on SO(3)^19 | geodesic midpoint on the flow ODE, t from 0 to 1 | the same | 100 (200) | rotations by construction |
| PoseD-Flow (their checkpoint, `PoseDFlowPretrained`; retrained, `PoseDFlow`) | exp(N(0, I)) per joint in axis-angle (a wrapped normal of 1 rad, their `SO3.random`) | their solver, `method` "midpoint" (paper) or "euler" (their script: Euler 1000, raw weights) | the ambient space R^(9J), no projection on the way | midpoint 100 (200) | `projx`: QR onto orthogonal matrices (see `docs/posedflow_cuda_qr_projection.md`) |
| DPoser | N(0, I) in normalized axis-angle | Euler–Maruyama on the reverse VP SDE, `sample_steps` times from 1 to `sample_eps` = 1e-3 (their sampler) | R^(3J), Euclidean | 1000 (1000) | the mean of the last step (no noise), denormalized, axis-angle to rotations |

The time grids are uniform, except RSGM with `schedule="edm"`.

### The RSGM samplers

`sampler` and `schedule` of `RSGM` (see the module doc of `hp/models/rsgm.py`):

| `sampler` | equation | order | NFE per step |
|---|---|---|---|
| `"euler_maruyama"` (default) | the reverse SDE | 1 (weak) | 1 |
| `"euler"` | the probability flow ODE, dx/dt = −β(t)/2 · score | 1 | 1 |
| `"heun"` | the same ODE; the velocities in the frame of x averaged (Runge–Kutta–Munthe-Kaas of order 2) | 2 | 2 |

`schedule`: `"uniform"` in t (default), or `"edm"` (Karras et al. 2022): uniform in σ^(1/ρ) for σ = √s(t), `schedule_rho` = 7, so the steps are dense where the noise is small. The ODE samplers are deterministic given the noise at t = 1. The log-likelihood does not depend on the sampler (it is the probability flow ODE with `likelihood_steps` midpoint steps), so the models with a new sampler are evaluated without it (`likelihood=False` in `MODELS`).

Found in `explore/evaluation/rsgm_samplers.ipynb` (RSGM 512×4 their c10, 10-200 NFE): the ODE beats Euler–Maruyama at every NFE, Euler on the uniform grid is best at 10-20 NFE, Heun with the EDM grid is closest to real poses from 50 NFE; the EDM grid hurts the first-order samplers; Euler–Maruyama contracts the samples (low APD).

### The sampler variants in the Pareto comparison

`hp.evaluation.pareto.MODELS` (`explore/evaluation/pareto_posedflow_vs_rsgm.ipynb`, `scripts/evaluate_pareto.py`):

| family | variants | NFE |
|---|---|---|
| RSGM 512×4 their c10 (3 seeds), 512×4 their full, 1536×6 their c10 | own (Euler–Maruyama 100); ODE Euler 50 (uniform); Heun 25 steps with the EDM grid | 100; 50; 50 |
| RSGM 512×4 their c10 | Euler–Maruyama 300, 1000 | 300, 1000 |
| PoseD-Flow, their checkpoint | midpoint 100 (paper), 50, 25, 10; Euler 50, 20 | 200, 100, 50, 20; 50, 20 |
| DPoser their full, their c10 | Euler–Maruyama 1000, 100, 50, 20 | 1000, 100, 50, 20 |
| HuProSO3 flow (our AMASS) | its inverse pass | 1 |

## What has been computed on which arena

From the cache files on the Mac (+ result there, − none, empty: not applicable).

| model | Harness, our test | HuProSO3 | Harness + PoseD-Flow, their test |
|---|---|---|---|
| flow full | + | + | |
| flow full, seed 11 | | | + (`HuProSO3_flow_our_AMASS`) |
| flow c30 | + | + | |
| flow authors `id_0900` | + | + | |
| RSGM c30 | + | + | |
| RSGM full | + | | |
| RSGM t23 | + | | + |
| RSGM t23 AdamW | + | + | |
| RSGM r2t12 | + | | |
| RSGM c10 1536×6 | − | + | |
| RSGM c10 512×4 | − | + | |
| PoseRFM full | + | + | |
| PoseRFM t24 @30k | + | + | |
| PoseD-Flow, their checkpoint | − | − | + |
| PoseD-Flow retrained | − | **−** (row in the notebook, no cache) | + |
| RSGM their c10/full | | | + |
| DPoser | | | + |

Occlusion: no cache, the results are in the outputs of the notebooks.

## Not in dev

Uncommitted: motion from the prior (`sequence_sampling.py`, `motion_from_pose_prior.ipynb`, branch `conditions`), tree features and `DPoser.log_prob_so3` (`rsgm-tree-features`). GPU noise: `experimental/igso3_gpu`, not merged.
