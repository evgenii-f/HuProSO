# Motion denoising: RSGM against DPoser and PoseD-Flow (PoseD-Flow's protocol)

PoseD-Flow (CVPR 2026, Tab. 3) cleans the noisy 3D joints of a motion by fitting body poses to them under a pose prior.
Does RSGM, our score model on SO(3)^J, do this better than DPoser and PoseD-Flow on their protocol? Which way of
using the prior works for it: their D-Flow or DPoser's optimization (score distillation, SDS)?

## Summary

- **On the test, RSGM in DPoser's optimization ("ours", RSGM 1536x6 c10) has a lower MPJPE than DPoser at both noise
  levels, by a small margin.** 40 mm: 17.73 against 17.89 mm, a paired difference of −0.16 mm [−0.23, −0.09];
  RSGM is better on 65% of the 190 sequences. 100 mm: 28.59 against 29.10 mm, −0.52 mm [−0.65, −0.38], 73%.
  PoseD-Flow's checkpoint is behind both: 18.62 and 33.77 mm.
- **MPVPE is a tie with DPoser:** 30.84 against 30.74 mm and 41.07 against 40.92 mm. Both intervals contain 0.
- **At 100 mm, the gain comes from more steps.** The tuned settings use 3 x 133 steps, DPoser uses 3 x 80. At DPoser's
  number of steps, RSGM ties with DPoser: +0.10 mm [−0.06, +0.26]. At 40 mm (3 x 67 against 3 x 60), RSGM is better
  at DPoser's number of steps too: −0.19 mm.
- **The network hardly matters.** On the test, every isotropic RSGM at the tuned settings lies within 0.06 mm of
  "ours": 1536x6, 512x4 c10, 512x4 full, res_norm and abl_residual.
- **D-Flow does not suit the diffusion models.** RSGM via D-Flow scores 19.94 / 32.29 mm on the test, and DPoser via
  D-Flow 24.03 / 51.71. Score distillation is the way for RSGM.
- **The settings had to be tuned.** The hand-picked first guess, tested by mistake before the tuning, loses 3.9 mm to
  DPoser at 100 mm. Tuned on validation, the same model wins by 0.5 mm.
- **Decision.** For motion denoising we use RSGM with SDS at the tuned settings. Per-joint noise in training does not
  help here (validation; see [the per-joint noise report](2026-11-18_denoising_per_joint_noise.md)).

## Setup

### Protocol (`src/hp/evaluation/denoising.py`, `src/hp/models/motion_denoising.py`)

- **Test.** The 190 HumanEva chunks of PoseD-Flow and DPoser: AMASS SMPL-X stageii fits. The first and last 10% of
  each file are cut, the motion is resampled to 30 Hz by linspace, and it is chunked into 60 frames. Their scripts as
  published give 1002 chunks (240 without the 10% cut). This preprocessing, DPoser's, gives exactly the 190 of the
  paper. HumanEva is in their validation split, so no model trained on their training data has seen it.
- **Validation.** 100 chunks of HDM05, SFU and MoSh with the same preprocessing, drawn with seed 0. All settings were
  chosen there. Both splits are in `~/data/experiments/denoising/sequences.pt`.
- **Noise.** Gaussian, σ = 40 or 100 mm, on the 22 body joints of SMPL-X neutral (betas 0, root orientation identity),
  seeded by the sequence index. The input MPJPE is 63.8 / 159.5 mm on the test and 63.7 / 159.3 mm on validation.
- **Metrics (mm).** MPJPE of the 22 joints, the metric of their table, averaged over all frames. MPVPE of all
  vertices with the toes at rest is an addition and is not in the paper.
- **D-Flow.** As PoseD-Flow, with their settings. One ODE start per frame, Euler 100. The loss is the joint MSE, plus a
  temporal geodesic term (weight 1e-2 at 40 mm, 1e-1 at 100 mm), plus 1e-5 x the running cost. Their RiemannianAdam
  at lr 0.25, for 300 / 400 steps.
- **DPoser's optimization.** Rewritten from DPoser-X `motion_denoising.py`: Adam on the axis-angle poses with a data
  term, a temporal term and the prior term, for 3 iterations of `steps_per_iter` steps. DPoser's own settings are
  3 x 60 steps with t from 0.25 to 0.05 at 40 mm, and 3 x 80 steps with t from 0.25 to 0.1 at 100 mm, weight 1. For
  RSGM the prior term is score distillation on SO(3)^J (`dposer_denoise`): at every step the pose is noised by
  IGSO(3) at time t, denoised by one Tweedie step, and pulled toward the result with weight `sds_weight`.
- **"w/o prior".** The same optimization without the prior term.

### Models

| row | weights |
|---|---|
| PoseD-Flow | their checkpoint, `~/data/posedflow_ckpt/checkpoints/epoch-009_step-0_loss-0.0000.ckpt` |
| DPoser | ours, retrained on their full data, `~/data/experiments/full/dposer_posedflow_s10/run_0001` |
| RSGM 1536x6 c10 | `~/data/experiments/rsgm_posedflow_s10/run_0001` |
| RSGM 512x4 c10 | `~/data/experiments/rsgm_posedflow_small_s10/run_0001` |
| RSGM 512x4 full | `~/data/experiments/full/rsgm_posedflow_small_s10/run_0001` |
| RSGM res_norm c10 | `~/data/experiments/rsgm_posedflow_res_norm_s10/run_0001` |
| RSGM abl_residual c10 | `~/data/experiments/rsgm_posedflow_abl_residual_s10/run_0001` |

All runs use seed 10. "c10" is the 10% condition subset of their training data, and "full" is all of it.

### Tuning and the rule for the test

- **Studies.** Two Optuna studies (TPE, 40 trials each) tuned the SDS of RSGM 512x4 c10 on the 100 validation chunks,
  one per noise level: `config/denoising_tuning/rsgm_sds_{40,100}mm.yaml`, `scripts/tune_denoising.py`. The objective
  was MPJPE. The space was t_max, t_min as a fraction of t_max, sds_weight, lr, and the number of steps per
  iteration: 67 or 133 at 40 mm, and 80, 133 or 267 at 100 mm.
- **Best trials.**
  - 40 mm, trial 36, 18.69 mm: t_max 0.0595, t_min 0.387 t_max, weight 7.75, lr 0.0413, 3 x 67 steps.
  - 100 mm, trial 29, 30.87 mm: t_max 0.0515, t_min 0.881 t_max, weight 4.38, lr 0.0422, 3 x 133 steps.
- **The rule, written in the YAML files before the test.**
  - The best trial's settings are used for every RSGM row.
  - The RSGM row with the lowest validation MPJPE at these settings is "ours".
  - A second row reports RSGM 512x4 at DPoser's number of steps, with the other settings of the best trial.
  - The test is run once per row.
- **abl_residual.** It joined later, from the per-joint-noise line, as a declared extra row.
- **Untuned rows.** DPoser, PoseD-Flow and the D-Flow rows run with their authors' settings. Studies of DPoser
  by the same procedure (`config/denoising_tuning/dposer_sds_{40,100}mm.yaml`, written 2026-11-20 with their rule)
  are running; until they report, the DPoser rows here are untuned.
- **Machines.** The test ran on workstation1, cuda, in one run (`test2.log`). The validation rows ran on the
  servers and the Mac.

### The settings of every row

DPoser's optimization: Adam on the axis-angle poses, 3 iterations, the noise time t going down linearly from t_max
to t_min over all steps. t is the model's own time: RSGM's IGSO(3) time and DPoser's VP-SDE time are on different
scales, so their windows are not comparable. Validation and test MPJPE in mm; for the RSGM rows a range over the
isotropic networks.

| model | noise | t_max | t_min | sds_weight | lr | steps | chosen by | valid | test |
|---|---|---|---|---|---|---|---|---|---|
| RSGM, every isotropic network | 40 mm | 0.0595 | 0.0230 (0.387 t_max) | 7.75 | 0.0413 | 3 x 67 = 201 | `rsgm_sds_40mm`, trial 36, on 512x4 c10 | 18.57–18.73 | 17.68–17.74 |
| RSGM, every isotropic network | 100 mm | 0.0515 | 0.0453 (0.881 t_max) | 4.38 | 0.0422 | 3 x 133 = 399 | `rsgm_sds_100mm`, trial 29, on 512x4 c10 | 30.79–30.92 | 28.59–28.65 |
| RSGM 512x4 c10, DPoser's steps | 40 / 100 mm | as above | as above | as above | as above | 3 x 60 / 3 x 80 | the rule (equal budget) | 18.75 / 31.56 | 17.70 / 29.20 |
| RSGM, first guess | 40 / 100 mm | 0.05 | 0.01 | 10 | 0.03 | 3 x 60 / 3 x 80 | by hand, 10 sequences at 40 mm | 19.00 / 35.72 (1536x6) | 17.69 / 33.00 (1536x6) |
| RSGM abl_residual_aniso (per joint) | 40 mm | 0.0166 | 0.0064 | 7.75 | 0.0413 | 3 x 67 | "t geo" map of the isotropic set (`rsgm_sds_aniso_40mm` found nothing better) | 19.47 | – |
| RSGM abl_residual_aniso (per joint) | 100 mm | 0.0170 | 0.0067 (0.394 t_max) | 7.53 | 0.0378 | 3 x 133 | `rsgm_sds_aniso_100mm`, trial 22 | 32.68 | – |
| DPoser full | 40 mm | 0.25 | 0.05 | 1 | 0.03 | 3 x 60 = 180 | the authors (not tuned) | 19.19 | 17.89 |
| DPoser full | 100 mm | 0.25 | 0.10 | 1 | 0.03 | 3 x 80 = 240 | the authors (not tuned) | 32.23 | 29.10 |
| w/o prior | 40 / 100 mm | – | – | – (no prior term) | 0.03 | 3 x 60 / 3 x 80 | DPoser's settings without the prior term | 23.43 / 50.54 | 22.73 / 49.57 |

D-Flow: the ODE start of every frame is optimized through Euler 100, start mixing alpha 0.75, with RiemannianAdam
for the priors on SO(3)^J and Adam for DPoser's Euclidean latent.

| model | noise | lr | steps | temp_weight | running_weight | chosen by | valid | test |
|---|---|---|---|---|---|---|---|---|
| PoseD-Flow, their checkpoint | 40 mm | 0.25 | 300 | 1e-2 | 1e-5 | the authors | 19.83 | 18.62 |
| PoseD-Flow, their checkpoint | 100 mm | 0.25 | 400 | 1e-1 | 1e-5 | the authors | 31.73 | 33.77 |
| RSGM 512x4 c10 / DPoser full | 40 / 100 mm | as PoseD-Flow | as PoseD-Flow | as PoseD-Flow | as PoseD-Flow | PoseD-Flow's settings, not tuned | 31.81 / 33.53 and 24.22 / 50.78 | 19.94 / 32.29 and 24.03 / 51.71 |

What tuning changed for RSGM: a narrow, low window of noise (t about 0.05, where DPoser uses 0.25 to 0.05), a prior
weight 4-8 times DPoser's, and at 100 mm almost a single noise level (t_min 0.88 t_max) with more steps. The first
guess had a wider and lower window (0.05 to 0.01) and fewer steps. At 100 mm it lost 4.4 mm to the tuned set on the
test (33.00 against 28.59 mm). In the pictures of `motion_denoising.ipynb` its bodies also look less natural; that
is a visual impression, not a measurement.

## Results

### Validation (100 chunks), MPJPE / MPVPE, mm

| row | steps (40 / 100 mm) | 40 mm | 100 mm |
|---|---|---|---|
| w/o prior | 180 / 240 | 23.43 / 49.29 | 50.54 / 72.63 |
| PoseD-Flow, their checkpoint (D-Flow) | 300 / 400 | 19.83 / **37.19** | 31.73 / 56.89 |
| DPoser, its own method | 180 / 240 | 19.19 / 38.15 | 32.23 / 54.38 |
| DPoser via D-Flow | 300 / 400 | 24.22 / 78.47 | 50.78 / 77.19 |
| RSGM 512x4 c10, D-Flow | 300 / 400 | 31.81 / 53.45 | 33.53 / 57.07 |
| **RSGM 1536x6 c10, SDS ("ours")** \* | 201 / 399 | **18.57** / 37.62 | **30.79** / 51.22 |
| RSGM 512x4 c10, SDS | 201 / 399 | 18.71 / 38.19 | 30.87 / 51.43 |
| RSGM 512x4 full, SDS \* | 201 / 399 | 18.66 / 38.16 | 30.90 / 51.52 |
| RSGM res_norm c10, SDS \* | 201 / 399 | 18.73 / 37.99 | 30.92 / 51.21 |
| RSGM abl_residual c10, SDS | 201 / 399 | 18.63 / 37.97 | 30.81 / 51.26 |
| RSGM 512x4 c10, SDS, DPoser's steps | 180 / 240 | 18.75 / 38.06 | 31.56 / **51.20** |

\* These rows ran on the servers, and their cache files are not on the Mac. The numbers come from the summary printed
by `evaluate_denoising.py` there. The 512x4 c10 row above is the Mac run, cache_aniso; its server run gave 18.72 /
30.87.

The RSGM settings were tuned on these same chunks, so the RSGM rows are optimistic. They differ among themselves by
at most 0.16 mm.

### Test (190 HumanEva chunks), MPJPE / MPVPE, mm

| row | 40 mm | 100 mm |
|---|---|---|
| paper: w/o prior / VPoser / Pose-NDF | 24.19 / 23.42 / 22.13 | 51.48 / 49.10 / 46.10 |
| paper: DPoser / PoseRFM (their model) | 19.87 / 18.88 | 33.18 / 34.68 |
| w/o prior | 22.73 / 43.83 | 49.57 / 67.56 |
| PoseD-Flow, their checkpoint (D-Flow) | 18.62 / 31.76 | 33.77 / 55.04 |
| DPoser, its own method | 17.89 / **30.74** | 29.10 / **40.92** |
| DPoser via D-Flow | 24.03 / 79.64 | 51.71 / 79.80 |
| RSGM 512x4 c10, D-Flow | 19.94 / 32.92 | 32.29 / 50.26 |
| **RSGM 1536x6 c10, SDS ("ours")** | 17.73 / 30.84 | **28.59** / 41.07 |
| RSGM 512x4 c10, SDS | **17.68** / 31.12 | **28.59** / 41.39 |
| RSGM 512x4 full, SDS | **17.68** / 31.09 | 28.65 / 41.39 |
| RSGM res_norm c10, SDS | 17.73 / 30.87 | 28.63 / 41.32 |
| RSGM abl_residual c10, SDS | 17.74 / 31.18 | **28.59** / 41.40 |
| RSGM 512x4 c10, SDS, DPoser's steps | 17.70 / 31.07 | 29.20 / 41.53 |
| RSGM 1536x6 c10, SDS, first guess (see below) | 17.69 / 30.36 | 33.00 / 43.33 |

Our reruns are lower than the paper's numbers for every row we share with it: w/o prior by 1.5 / 1.9 mm, DPoser by
2.0 / 4.1 mm and PoseD-Flow by 0.3 / 0.9 mm. Only rows run on the same protocol here are compared. Likely causes are our SMPL-X fits of HumanEva and the
body model; their data may be the SMPL+H fits.

### Paired against DPoser on the test (per-sequence MPJPE, row − DPoser)

Each entry is the mean difference in mm, its 95% bootstrap interval over the 190 sequences, and the share of
sequences where the row is better.

| row | 40 mm | 100 mm |
|---|---|---|
| PoseD-Flow | +0.73 [+0.54, +0.91], 26% | +4.67 [+3.60, +5.74], 32% |
| **RSGM 1536x6, SDS ("ours")** | **−0.16 [−0.23, −0.09], 65%** | **−0.52 [−0.65, −0.38], 73%** |
| RSGM 512x4 c10, SDS | −0.21 [−0.28, −0.14], 65% | −0.51 [−0.65, −0.37], 75% |
| RSGM 512x4 full, SDS | −0.21 [−0.28, −0.14], 69% | −0.46 [−0.58, −0.33], 71% |
| RSGM res_norm, SDS | −0.16 [−0.23, −0.08], 66% | −0.47 [−0.61, −0.33], 71% |
| RSGM 512x4 c10, SDS, DPoser's steps | −0.19 [−0.26, −0.12], 63% | +0.10 [−0.06, +0.26], 53% |
| RSGM 512x4 c10, D-Flow | +2.05 [+1.64, +2.51], 18% | +3.19 [+2.42, +3.97], 28% |
| RSGM 1536x6, SDS, first guess | −0.19 [−0.24, −0.15], 76% | +3.90 [+3.72, +4.08], 0% |

The MPVPE of "ours" − DPoser is +0.10 [−0.07, +0.28] at 40 mm and +0.15 [−0.20, +0.50] at 100 mm. The other RSGM
rows are 0.1-0.6 mm worse than DPoser in MPVPE.

Against "ours", the other tuned RSGM rows differ by −0.05 to +0.06 mm in MPJPE. At 100 mm, RSGM 512x4 at DPoser's
number of steps is 0.61 mm [0.53, 0.68] worse than the same model at 3 x 133 steps. At 40 mm the two are equal:
+0.02 mm [−0.00, +0.04].

Per subject at 100 mm (S1 / S2 / S3), "ours" is 28.43 / 28.59 / 28.75 and DPoser is 28.97 / 29.22 / 29.15 mm. The gain
holds for each subject.

### Findings on the way

- **PoseD-Flow's optimizer.** Their RiemannianAdam (geoopt, on their `SO3Power`) keeps one second moment per joint,
  so a joint turns by about lr/√2 per step. Adam per coordinate takes steps up to √6 times larger at the same lr.
  With plain Adam at their lr of 0.25, PoseD-Flow gave 25.9 mm on 2 validation chunks; with their optimizer it gave
  18.5 mm. D-Flow now has `optimizer="riemannian"`, which the denoising rows use. The completion results of
  [2026-11-13](2026-11-13_pose_completion_rsgm_vs_dposer.md) used plain Adam for all models and were not re-run.
- **An MPS bug.** On the Mac (MPS, torch 2.13), the SMPL-X `BodyModel` returns wrong joints, off by up to 9 m, for
  more than 1024 poses in one call. With 1024 or fewer it matches the CPU to 4e-7. `Body` now runs in chunks of 512
  poses, and the metrics version was raised to 2 so that the cache is recomputed. CUDA is not affected.
- **The first-guess test, a mistake reported as such.** The test of the RSGM SDS rows was first run (2026-11-17)
  with the commit before the tuned settings. That commit used the hand settings: t from 0.05 to 0.01, weight 10,
  DPoser's steps. Those files are kept under the label "first guess", and the tuned rows were then tested once. The
  first guess is close at 40 mm (17.69 mm) and far behind at 100 mm (33.00 mm). It was not used for any choice.
- **RSGM via D-Flow, validation against test.** RSGM via D-Flow gives 31.81 mm on validation but 19.94 mm on the test
  at 40 mm. The reason is a tail: on 4 of the 100 validation chunks D-Flow diverges, to 122-166 mm. These are SFU
  yoga and HDM05 chunks. Without them the mean is 27.18 mm, and the median is 21.33 mm on validation against 19.22
  mm on the test. The 190 HumanEva chunks (walking, jogging, gestures and the like) have no such chunk; their
  maximum is 43 mm.
- **Per-joint noise.** Per-joint noise in training (aniso, per joint) is worse than isotropic noise on validation; see
  [the per-joint noise report](2026-11-18_denoising_per_joint_noise.md).
- **Per-axis noise (aniso_x).** A quick look (2026-11-20) used the first 30 validation chunks, the isotropic SDS
  settings with "t geo", and the per-axis model `fact_NA_x` from `experimental/aniso-models` at 2a8946c. It is worse
  still. MPJPE at 40 / 100 mm:

  | model | 40 mm | 100 mm |
  |---|---|---|
  | isotropic fact_N | 19.48 | 32.22 |
  | per-joint fact_NA | 20.20 | 34.23 |
  | per-axis fact_NA_x | 21.61 | 38.18 |
  | DPoser, same chunks | 19.78 | 33.14 |

  This was not a protocol run: it ran from a scratch overlay of that branch, and nothing was cached in the repository.

## Conclusions

- RSGM with score distillation is a slightly better prior than DPoser for motion denoising in MPJPE: 0.16 mm at 40 mm
  and 0.52 mm at 100 mm on the test. It ties in MPVPE. It beats PoseD-Flow clearly, by 0.9 and 5.2 mm. At 100 mm the
  gain needs about 1.7 times DPoser's steps (399 against 240); at equal steps it is a tie.
- PoseD-Flow's D-Flow is a method for flow models. For diffusion priors, RSGM and DPoser alike, SDS is much better,
  and D-Flow diverges on some validation chunks.
- The tuning is worth it: the noise window of SDS has to be narrow (t_max about 0.05), and at 100 mm it has to be
  almost a single level (t_min 0.88 t_max).
- Choice of network: any isotropic RSGM will do. The rule picked 1536x6, but 512x4 c10 is just as good on the test
  and much cheaper.

## What it does not show

- **One seed** of each model and of the optimization. The intervals cover the choice of sequences only. The
  differences among the RSGM networks (≤ 0.06 mm on the test) are below what a second seed would likely move.
- **The settings were tuned for RSGM and not for DPoser or PoseD-Flow,** which run with their authors' settings. A
  tuned DPoser, especially with more steps at 100 mm, might close the gap. DPoser's own steps at 100 mm are fewer.
- **The edge of the search space.** At 100 mm the best t_min fraction (0.88) is near the edge of its range (0.9).
  Few trials tried 3 x 133 steps at 40 mm or 3 x 267 at 100 mm. A wider search was not run, and any new search would
  need a new rule before a new test.
- **HPS,** the second dataset of their table, is not evaluated yet. The data has been downloaded.
- **Absolute numbers** are not comparable with the paper. Our reruns of their rows are 0.3-4.1 mm lower.
- **The validation table** mixes Mac and server runs. Three tuned validation rows, marked \*, are not in the Mac
  cache.

## Where things are

- **Code.**
  - `src/hp/evaluation/denoising.py`: data, noise, metrics, cache, and the soft stop and resume of runs.
  - `src/hp/models/motion_denoising.py`: D-Flow, DPoser's optimization, and SDS for RSGM.
  - `src/hp/models/dflow.py`: RiemannianAdam.
  - `src/hp/evaluation/denoising_results.py`: the loader, paired bootstrap and groups.
  - `src/hp/visualization/denoising_figure.py`.
- **Tests.** `tests/test_denoising.py`, `tests/test_denoising_results.py`.
- **Scripts.**
  - `scripts/evaluate_denoising.py`: the rows and the tuned settings `SDS`.
  - `scripts/tune_denoising.py`.
  - Configs: `config/denoising_tuning/rsgm_sds_{40,100}mm.yaml`.
- **Notebooks.**
  - `explore/evaluation/motion_denoising.ipynb`: the tables of this report, the paired tests, the groups, the error
    per frame, and pictures of sequences.
  - `explore/evaluation/denoising_optimization.ipynb`: the course of the studies.
- **Data.**
  - Sequences: `~/data/experiments/denoising/sequences.pt` (Mac and both servers).
  - Cache: `~/data/experiments/denoising/cache/<row>/<split>_<sigma>mm.pt`. All test rows are on the Mac, copied
    from workstation1. Per-joint rows are in `cache_aniso/`.
  - Studies: `~/data/experiments/denoising/optuna/rsgm_sds_{40,100}mm.{db,json}`.
  - Test log: `~/data/experiments/denoising/test2.log` on workstation1, copied to the Mac as `test2_workstation1.log`.
