# Motion denoising: RSGM against DPoser and PoseD-Flow (PoseD-Flow's protocol)

PoseD-Flow (CVPR 2026, Tab. 3) cleans the noisy 3D joints of a motion by fitting body poses to them under a pose prior.
Does RSGM, our score model on SO(3)^J, do this better than DPoser and PoseD-Flow on their protocol? Which way of
using the prior works for it: their D-Flow or DPoser's optimization (score distillation, SDS)?

## Summary

- **Tuned alike, RSGM matches DPoser in MPJPE with half the steps, and beats it at the same steps.** DPoser was
  tuned afterwards by the same procedure as RSGM (section "2026-11-20: DPoser tuned the same way") and chose about
  twice RSGM's steps (399 against 201 at 40 mm, 801 against 399 at 100 mm). Against it, "ours" (RSGM 1536x6 c10 with
  SDS) differs by −0.02 mm [−0.08, +0.04] at 40 mm and −0.10 mm [−0.24, +0.03] at 100 mm: a tie. RSGM 1536x6 tuned
  again at DPoser's steps (section "2026-11-21: RSGM at the tuned DPoser's budget") is better than the tuned DPoser
  by 0.19 mm [0.15, 0.23] at 40 mm and 0.19 mm [0.07, 0.31] at 100 mm: 17.56 against 17.75 and 28.50 against 28.69 mm.
- **In MPVPE, "ours" is better than the tuned DPoser at 40 mm,** 30.84 against 31.49 mm, −0.65 mm [−0.83, −0.47]; at
  100 mm it is a tie, −0.09 mm [−0.39, +0.22]. The tuning of DPoser, on MPJPE alone, cost it 0.75 mm of MPVPE at 40 mm.
- **Against DPoser at its authors' settings, "ours" is better in MPJPE,** by a small margin: 17.73 against 17.89 mm,
  −0.16 mm [−0.23, −0.09], and 28.59 against 29.10 mm, −0.52 mm [−0.65, −0.38]; MPVPE is a tie. This was the
  comparison the test was planned for; it set tuned RSGM against untuned DPoser.
- **Both diffusion priors beat PoseD-Flow's checkpoint clearly:** 18.62 and 33.77 mm, 0.9 and 5.2 mm behind "ours".
- **The network hardly matters.** On the test, every isotropic RSGM at the tuned settings lies within 0.06 mm of
  "ours": 1536x6, 512x4 c10, 512x4 full, res_norm and abl_residual.
- **D-Flow does not suit the diffusion models.** RSGM via D-Flow scores 19.94 / 32.29 mm on the test, and DPoser via
  D-Flow 24.03 / 51.71. Score distillation is the way for both.
- **The settings had to be tuned, for both priors.** RSGM's hand-picked first guess loses 3.9 mm to DPoser at 100 mm;
  tuned, it wins. DPoser gains 0.14 / 0.41 mm from its tuning, mostly from more steps.
- **Decision.** For motion denoising we use RSGM with SDS at the tuned settings. The claim the test supports: "as
  accurate as a tuned DPoser at half the network calls, and 0.2 mm more accurate in MPJPE at the same calls"; in
  MPVPE RSGM is better or level at 40 mm and level at 100 mm. The margins are small (0.1-0.2 mm, one seed).
  Per-joint noise in training does not help here (validation; see
  [the per-joint noise report](2026-11-18_denoising_per_joint_noise.md)).

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
- **Untuned rows.** PoseD-Flow and the D-Flow rows run with their authors' settings. DPoser runs with its authors'
  settings and, since 2026-11-20, also with settings tuned like RSGM's (section "2026-11-20: DPoser tuned the same
  way").
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
| DPoser full, tuned | 40 mm | 0.224 | 0.079 (0.351 t_max) | 0.76 | 0.0407 | 3 x 133 = 399 | `dposer_sds_40mm`, trial 35 | 18.72 | 17.75 |
| DPoser full, tuned | 100 mm | 0.375 | 0.182 (0.485 t_max) | 0.15 | 0.0219 | 3 x 267 = 801 | `dposer_sds_100mm`, trial 8 | 31.60 | 28.69 |
| RSGM 1536x6, tuned DPoser's budget | 40 mm | 0.0769 | 0.0229 (0.298 t_max) | 3.42 | 0.0356 | 3 x 133 = 399 | `rsgm_sds_budget_40mm`, trial 25, on 512x4 c10 | 18.62 (512x4) | 17.56 |
| RSGM 1536x6, tuned DPoser's budget | 100 mm | 0.0515 | 0.0454 (0.881 t_max) | 4.38 | 0.0422 | 3 x 267 = 801 | `rsgm_sds_budget_100mm`, trial 0, on 512x4 c10 | 30.89 (512x4) | 28.50 |
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
| DPoser, its own method, tuned (best trials) | 399 / 801 | 18.72 / 38.98 | 31.60 / 55.06 |
| DPoser via D-Flow | 300 / 400 | 24.22 / 78.47 | 50.78 / 77.19 |
| RSGM 512x4 c10, D-Flow | 300 / 400 | 31.81 / 53.45 | 33.53 / 57.07 |
| **RSGM 1536x6 c10, SDS ("ours")** | 201 / 399 | **18.57** / 37.62 | **30.79** / 51.22 |
| RSGM 512x4 c10, SDS | 201 / 399 | 18.71 / 38.19 | 30.87 / 51.43 |
| RSGM 512x4 full, SDS | 201 / 399 | 18.66 / 38.16 | 30.90 / 51.52 |
| RSGM res_norm c10, SDS | 201 / 399 | 18.73 / 37.99 | 30.92 / 51.21 |
| RSGM abl_residual c10, SDS | 201 / 399 | 18.63 / 37.97 | 30.81 / 51.26 |
| RSGM 512x4 c10, SDS, DPoser's steps | 180 / 240 | 18.75 / 38.06 | 31.56 / **51.20** |

The tuned rows of 1536x6, 512x4 full and res_norm ran on the servers. Their files were copied to the Mac cache on
2026-11-21 and give the numbers above. The 512x4 c10 row above is the Mac run, in cache_aniso; its server run gave
18.72 / 30.87.

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
| DPoser, its own method, tuned | 17.75 / 31.49 | 28.69 / 41.16 |
| RSGM 1536x6 c10, SDS, tuned DPoser's budget | **17.56** / 31.20 | **28.50** / 41.28 |
| DPoser via D-Flow | 24.03 / 79.64 | 51.71 / 79.80 |
| RSGM 512x4 c10, D-Flow | 19.94 / 32.92 | 32.29 / 50.26 |
| **RSGM 1536x6 c10, SDS ("ours")** | 17.73 / 30.84 | **28.59** / 41.07 |
| RSGM 512x4 c10, SDS | **17.68** / 31.12 | **28.59** / 41.39 |
| RSGM 512x4 full, SDS | **17.68** / 31.09 | 28.65 / 41.39 |
| RSGM res_norm c10, SDS | 17.73 / 30.87 | 28.63 / 41.32 |
| RSGM abl_residual c10, SDS | 17.74 / 31.18 | **28.59** / 41.40 |
| RSGM 512x4 c10, SDS, DPoser's steps | 17.70 / 31.07 | 29.20 / 41.53 |
| RSGM 1536x6 c10, SDS, first guess (see below) | 17.69 / 30.36 | 33.00 / 43.33 |
| RSGM abl_residual lik c10, SDS (after the fact) | 17.71 / 31.20 | 28.71 / 41.56 |
| RSGM abl_residual ln m3 c10, SDS (after the fact) | 17.76 / 31.08 | **28.53** / 41.22 |
| RSGM abl_residual ln m4 c10, SDS (after the fact) | 17.85 / 31.06 | 28.62 / 41.31 |

The rows "after the fact" were not meant for the test: the networks trained with more weight on small noise (lik,
log-normal noise levels) lost to the isotropic rows on validation and were left out. They were run on the test on
2026-11-21 on workstation1 all the same, after the other rows. They were not used for any choice, and they fall in
the range of the other RSGM rows (17.71-17.85 and 28.53-28.71 mm).

Our reruns are lower than the paper's numbers for every row we share with it: w/o prior by 1.5 / 1.9 mm, DPoser by
2.0 / 4.1 mm and PoseD-Flow by 0.3 / 0.9 mm. Only rows run on the same protocol here are compared. Likely causes are our SMPL-X fits of HumanEva and the
body model; their data may be the SMPL+H fits.

### Paired against DPoser at its authors' settings on the test (per-sequence MPJPE, row − DPoser)

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

## 2026-11-20: DPoser tuned the same way

The test above sets RSGM at settings tuned on validation against DPoser at its authors' settings. To compare the
priors and not the tuning, DPoser was tuned by the same procedure.

- **Studies.** `config/denoising_tuning/dposer_sds_{40,100}mm.yaml`: DPoser full in its own optimization, 100
  validation chunks, 40 trials each (TPE), objective MPJPE, run on workstation1 (cuda). The space is that of the RSGM
  studies with DPoser's own scale of t (t_max 0.03-0.8, log), sds_weight 0.1-30 (log), lr 0.01-0.1, and the same
  numbers of steps plus DPoser's own (60, 67, 133 at 40 mm; 80, 133, 267 at 100 mm). Trial 0 is DPoser's settings,
  trial 1 the same at 3 x 133 steps.
- **The rule, written in the files before the runs.** The best trial forms one more row, "DPoser: their full, DPoser
  method, tuned", run once on the test. Both DPoser rows stay in the report; "ours" is not chosen again.
- **Best trials** (validation MPJPE / MPVPE, mm): 40 mm trial 35, 18.72 / 38.98 (DPoser's settings 19.19 / 38.18);
  100 mm trial 8, 31.60 / 55.06 (32.24 / 54.38). Trial 1, DPoser's settings at 3 x 133 steps, already gives 19.07 and
  31.89: most of the gain is the number of steps. The best trials keep DPoser's broad window (t from about 0.2-0.4 down
  to 0.08-0.18) and a weight below 1, and both chose the largest number of steps offered (3 x 133 and 3 x 267).

Test, paired over the 190 sequences, "ours" − tuned DPoser (mean difference in mm, 95% bootstrap interval, share of
sequences where "ours" is better):

| metric | 40 mm | 100 mm |
|---|---|---|
| MPJPE | −0.02 [−0.08, +0.04], 52% | −0.10 [−0.24, +0.03], 55% |
| MPVPE | −0.65 [−0.83, −0.47], 68% | −0.09 [−0.39, +0.22], 57% |
| steps (network calls) | 201 against 399 | 399 against 801 |

- The tuned DPoser gains 0.14 mm [0.11, 0.18] at 40 mm and 0.41 mm [0.31, 0.52] at 100 mm in MPJPE over its
  authors' settings, and loses 0.75 mm [0.60, 0.91] and 0.24 mm [−0.02, +0.50] in MPVPE.
- Every tuned isotropic RSGM row is within −0.07 to −0.01 mm of the tuned DPoser at 40 mm and −0.10 to −0.04 mm at
  100 mm in MPJPE; only 512x4 c10 and 512x4 full at 40 mm have intervals that exclude 0, by 0.00-0.01 mm.
- At DPoser's original steps (180 / 240) RSGM 512x4 gets 17.70 / 29.20 mm: as the tuned DPoser at 40 mm with less
  than half its steps, 0.51 mm [0.33, 0.69] behind it at 100 mm with 30% of its steps.
- Per subject (S1 / S2 / S3) the tuned DPoser is 17.84 / 17.85 / 17.57 and 28.52 / 28.74 / 28.82 mm; "ours" is
  17.82 / 17.66 / 17.69 and 28.43 / 28.59 / 28.75 mm.

## 2026-11-21: RSGM at the tuned DPoser's budget

The tuned DPoser uses twice RSGM's steps. The RSGM studies had tried those numbers of steps in only 2 of 40 trials
each, so whether RSGM gains from them was open.

- **Studies.** `config/denoising_tuning/rsgm_sds_budget_{40,100}mm.yaml`: RSGM 512x4 c10, steps fixed at the tuned
  DPoser's (3 x 133 at 40 mm, 3 x 267 at 100 mm), t_max, t_min, sds_weight and lr tuned again (lr from 0.005, t_min
  up to 0.95 t_max at 100 mm), 30 trials each, 100 validation chunks, objective MPJPE, workstation1. Trial 0: the
  tuned RSGM settings at twice the steps; trial 1: the same at half the lr.
- **The rule, written in the files before the runs.** The best trial forms the row "RSGM 1536x6: their c10, DPoser
  method, DPoser's tuned budget" (the network of "ours"), run once on the test and compared with the tuned DPoser.
  "ours" is not chosen again.
- **Best trials** (validation, RSGM 512x4 c10, MPJPE / MPVPE, mm): 40 mm trial 25, 18.62 / 39.54 (t_max 0.0769, t_min
  0.298 t_max, weight 3.42, lr 0.0356), against 18.69 / 38.20 at 3 x 67 steps; 100 mm trial 0, i.e. the earlier
  settings at twice the steps, 30.89 / 52.32, against 30.87 / 51.39 at 3 x 133. On validation the extra steps gave
  0.07 mm at 40 mm and nothing at 100 mm, at a cost of 1 mm of MPVPE.

Test, 190 sequences, MPJPE / MPVPE (mm), and paired differences of the budget row (mean, 95% interval, share of
sequences where the budget row is better):

| row | steps | 40 mm | 100 mm |
|---|---|---|---|
| RSGM 1536x6, tuned DPoser's budget | 399 / 801 | **17.56** / 31.20 | **28.50** / 41.28 |
| RSGM 1536x6 ("ours") | 201 / 399 | 17.73 / **30.84** | 28.59 / **41.07** |
| DPoser, tuned | 399 / 801 | 17.75 / 31.49 | 28.69 / 41.16 |
| budget row − tuned DPoser, MPJPE | | −0.19 [−0.23, −0.15], 75% | −0.19 [−0.31, −0.07], 61% |
| budget row − tuned DPoser, MPVPE | | −0.29 [−0.47, −0.10], 58% | +0.13 [−0.17, +0.43], 52% |
| budget row − "ours", MPJPE | | −0.17 [−0.23, −0.12], 68% | −0.09 [−0.16, −0.02], 59% |
| budget row − "ours", MPVPE | | +0.36 [+0.16, +0.57], 42% | +0.21 [+0.06, +0.37], 43% |

- At an equal budget RSGM is more accurate than the tuned DPoser in MPJPE at both noise levels, by 0.19 mm; in
  MPVPE it is better at 40 mm and level at 100 mm.
- Against "ours", the extra steps give 0.17 / 0.09 mm of MPJPE and cost 0.36 / 0.21 mm of MPVPE: the same trade
  as on validation. The test gain at 40 mm (0.17 mm) is larger than validation suggested (0.07 mm on 512x4).

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

- Tuned the same way, RSGM with score distillation is at least as accurate a prior as DPoser for motion denoising on
  this test: level in MPJPE with half the optimizer steps and network calls (differences of 0.02 and 0.10 mm), and
  0.19 mm better at the same steps. In MPVPE it is better at 40 mm (0.29-0.65 mm) and level at 100 mm. Against DPoser
  at its authors' settings, RSGM is 0.16 / 0.52 mm better; that gap was mostly DPoser's untuned step count.
- More steps buy RSGM a little MPJPE (0.1-0.2 mm) for a little MPVPE (0.2-0.4 mm). "Ours" at 201 / 399 steps is the
  cheaper point of that trade, and stays the row we report.
- Both priors beat PoseD-Flow clearly, by 0.9 and 5.1-5.2 mm.
- PoseD-Flow's D-Flow is a method for flow models. For diffusion priors, RSGM and DPoser alike, SDS is much better,
  and D-Flow diverges on some validation chunks.
- The tuning is worth it for both priors. For RSGM the noise window of SDS has to be narrow (t_max about 0.05-0.08),
  and at 100 mm almost a single level (t_min 0.88 t_max). For DPoser the gain is mostly more steps.
- Choice of network: any isotropic RSGM will do. The rule picked 1536x6, but 512x4 c10 is just as good on the test
  and much cheaper.
- The line is closed for now. Open for later: HPS, the second dataset of their table; a joint MPJPE + MPVPE objective;
  a second seed; guided sampling (DPS-like) with RSGM instead of SDS.

## What it does not show

- **One seed** of each model and of the optimization. The intervals cover the choice of sequences only. The
  differences among the RSGM networks (≤ 0.06 mm on the test) are below what a second seed would likely move.
- **Budgets compared at the tuned DPoser's choice only.** RSGM was tuned again at DPoser's steps, but DPoser was not
  tuned at RSGM's (201 / 399). That DPoser at RSGM's steps would be worse than at its own is likely (its own
  settings at 3 x 133 already gain over 3 x 60 / 3 x 80) but not measured. PoseD-Flow runs with its authors' settings.
- **The objective was MPJPE only.** The tuned DPoser traded MPVPE for MPJPE at 40 mm; a joint objective might move both
  priors.
- **The edge of the search space.** At 100 mm the best t_min fraction of the first study (0.88) was near the edge of
  its range (0.9); the budget study widened it to 0.95 and kept 0.88.
- **The settings were tuned on RSGM 512x4 c10** and used for 1536x6. Tuned on 1536x6 itself, it might gain a little.
- **HPS,** the second dataset of their table, is not evaluated yet. The data has been downloaded.
- **Absolute numbers** are not comparable with the paper. Our reruns of their rows are 0.3-4.1 mm lower.
- **The validation table** mixes Mac and server runs. The validation numbers of the first guess (19.00 / 35.72 mm
  for 1536x6) are no longer in a file: the tuned runs replaced them in the cache.

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
  - Configs: `config/denoising_tuning/rsgm_sds_{40,100}mm.yaml`, `dposer_sds_{40,100}mm.yaml`,
    `rsgm_sds_budget_{40,100}mm.yaml`; the settings are `SDS`, `DPOSER_TUNED` and `SDS_BUDGET` in
    `scripts/evaluate_denoising.py`.
- **Notebooks.**
  - `explore/evaluation/motion_denoising.ipynb`: the tables of this report, the paired tests, the groups, the error
    per frame, and pictures of sequences.
  - `explore/evaluation/denoising_optimization.ipynb`: the course of the studies.
- **Data.**
  - Sequences: `~/data/experiments/denoising/sequences.pt` (Mac and both servers).
  - Cache: `~/data/experiments/denoising/cache/<row>/<split>_<sigma>mm.pt`. All test rows are on the Mac, copied
    from workstation1. Per-joint rows are in `cache_aniso/`.
  - Studies: `~/data/experiments/denoising/optuna/{rsgm,dposer,rsgm_budget}_sds_{40,100}mm.{db,json}` (the budget
    ones are `rsgm_sds_budget_*`; Mac and workstation1).
  - Test logs on workstation1: `test2.log` (the tuned RSGM rows; on the Mac as `test2_workstation1.log`),
    `test_abl_rest.log` (lik / ln), `test_dposer_tuned.log` and `test_rsgm_budget.log`, in
    `~/data/experiments/denoising/`.
