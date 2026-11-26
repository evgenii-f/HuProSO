# Pose completion: RSGM against PoseD-Flow and DPoser

What we did between 2026-11-13 and 2026-11-15 on branch `experimental/completion`: RSGM as the prior of pose completion
in the protocol of PoseD-Flow, two completion procedures (D-Flow and DPoser's score distillation), why RSGM loses to
DPoser in DPoser's procedure, and whether training RSGM with the likelihood weighting or a better ODE solver closes the
gap.

## Summary

All numbers on the 5000 test poses of PoseD-Flow, K = 10, MPVPE in mm averaged over the four occlusions.

**Final result (2026-11-20, sections 15-16).** Six fronts (family × completion method, all networks of a family),
10 points each chosen on validation by a rule fixed before the test, each run once on the test, compared by the
hypervolume over APD 4.0-8.5 cm (band mean in mm, lower is better):

- **DPoser's own optimization is the best front** in the mean and in best of 10: band mean 86.85 mm, then RSGM
  inpainting 90.05, DPoser inpainting 91.71, RSGM D-Flow 92.91, PoseD-Flow D-Flow 93.28, RSGM with DPoser's
  optimization 96.69. The order is the same on validation.
- **Under the same procedure the two priors are on par.** Inpainting: RSGM ahead on the test (90.05 against 91.71),
  DPoser on validation (91.10 against 92.40). D-Flow: RSGM against PoseD-Flow equal in the mean within the choice of
  the band, clearly ahead in best of 10 (HV 71.64 against 59.11).
- **The gap between RSGM and DPoser is the procedure, not the prior**: DPoser's optimization works on DPoser and is
  the weakest front on RSGM.
- **Hard poses (top 5% for all seven compared rows)** have 2-2.5 times the average error; which poses are hard depends
  mostly on the procedure; on them the diverse rows (DPoser t28, RSGM inpainting t108) give the best best of 10.

- **The training data covers the held-out poses as well at c10 as in full (section 17)**: the nearest training
  pose of a validation or test pose is 1-13% closer in the full split; the hard poses are 1.7-2.3 times farther from
  the data than the rest in both, so reweighting the training poses cannot help them. (The last part is corrected on
  2026-11-25, see section 17.)

- **State on 2026-11-26 (section 18).** Base RSGM: `fact_RDW` β 0, one network for every protocol. Its fronts on
  the test: inpainting 90.05 mm, D-Flow 91.53 (DPoser's optimization 86.85, PoseD-Flow 93.28). By APD, DPoser's
  optimization and PoseD-Flow lead below 6 cm, RSGM inpainting at 6-6.5 cm (84.61 mm), DPoser's optimization from
  6.5 to 8 cm; one training seed per network.

The dated bullets below are the history that led there.

- **Best RSGM row: 512×4 with D-Flow and Heun EDM inside, 50 network calls**: best of 10 49.92 (PoseD-Flow 50.26,
  DPoser 47.22), the most diverse hypotheses of all rows, a quarter of PoseD-Flow's calls; mean 88.32 against 79.65
  (PoseD-Flow) and 81.19 (DPoser). More calls (100, 200) change nothing.
- **D-Flow's lr and running weight are at their optimum** with PoseD-Flow's defaults (Optuna on validation: 90.55 vs
  91.40 mm).
- **SDS (DPoser's score distillation) on RSGM** loses to DPoser mainly through an inaccurate score at small noise
  (the hybrid diagnostic); training with the likelihood weighting improves it from 95.92 to 92.27 on the test, still
  11 mm behind DPoser.
- **The likelihood weighting hurts D-Flow** by 4.6-5.8 mm in the mean (1536×6): it is a choice per procedure, keep
  "std" for D-Flow.
- **The remaining mean gap is the model**, not the procedure or the solver; next: the residual + LayerNorm network of
  the architecture ablation, with "std", in D-Flow with Heun EDM 50.

- **2026-11-15, inpainting (validation, section 10)**: completing by the whole reverse process of RSGM, with DPS
  guidance and RePaint resampling, beats D-Flow on all three metrics at once (100 validation poses: 86.68 / 48.19 mm,
  APD 6.42 cm against 90.88 / 52.40 / 6.08), 1.4 times faster on the Mac (on the test, with the noise on the GPU,
  about 10 times: section 18); 3.4 mm behind DPoser in the mean, ahead in best of 10
  and APD. **The gain is the procedure, not the prior**: the same inpainting on DPoser c10 gives 87.60 / 47.52 / 6.61,
  on par with RSGM. The claim of the bullet above ("the remaining gap is the model") is thereby revised: at an equal
  procedure the two priors are equal; DPoser's own optimization is still the best mean and 15 times faster.

- **2026-11-16, inpainting on the test (section 11)**: the four points chosen on validation by the rule of 10.4, run
  once on the 5000 test poses. Best: the residual network of the ablation, 85.50 / 46.70 mm, APD 6.45 cm, the lowest
  best of 10 of all rows (DPoser 47.22, PoseD-Flow 50.26) and 2.8 mm better than D-Flow in the mean; still 4.3 mm
  behind DPoser and 5.9 mm behind PoseD-Flow in the mean, mostly on the arms.
- **2026-11-16, fairness check (validation, section 12)**: section 10 favoured RSGM (Optuna against a hand grid,
  twice the network calls). Tuned the same way (120 trials, 1000 validation poses), DPoser c10 inpainting reaches the
  front of RSGM (88.26 / 49.22 mm / 6.46 cm against 88.22 / 49.30 / 6.39); DPoser's own optimization stays about
  2 mm lower in the mean at a lower APD (86.17 / 49.99 / 6.02 with 3 × 200 steps). On the test the chosen points
  confirm it: DPoser c10 inpainting 85.94 / 47.90 / 6.19 against RSGM residual 85.50 / 46.70 / 6.45, its own method
  (3 × 200 steps) 84.10 / 48.59 / 5.80.
- **2026-11-16, fronts over bands of APD (validation, section 13)**: averaged over APD bands from 6.0 cm up, the
  inpainting front of DPoser c10 is 1.2-1.9 mm lower in the mean than that of the best RSGM network (abl residual);
  at the threshold of the rule (6.4) they meet, hence the equal test points. Among the RSGM networks the order is
  residual ≲ norm < deep ≈ 1536×6 lik full; the likelihood weighting brings inpainting nothing.
- **2026-11-19, PoseD-Flow with a tuned D-Flow (test, section 14)**: trial 17 of the PoseD-Flow study (lr 0.05),
  better than their protocol on validation in all three metrics, is not on the test: 81.40 / 49.13 mm / 5.10 cm
  against 79.65 / 50.26 / 4.61, worse in the mean by 2-3 mm pose by pose on legs and arms. Not the batch (checked);
  by difficulty, its extra spread costs mean on easy poses and pays in best of 10 on ambiguous ones, and the
  validation poses are harder (a hypothesis). Their protocol stays the PoseD-Flow row.

## 1. Protocol

The test protocol of PoseD-Flow (their Tab. 2): 5000 test poses of their AMASS split (`PoseDFlow_test.pt` in
`~/data/AMASS_rotations_posedflow_c10`), four occlusions (`left_leg`, `legs`, `arms`, `trunk`: the joints hidden), K = 10
hypotheses per pose. Metrics, in mm on the SMPL mesh:

- **MPVPE mean**: the vertex error averaged over the 10 hypotheses (lower is better);
- **APD**: the mean distance between the hypotheses (cm; diversity, higher is more diverse);
- **best of 10**: the error of the best hypothesis per pose (lower is better: one of the hypotheses is right).

The test is run once per method with settings chosen before it; everything tuned is tuned on 1000 fixed validation
poses (`valid_poses(n=1000)`, `PoseDFlow_valid.pt`). Completions and metrics are cached per row in
`~/data/experiments/completion/cache/<row>/` (`scripts/evaluate_completion.py`, `hp.evaluation.completion`).

## 2. Two completion procedures

**D-Flow** (PoseD-Flow's method, `hp.models.dflow`): optimize the start noise x0 of the sampling ODE so that the end
pose matches the visible joints. 200 Adam steps (300 for arms), lr 0.1, loss = geodesic error on the visible joints +
1e-3 · running cost along the ODE; the start is the ODE run forward from the observation (their heuristic), then
mixed with noise; at the end the visible joints are replaced by the observation. Every Adam step backpropagates through
the whole ODE (100 steps by default): expensive, but it uses the prior at every noise level.

**Score distillation, SDS** (DPoser's own method, DPoser-X `DPoserComp.optimize`, `hp.models.score_completion`):
optimize the pose itself. 2 × 100 Adam steps, t linear from 0.15 to 0.05; each step noises the current pose, denoises
it in one step (Tweedie) without gradient, and pulls the pose toward the denoised one, next to the data term on the
visible joints. One forward pass per step, no backpropagation through the network, and only small noise levels.

- `dposer_complete`: DPoser's procedure one to one (z-scored axis-angle). Our reproduction of DPoser on the test:
  76.3 / 102.9 / 102.2 / 43.4 against 78.3 / 102.5 / 104.9 / 44.6 in the paper (MPVPE mean, the four occlusions).
- `rsgm_complete`: the same on SO(3)^19 for RSGM: pose = start · exp(scale · u), IGSO(3) noise of Brownian time s(t),
  denoising x0_hat = x_t exp(s · score), squared geodesic angles in both terms.

## 3. Starting point (test)

| method | left_leg | legs | arms | trunk | avg mean / best of 10 |
|---|---|---|---|---|---|
| PoseD-Flow (their checkpoint, midpoint 100 = 200 NFE) | 79.8 / 5.15 / 47.2 | 92.4 / 6.60 / 59.6 | 106.9 / 5.63 / 66.7 | 39.5 / 1.07 / 27.6 | 79.65 / 50.26 |
| DPoser (their full, own method) | 76.3 / 6.46 / 38.3 | 102.9 / 7.46 / 65.0 | 102.2 / 5.64 / 62.5 | 43.4 / 1.87 / 23.1 | 81.19 / 47.22 |
| RSGM 512×4 c10, D-Flow (mean of seeds 10, 11, 12) | 93.3 / 6.31 / 51.8 | 104.0 / 7.35 / 66.4 | 108.8 / 5.35 / 71.9 | 43.1 / 1.17 / 30.1 | 87.31 / 55.04 |
| RSGM 512×4 full, D-Flow | 92.8 / 6.24 / 51.6 | 103.9 / 7.33 / 66.5 | 112.0 / 5.27 / 74.8 | 42.7 / 1.17 / 29.9 | 87.85 / 55.71 |
| RSGM 1536×6 c10, D-Flow | 95.1 / 6.37 / 52.1 | 106.1 / 7.48 / 66.8 | 114.5 / 6.14 / 70.6 | 41.5 / 1.24 / 28.2 | 89.31 / 54.43 |
| RSGM 512×4 c10, SDS (validation trial 5) | 91.1 / 4.47 / 60.0 | 120.9 / 5.00 / 94.6 | 129.2 / 3.17 / 103.5 | 42.5 / 0.75 / 32.5 | 95.92 / 72.65 |

Cells: MPVPE mean / APD / best of 10. RSGM with D-Flow is about 8 mm behind PoseD-Flow in the mean; with SDS it is
about 15 mm behind DPoser, with a collapsed diversity. Conditional RSGM / HuProSO sampled directly from p(pose | visible
joints) were 50-70 mm worse than D-Flow (not in the table).

## 4. Tuning SDS and D-Flow on validation (Optuna)

`scripts/tune_completion.py` with a YAML file per study (`config/completion_tuning/`), each study in its own SQLite file
in `~/data/experiments/completion/optuna/`; all on RSGM 512×4 c10 std, 1000 validation poses, the same seeds in every
trial.

- **`rsgm_sds_wide`** (SDS, objective: mean MPVPE, 80 trials, file `completion_dposer_rsgm_posedflow_small_s10_v1000.db`):
  a plateau at 94.45 mm (15 best within 0.3 mm), all with a noise start, scale std and a narrow noise window. Trial 5:
  t_max 0.0319, t_min = 0.592 · t_max, lr 0.0478, prior_weight 0.0907. The mean objective rewards hypotheses that agree
  (APD collapses), hence:
- **`rsgm_sds_pareto`** (SDS, mean and best of 10, 60 trials): 19 points on the front, mean 93.11 → 100.54 while best
  of 10 goes 70.56 → 58.31. All points use 200 steps per iteration; the diversity is set by lr (0.05 → 0.2), not by the
  noise window (t_max 0.027-0.05 on the whole front; t_max > 0.08 gave at best 94.76 / 70.64).
- **D-Flow** (`completion_dflow_rsgm_posedflow_small_s10_v1000`, mean, 25 trials): the best is trial 7, lr 0.082,
  running_weight 0.00113, **90.55 / 56.08**; PoseD-Flow's defaults (lr 0.1, 1e-3) give 91.40 / 56.41. The running
  weight has a narrow good window around 1e-3 (< 3e-4 or > 7e-3: 100-122 mm); the inverted start never helps (best
  94.27). Their defaults are at the optimum.

Rule fixed before the test (in `rsgm_sds_pareto.yaml`): among the front points not worse than D-Flow in best of 10, the
lowest mean; if none, the lowest mean of the front. No SDS point reaches 56.08, and D-Flow dominates the whole SDS
front (90.55 < 93.11 and 56.08 < 58.31). By the rule: trial 1 of the Pareto study (trial 5 with 200 steps).

## 5. Why SDS on RSGM loses to DPoser: diagnostics

Scripts in `explore/evaluation/completion_diagnostics/` (validation poses):

1. **The IGSO(3) heat kernel** of RSGM is exact (checked against the series).
2. **One-step denoising** (`denoise_dposer_vs_rsgm.py`): at matched noise RSGM's Tweedie step denoises as well as
   DPoser's or better. Not the bottleneck.
3. **The hybrid** (`hybrid.py`): DPoser's procedure one to one (its variable, losses, schedule, lr), with only the
   one-step denoiser swapped for RSGM's (IGSO(3) noise of the same mean angle, RSGM's Tweedie step, back to DPoser's
   coordinates). The control with DPoser's own denoiser reproduces DPoser. The gap control → hybrid is the model; the
   gap hybrid → our SDS is the procedure.

   | denoiser (500 validation poses) | left_leg | legs | arms | trunk |
   |---|---|---|---|---|
   | DPoser (control) | 79.4 / 6.31 / 39.7 | 103.3 / 7.16 / 66.2 | 110.6 / 5.61 / 74.1 | 43.5 / 1.96 / 24.0 |
   | RSGM 512×4 c10 std | 123.7 / 10.15 / 60.3 | 143.7 / 11.74 / 85.3 | 137.1 / 8.03 / 88.1 | 49.1 / 2.30 / 25.9 |

4. **Trajectories** (`trajectory.py`, the hidden-joint error in degrees along the 200 steps): in the first phase RSGM
   converges about 3° worse; in the second phase (the smallest noise, t_RSGM ≈ 0.011 → 0.003) RSGM drifts away
   (left_leg 13.8° → 17.0°, the spread of the hypotheses almost doubles) while DPoser stays (10.8° → 11.5°).

Hypothesis: the score of RSGM is inaccurate at the bottom of its training range. With weighting "std" the loss times
are uniform on [0.002, 1], about 2.3% of training at t ≤ 0.025, where SDS works.

## 6. Training with the likelihood weighting

`model.args.weighting=likelihood` draws the loss times so that about 33.6% of training is at t ≤ 0.025 (52.5% at t ≤
0.1). Four runs (`scripts/train_rsgm_posedflow.py`, configs `small_lik` and `search_lik`, 30k steps, batch 4096, seed
10) on workstation2: 512×4 and 1536×6, on c10 (1.86M poses, 66 epochs) and on all frames (18.6M, 6.6 epochs).
Runs: `rsgm_posedflow_{small,search}_lik_s10`, `full/rsgm_posedflow_{small,search}_lik_s10`.

**Validation loss rises after a minimum**, also on all frames:

| run | min (step) | final (30k) |
|---|---|---|
| 512×4 lik c10 | 25.31 (10k) | 25.60 |
| 1536×6 lik c10 | 25.30 (5k) | 26.52 |
| 512×4 lik full | 25.28 (11k) | 25.65 |
| 1536×6 lik full | 25.22 (5k) | 25.99 |

**One-step denoising does not improve** (`denoise_weighting.py`, error after / noise before, 2000 poses; σ = 0.01,
0.02, 0.05, 0.1, 0.2 rad per coordinate): 512×4 std 0.988 / 0.946 / 0.828 / 0.689 / 0.522, 512×4 lik 1.005 / 0.960 /
0.833 / 0.690 / 0.527, 1536×6 lik full 1.009 / 0.971 / 0.848 / 0.702 / 0.529.

**But the iterated use improves a lot** (hybrid, 500 validation poses):

| denoiser | left_leg | legs | arms | trunk |
|---|---|---|---|---|
| RSGM 512×4 std c10 | 123.7 | 143.7 | 137.1 | 49.1 |
| RSGM 512×4 lik c10 | 108.0 | 132.6 | 128.3 | 46.4 |
| RSGM 512×4 lik full | 104.1 | 130.1 | 127.6 | 47.7 |
| RSGM 1536×6 lik c10 | 89.6 | 112.8 | 123.3 | 45.6 |
| RSGM 1536×6 lik full | 89.2 | 113.5 | 121.5 | 45.4 |
| DPoser (control) | 79.4 | 103.3 | 110.6 | 43.5 |

Trajectories (final hidden-joint error, left_leg / arms): 512×4 std 16.98° / 29.11°, 512×4 lik 14.92° / 26.58°,
1536×6 lik full 12.54° / 24.25° (stable from step 100 on), DPoser 11.52° / 22.13°. The drift of the second phase is
gone with the large network; the weighting needs the capacity, the amount of data hardly matters.

**SDS, validation** (`sds_valid.py`, settings of trial 5, mean over the occlusions of MPVPE mean / best of 10): 512×4
std 94.45 / 71.85, 512×4 lik c10 93.02 / 70.78, its 10k checkpoint 91.34 / 69.75, 1536×6 std c10 93.15 / 71.12,
1536×6 lik c10 91.19 / 70.00, 512×4 lik full 92.48 / 70.76, **1536×6 lik full 90.45 / 68.73**. A wider noise window
(t_max 0.08) is not better.

**SDS, test** (`sds_test.py`, trial 5, fixed before):

| model | left_leg | legs | arms | trunk | avg mean |
|---|---|---|---|---|---|
| 512×4 std c10 | 91.1 / 4.47 / 60.0 | 120.9 / 5.00 / 94.6 | 129.2 / 3.17 / 103.5 | 42.5 / 0.75 / 32.5 | 95.92 |
| 1536×6 std c10 | 90.7 / 4.19 / 61.1 | 121.1 / 4.77 / 95.0 | 124.4 / 3.34 / 97.4 | 41.6 / 0.68 / 32.5 | 94.46 |
| 512×4 lik c10 | 86.3 / 4.30 / 57.0 | 119.1 / 4.82 / 92.7 | 126.6 / 3.27 / 99.4 | 43.0 / 0.74 / 33.2 | 93.73 |
| 512×4 lik c10, 10k checkpoint | 86.5 / 4.26 / 57.1 | 119.3 / 4.78 / 93.6 | 128.9 / 2.92 / 105.1 | 43.9 / 0.78 / 33.3 | 94.64 |
| 512×4 lik full | 90.9 / 4.26 / 61.3 | 120.9 / 4.75 / 95.6 | 126.3 / 3.29 / 98.8 | 41.8 / 0.73 / 32.1 | 94.99 |
| 1536×6 lik c10 | 87.7 / 4.09 / 58.3 | 119.6 / 4.87 / 91.9 | 128.5 / 3.04 / 103.8 | 42.3 / 0.70 / 33.1 | 94.53 |
| **1536×6 lik full** | **84.8** / 4.22 / **55.1** | 119.2 / 4.74 / 92.2 | 124.4 / 3.11 / 99.1 | **40.7** / 0.68 / 31.7 | **92.27** |

−3.7 mm for the best model; half of the validation gain is lost on the test, and the 10k checkpoint (the minimum of
the validation loss) is not better than the final weights.

**D-Flow** with the same models: validation (`dflow_valid.py`, 300 poses) 512×4 std 89.67 / 54.83 against 512×4 lik
90.32 / 54.27; test (`dflow_test_extra.py`, PoseD-Flow's settings, Euler 100):

| model | left_leg | legs | arms | trunk | avg mean / best of 10 |
|---|---|---|---|---|---|
| 1536×6 std c10 | 95.1 / 6.37 / 52.1 | 106.1 / 7.48 / 66.8 | 114.5 / 6.14 / 70.6 | 41.5 / 1.24 / 28.2 | 89.31 / 54.43 |
| 1536×6 lik c10 | 99.3 / 7.44 / 52.1 | 112.1 / 8.83 / 67.6 | 117.5 / 7.15 / 69.0 | 46.7 / 1.46 / 30.6 | 93.89 / 54.86 |
| 1536×6 lik full | 102.3 / 7.52 / 54.1 | 111.1 / 8.41 / 68.1 | 121.6 / 7.28 / 70.2 | 45.1 / 1.52 / 28.8 | 95.06 / 55.29 |

The weighting **hurts D-Flow** by 4.6-5.8 mm in the mean, on every occlusion, with the same best of 10 and a larger
spread. D-Flow runs the ODE through all t, and the middle and large t, where the weighting trains less, decide where a
latent goes; SDS uses only small t.

## 7. Steps of SDS

`sds_steps_valid.py`, trial 5, validation, mean / best of 10 (NFE = 2 × steps per iteration):

| steps per iteration | 512×4 std c10 | 1536×6 lik full |
|---|---|---|
| 100 | 94.45 / 71.85 | 90.45 / 68.73 |
| 200 | 93.11 / 70.56 | 87.60 / 66.38 |
| 400 | 92.08 / 69.69 | 86.53 / 65.76 |
| 800 | 92.08 / 69.40 | 85.73 / 65.41 |

The time grows linearly (one forward pass per step, 1.0 → 1.75 min per Pareto trial from 100 to 200 steps on an RTX
4090). The weak model saturates at 400; the strong one still gains at 800. With 400 steps the best SDS is better than
D-Flow on validation in the mean (86.5 against 89.7-90.6) and much worse in best of 10 (65.8 against ~55). Even 1600
NFE without backpropagation is far cheaper than D-Flow. Comparisons with DPoser at its budget should use 100 steps (200
NFE).

## 8. ODE solver inside D-Flow

The same RSGM 512×4 c10 std, test, `dflow_test_extra.py` (Heun is two network calls per step):

| ODE inside D-Flow | left_leg | legs | arms | trunk | avg mean / best of 10 |
|---|---|---|---|---|---|
| Euler 50 | 95.7 / 6.57 / 53.0 | 106.2 / 7.63 / 67.4 | 114.1 / 5.52 / 74.6 | 43.9 / 1.21 / 30.4 | 89.97 / 56.35 |
| Euler 100 (default) | 94.5 / 6.45 / 52.3 | 105.3 / 7.44 / 67.0 | 111.6 / 5.41 / 73.3 | 43.8 / 1.17 / 30.6 | 88.79 / 55.79 |
| **Heun EDM, 50 NFE** | 92.5 / 7.46 / 46.2 | 105.4 / 8.64 / 62.4 | 111.5 / 6.47 / 65.6 | 43.8 / 1.71 / 25.5 | **88.32 / 49.92** |
| Heun EDM, 100 NFE | 92.9 / 7.50 / 46.1 | 106.3 / 8.68 / 62.7 | 111.7 / 6.48 / 65.7 | 44.2 / 1.74 / 25.6 | 88.76 / 50.01 |
| Heun EDM, 200 NFE | 93.7 / 7.61 / 46.1 | 106.9 / 8.77 / 62.9 | 111.7 / 6.49 / 65.9 | 44.2 / 1.75 / 25.5 | 89.11 / 50.08 |

Heun with the EDM times (dense at small t) solves the ODE already at 50 NFE: more calls change nothing. Against Euler
100 it gains 5.9 mm in best of 10 and 1 cm in APD at half the calls; Euler with uniform steps is coarse at small t,
where the pose is finished, and pulls the hypotheses together. With 50 NFE (a quarter of PoseD-Flow's 200) best of 10
equals PoseD-Flow's (49.92 against 50.26), APD is the highest of all rows, and the mean stays 8.7 mm behind: the
solver does not close the mean gap.

## 9. Conclusions

1. The best RSGM row for the comparison: **512×4 std with D-Flow and Heun EDM, 50 NFE** — best of 10 at the level of
   PoseD-Flow, the most diverse hypotheses, a quarter of their network calls; 8.7 mm behind in the mean.
2. The tuning of D-Flow's lr and running weight gives nothing: PoseD-Flow's defaults are at the optimum.
3. The likelihood weighting trades D-Flow for SDS: +3.7 mm for SDS on the test (more with more steps), −5 mm for
   D-Flow. It is a choice per procedure, not a better model; for D-Flow keep "std".
4. The inaccurate score at small t explained part of the gap of SDS to DPoser (the hybrid improves by up to 35 mm), not
   all of it: the best SDS is 92.3 on the test against DPoser's 81.2, about our D-Flow at best.
5. The validation loss of the likelihood runs rises after 5-11k steps, but the earlier checkpoint is not better on the
   test.
6. The remaining mean gap is the model. The ablation of the time-conditioned network
   ([2026-11-14_score_network_architectures.md](2026-11-14_score_network_architectures.md), branch
   `experimental/score-net-ablation`) points at residual connections and LayerNorm as what helps; the next candidate
   is such a network with weighting "std", evaluated with D-Flow and Heun EDM 50.

## 10. 2026-11-15: Inpainting with the reverse process (validation)

**Question.** SDS and D-Flow use the prior only at small noise (SDS) or through one ODE (D-Flow). Does the whole
reverse process of RSGM, with the visible joints kept consistent with the observation, complete poses better?

**Method** (`hp.models.inpainting.rsgm_inpaint`, `run(method="inpaint")`, branch `experimental/rsgm-inpainting`).
The reverse SDE of RSGM (geodesic Euler-Maruyama) from uniform rotations at t = 1 down to eps, with three options:

- *replacement*: before every step the visible joints are drawn from the exact q(x_t | y), the observation noised by
  IGSO(3) of s(t) (the forward process noises every joint independently);
- *resampling* (RePaint, jump 1): a step is repeated r times, the pose at t_next noised back to t by the forward
  Brownian motion in between;
- *guidance* (DPS): the step is moved by −ζ · grad / sqrt(loss) of the squared geodesic error of the visible joints of
  the one-step denoising x̂0 = x_t exp(s(t) · score) (Tweedie on SO(3)), the gradient through the network.

`dposer_inpaint` does the same for DPoser in its z-scored axis-angle space (exact Gaussian replacement, the exact
forward transition of the sub-VP SDE for resampling, the same geodesic DPS loss); it is on the branch
`experimental/dposer-inpainting`.

**Setup.** The first 100 of the 1000 fixed validation poses (`valid_poses`), K = 10, the four occlusions, seeds as the
Optuna studies; MPVPE mean and best of 10 in mm, APD in cm, averaged over the occlusions; seconds on an M-series Mac
(MPS). Scripts: `explore/evaluation/completion_diagnostics/inpaint_valid.py` (inpainting variants),
`dflow_valid.py` (D-Flow, Heun EDM 50), `baselines_valid.py` (PoseD-Flow and DPoser); the rows are in
`~/data/experiments/completion/{inpaint,dflow,baselines}_valid_mac.jsonl`. Nothing in this section is on the test.

### 10.1 Variants on RSGM 512×4 std c10

| variant | mean | best of 10 | APD | s |
|---|---|---|---|---|
| replacement, uniform 200 steps | 124.82 | 62.04 | 10.68 | 4 |
| replacement, EDM 200 | 124.61 | 64.13 | 10.47 | 3 |
| replacement, EDM 200, r = 5 | 106.24 | 56.20 | 8.37 | 16 |
| DPS ζ = 0.02, EDM 200 | 129.15 | 65.08 | 10.69 | 6 |
| DPS 0.1 | 101.51 | **52.53** | 8.20 | 6 |
| DPS 0.2 | 100.71 | 53.00 | 8.01 | 6 |
| DPS 0.5 | 119.39 | 68.37 | 8.71 | 6 |
| DPS 0.2, 500 steps | 110.08 | 61.59 | 8.47 | 14 |
| replacement + DPS 0.05 | 104.68 | 53.55 | 8.39 | 7 |
| DPS 0.2, r = 5 | 97.26 | 55.06 | 7.08 | 32 |
| replacement + DPS 0.2, r = 5 | **94.92** | 61.11 | 5.32 | 46 |
| *D-Flow, Heun EDM 50 (reference)* | *90.88* | *52.40* | *6.08* | *96* |

Replacement alone does not tie the hidden joints to the visible ones (APD 10.5, mean 125). DPS is the lever; its total
strength is about ζ × steps, so too much of it (0.5, or 0.2 over 500 steps) hurts. Resampling trades diversity for the
mean.

### 10.2 Nine networks at two settings

| network | DPS 0.1: mean / best / APD | replacement + DPS 0.2, r = 5: mean / best / APD |
|---|---|---|
| small std c10 (512×4) | 101.51 / 52.53 / 8.20 | 94.92 / 61.11 / 5.32 |
| small std c10, seed 11 | 101.23 / 52.50 / 8.09 | 93.22 / 61.27 / 5.16 |
| abl residual | 101.59 / 52.58 / 8.06 | 90.06 / 59.88 / 4.70 |
| abl norm | 101.30 / 53.40 / 8.12 | 90.37 / 60.44 / 4.82 |
| res norm | 101.43 / 52.65 / 8.03 | 95.65 / 65.24 / 4.81 |
| deep | 105.98 / 54.20 / 8.83 | **88.19** / 56.57 / 5.05 |
| search std c10 (1536×6) | 102.15 / 53.49 / 8.12 | 98.72 / 67.39 / 5.03 |
| small lik c10 | 102.06 / 52.61 / 8.28 | 96.38 / 64.81 / 5.09 |
| dposer like full | 102.67 / 53.41 / 8.22 | 91.25 / 59.89 / 4.88 |

With DPS alone the networks are equal (101-106 / 52.5-54); with resampling they spread by 10 mm at an equal APD. One
point per network is not a comparison of networks: the settings move a model along its trade-off of mean and APD, so
the networks are compared by Optuna fronts of (mean, APD) (10.4).

### 10.3 Against the baselines, per occlusion

Cells: mean / best of 10 / APD.

| row | left_leg | legs | arms | trunk | average | s |
|---|---|---|---|---|---|---|
| PoseD-Flow (their checkpoint, D-Flow) | 85.4 / 49.4 / 5.23 | **99.0** / 68.8 / 6.09 | 113.7 / 67.8 / 6.29 | **37.0** / 25.1 / 1.15 | 83.78 / 52.79 / 4.69 | 304 |
| DPoser full (its own optimization) | **80.0** / **39.0** / 6.56 | 100.1 / 60.7 / 7.41 | **111.7** / 72.2 / 5.95 | 41.3 / 24.0 / 1.93 | **83.28** / 48.95 / 5.46 | **4** |
| RSGM small, D-Flow Heun EDM 50 | 90.8 / 46.6 / 7.33 | 105.2 / 66.2 / 7.99 | 124.9 / 73.0 / 7.03 | 42.6 / 23.9 / 1.98 | 90.88 / 52.40 / 6.08 | 96 |
| RSGM abl residual, inpainting (trial 59) | 86.2 / 44.1 / 6.69 | 100.0 / 61.3 / 8.24 | 119.0 / **63.8** / 8.75 | 41.5 / 23.5 / 1.99 | 86.68 / **48.19** / 6.42 | 66 |
| RSGM abl residual, inpainting (trial 16) | 84.2 / 43.8 / 6.49 | 106.3 / **59.2** / 8.73 | 124.8 / 70.4 / 8.65 | 41.1 / **22.2** / 2.02 | 89.09 / 48.89 / 6.47 | 34 |

Trials 59 (400 steps, ζ = 0.0546, r = 5, no replacement) and 16 (200 steps, ζ = 0.0611, r = 5) are points of the
Optuna study `rsgm_inpaint_mean_apd_residual` after about 60 trials, rerun here by `inpaint_valid.py`
(variants `residual-trial59`, `residual-trial16`). Inpainting beats D-Flow on all three metrics; against DPoser it
loses 3.4 mm in the mean, mostly on the arms (+7 mm, with the best best of 10 and APD 8.75 against 5.95: the
hypotheses are too spread where the observation constrains little) and the left leg (+6 mm, there best of 10 too).
Legs and trunk are at DPoser's level.

### 10.4 Optuna studies (running at the time of writing)

On 1000 validation poses, `scripts/tune_completion.py`: `rsgm_inpaint_pareto` (512×4 std, objectives mean and best of
10) and `rsgm_inpaint_mean_apd_{deep,residual,norm}` (objectives mean and APD, maximized), 120 trials each, on
workstation2. The rule for the test, written in each configuration before it: among the points of the front not
worse than D-Flow Heun EDM 50 of 512×4 on the same poses (best of 10 52.43 mm for the first; APD 6.40 cm for the
others), the lowest mean. Their numbers go here when they are finished and copied to the Mac. Seen so far in the
summaries: the best points sit at the upper ends of the search space (r = 5, 400 steps) without replacement, so the
space should be widened (r up to 12, steps up to 800) in a next study. The finished studies and the test of the chosen
points are in section 11.

### 10.5 The same inpainting on DPoser

| DPoser model | variant | mean | best of 10 | APD | s |
|---|---|---|---|---|---|
| full | its own optimization (10.3) | **83.28** | 48.95 | 5.46 | **4** |
| full | replacement, EDM 200 | 108.81 | 54.71 | 8.97 | 6 |
| full | replacement, r = 5 | 90.50 | 48.33 | 6.87 | 23 |
| full | replacement, r = 10 | 87.25 | 48.60 | 6.36 | 45 |
| full | DPS ζ = 1 (no replacement) | 101.08 | 52.40 | 8.00 | 11 |
| full | DPS 2, r = 5 | 92.49 | 58.43 | 5.26 | 61 |
| full | DPS 0.055, 400 steps, r = 5 (RSGM's best setting) | 127.72 | 66.45 | 10.13 | 94 |
| full | replacement + DPS 0.5, r = 5 | 86.14 | **47.46** | 6.33 | 56 |
| c10, RSGM budget | replacement + DPS 0.5, r = 5 | 87.60 | 47.52 | 6.61 | 53 |
| c10, RSGM budget | replacement, r = 10 | 88.97 | 49.27 | 6.55 | 46 |
| *RSGM abl residual c10* | *inpainting, trial 59 (10.3)* | *86.68* | *48.19* | *6.42* | *66* |

DPS alone on DPoser needs a 10-20 times larger ζ (its coordinates are z-scored axis-angle, not radians) and is weak
even then; replacement is what works, the opposite of RSGM. With the best settings of each, DPoser c10 and RSGM c10
reach the same point (87.6 / 47.5 / 6.6 against 86.7 / 48.2 / 6.4). Inpainting moves both priors to a lower best of
10 and a higher APD at a slightly higher mean than DPoser's own optimization.

### 10.6 Conclusions of this phase

1. Inpainting is the best procedure for RSGM so far: on validation better than D-Flow in mean, best of 10 and APD,
   at fewer seconds; DPS without replacement and resampling r = 5 are the settings that matter.
2. The gain comes from the procedure. With the same inpainting DPoser trained on the same data is as good as RSGM;
   the earlier conclusion "the remaining gap is the model" (section 9, item 6) holds only within D-Flow and SDS.
3. A better mean than DPoser's own optimization (83.3 here) needs a better trade-off for both priors, e.g.
   inpainting followed by a short refinement of each hypothesis (DPoser's optimization from the inpainted pose), or a
   wider search (10.4).

**What it does not show.** 100 poses: differences below ~2 mm are not resolved. The DPoser variants are a hand-made
grid, not tuned as RSGM's were (Optuna). Times are on the Mac and include the network calls of all K hypotheses. No
test numbers: the test is run once for the points chosen by the rules of 10.4.

## 11. 2026-11-16: Inpainting on the test

**Question.** Do the inpainting settings chosen on validation (10.4) keep their gain on the test, and how do they stand
against PoseD-Flow and DPoser there?

**Setup.** Optuna studies on the first 1000 fixed validation poses, K = 10, `scripts/tune_completion.py`, EDM grid,
search space steps {100, 200, 400}, ζ in [0.01, 0.5] (log), r {1, 2, 3, 5}, replacement {yes, no}, on workstation2.
Models: `~/data/experiments/rsgm_posedflow_{small,deep,abl_residual,abl_norm}_s10/run_0001`, all "std" weighting,
c10, one training seed each. The rules, written in the configurations before the studies: for `rsgm_inpaint_pareto`
(objectives mean and best of 10) the lowest mean among the front points with best of 10 ≤ 52.43 mm; for
`rsgm_inpaint_mean_apd_*` (objectives mean and APD) the lowest mean among the front points with APD ≥ 6.40 cm. The
thresholds are D-Flow with Heun EDM 50 on 512×4 on the same poses (93.33 / 52.43 / 6.40). The chosen points were run
once on the 5000 test poses of PoseD-Flow, K = 10, as rows of `scripts/evaluate_completion.py` (commit `c01a464`, on
workstation2); the metrics are read from their caches. MPVPE in mm, APD in cm.

### 11.1 Studies and chosen points (validation, 1000 poses)

| study | network | trials | point | steps | ζ | r | replacement | mean | best of 10 | APD |
|---|---|---|---|---|---|---|---|---|---|---|
| `rsgm_inpaint_pareto` | 512×4 (small) | 60 | 52 | 400 | 0.0577 | 5 | no | 92.43 | 50.86 | 6.99 |
| `rsgm_inpaint_mean_apd_deep` | deep | 120 | 75 | 200 | 0.0975 | 5 | no | 91.29 | 50.19 | 6.84 |
| `rsgm_inpaint_mean_apd_residual` | abl residual | 120 | 81 | 400 | 0.0493 | 5 | no | **88.32** | **49.18** | 6.41 |
| `rsgm_inpaint_mean_apd_norm` | abl norm | 120 | 118 | 200 | 0.0952 | 5 | no | 89.55 | 49.72 | 6.57 |

All four points sit at r = 5, the upper end of the space, without replacement, at ζ 0.05-0.10. Points with a lower
mean exist on the fronts but below the APD threshold (residual trial 59: 88.22 / APD 6.39; deep trial 40: 89.82 /
5.69, with replacement). The Pareto study ran 60 of its 120 planned trials; its last front points (trials 51-56) differ by less than
0.2 mm.
A fifth study, `rsgm_inpaint_mean_apd_search_lik_full` (1536×6, likelihood weighting, full data), finished its 120
trials later (at 88 when this section was first written, rule point trial 71: 90.70 / 50.62 / 6.65). Its rule point is
trial 110 (400 steps, ζ 0.035, r 5, no replacement: 90.66 / 50.43 / 6.66): on validation dominated by residual and
by DPoser c10 inpainting, so it was not run on the test, a negative result (section 13).

### 11.2 Test (5000 poses)

Cells: mean / best of 10 / APD.

| row | left_leg | legs | arms | trunk | average |
|---|---|---|---|---|---|
| PoseD-Flow (their checkpoint, D-Flow) | 79.8 / 47.2 / 5.15 | **92.4** / 59.6 / 6.60 | 106.9 / 66.7 / 5.63 | **39.5** / 27.6 / 1.07 | **79.65** / 50.26 / 4.61 |
| DPoser (its own optimization) | **76.3** / **38.3** / 6.46 | 102.9 / 65.0 / 7.46 | **102.2** / 62.5 / 5.64 | 43.4 / 23.1 / 1.87 | 81.19 / 47.22 / 5.36 |
| RSGM 512×4, D-Flow Heun EDM 50 (section 8) | 92.5 / 46.2 / 7.46 | 105.4 / 62.4 / 8.64 | 111.5 / 65.6 / 6.47 | 43.8 / 25.5 / 1.71 | 88.32 / 49.92 / 6.07 |
| RSGM 512×4, inpainting t52 | 87.1 / 42.1 / 7.90 | 105.7 / 61.5 / 9.86 | 115.4 / 62.6 / 8.24 | 45.9 / 24.3 / 2.07 | 88.55 / 47.62 / 7.02 |
| RSGM deep, inpainting t75 | 86.6 / 42.5 / 7.72 | 107.1 / 63.2 / 9.61 | 122.4 / 67.3 / 8.28 | 44.3 / 23.6 / 1.96 | 90.11 / 49.15 / 6.89 |
| RSGM abl residual, inpainting t81 | 81.6 / 40.1 / 7.07 | 102.4 / 61.2 / 8.90 | 114.9 / 62.5 / 7.96 | 43.1 / **23.0** / 1.88 | 85.50 / **46.70** / 6.45 |
| RSGM abl norm, inpainting t118 | 85.7 / 42.2 / 7.57 | 105.3 / 62.1 / 9.50 | 116.5 / 63.4 / 8.04 | 43.1 / 23.5 / 1.87 | 87.64 / 47.80 / 6.74 |

Inpainting with the residual network has the lowest best of 10 of all rows and the highest APD among the rows within
6 mm of the best mean, but stays 4.3 mm behind DPoser and 5.9 mm behind PoseD-Flow in the mean.

- *Against D-Flow on the same 512×4 network*: equal mean (88.55 against 88.32), best of 10 2.3 mm lower, APD 7.0
  against 6.1 cm. With the residual network inpainting is better in all three.
- *Per occlusion against DPoser*: left_leg +5.3 mm (on validation +6), legs −0.5, trunk −0.3, arms +12.7 mm with an
  equal best of 10 (62.5): on the arms the hypotheses are spread too wide rather than wrong. Against PoseD-Flow the
  legs are also 10 mm behind.
- *Validation to test*: residual drops 2.8 mm in the mean (88.32 → 85.50), D-Flow on 512×4 about 5 mm (93.33 →
  88.32); residual and norm are first and second on both splits, small and deep swap places (within 1.6 mm).

### 11.3 Conclusions of this phase

1. The validation result holds on the test: inpainting is the best RSGM procedure, and with the residual network it
   reaches the best best of 10 of all methods, PoseD-Flow and DPoser included.
2. The mean is still behind DPoser (4.3 mm) and PoseD-Flow (5.9 mm); the gap is mostly the arms, where the hypotheses
   are more diverse than the observation warrants.
3. The best settings are at the boundary of the search space (r = 5); a wider search (r up to 12, steps up to 800),
   a short refinement of each hypothesis after inpainting, or a schedule of ζ are the next candidates, not yet run.

**What it does not show.** One training seed per network: differences of 1-2 mm between networks (residual against
norm) can be seed noise; deep being worst is clear. One test run per row. The Pareto study of 512×4 ran half its
trials. The metrics of the deep, residual and norm rows were read from their cache files directly
(`hp.evaluation.completion.cache_path` and `cached_metrics`): the copies of the weights on the Mac have another
modification time than on workstation2, so the cache check of `cached_run` refuses them; the rows themselves were
computed on workstation2 by `scripts/evaluate_completion.py`. No timing on the test.

## 12. 2026-11-16: Was DPoser compared fairly? DPoser c10 tuned like RSGM (validation)

**Question.** In section 10 RSGM's inpainting was tuned by Optuna (60-120 trials on 1000 validation poses), DPoser's
by a hand grid on 100 poses, and DPoser's own optimization ran with its defaults on DPoser full only. Does that favour
RSGM, and does "the two priors are equal under inpainting" survive when DPoser gets the same tuning?

**Check of the protocol** (code of `hp.evaluation.completion` and of the scripts of section 10). Shared by all rows:
the poses, the occlusion masks, K = 10, the seeds (`run` seeds every batch the same way), the metrics (`metrics()`).
Every method returns the visible joints copied from the observation (`rsgm_inpaint`, `dposer_inpaint`,
`dposer_complete`, D-Flow's `complete`). The 19-joint models (all of ours) set a hidden toe to rest while PoseD-Flow
predicts it; on these poses that costs 0.00 mm (the toes are at rest in the data). DPoser's optimization runs with
the defaults of their `DPoserComp.optimize` and their start (one N(0, I) start shared by the K hypotheses, which lowers
its APD by design). Found asymmetries, all in RSGM's favour:

- tuning: Optuna for RSGM, a hand grid for DPoser's inpainting, none for DPoser's method;
- network calls of the compared settings: RSGM trial 59, 400 × 5 = 2000 (each with a backward pass), against
  200 × 5 = 1000 for DPoser's inpainting point of 10.5;
- training of DPoser c10: DPoser's recipe for 200k steps (lr 2e-4, cosine, warm-up 1000, EMA 0.9999) cut to 30k steps,
  while RSGM's settings were chosen for 30k (lr 2e-3, EMA 0.999). Not retrained here.

**Setup.** Two more studies of `scripts/tune_completion.py` on DPoser c10
(`~/data/experiments/dposer_posedflow_rsgm_budget_s10/run_0001`), objectives mean (minimized) and APD (maximized),
on the same 1000 validation poses with the same seeds, 120 trials each as the RSGM studies, on workstation2:

- `dposer_c10_method_mean_apd`: DPoser's optimization, searched lr 0.01-0.3 (log), t_max 0.03-0.5 (log),
  t_min / t_max 0.05-0.9, iterations {1, 2, 3}, steps per iteration {50, 100, 200}; their loss weights and start;
  the first trial their defaults;
- `dposer_c10_inpaint_mean_apd`: inpainting (`dposer_inpaint`) in RSGM's space (steps {100, 200, 400}, resample
  {1, 2, 3, 5}, replacement on/off, EDM grid) with the guidance 0.1-3 instead of 0.01-0.5 (DPoser's coordinates are
  z-scored axis-angle, not radians).

The rule for the test, written in the configurations before the runs, is that of the RSGM studies (10.4): among the
front points with APD >= 6.40 cm the lowest mean, else the highest APD of the front.

**Results** (validation, 1000 poses, K = 10; MPVPE in mm and APD in cm, means over the four occlusions;
`explore/evaluation/completion_diagnostics/study_fronts.py`). The lowest mean among the finished trials with APD at
least a level, and the point of the rule (mean / best of 10 / APD):

| study | trials | APD >= 4.5 | APD >= 5.5 | APD >= 6.0 | APD >= 6.4 | APD >= 7.0 | rule point |
|---|---|---|---|---|---|---|---|
| RSGM abl residual, inpainting | 120 | 88.22 | 88.22 | 88.22 | 88.32 | 93.45 | t81: 88.32 / 49.18 / 6.41 |
| RSGM abl norm, inpainting | 120 | 89.55 | 89.55 | 89.55 | 89.55 | 93.99 | t118: 89.55 / 49.72 / 6.57 |
| RSGM deep, inpainting | 120 | 89.82 | 89.82 | 91.29 | 91.29 | 94.00 | t75: 91.29 / 50.19 / 6.84 |
| RSGM 1536×6 lik full, inpainting | 120 | 90.66 | 90.66 | 90.66 | 90.66 | 95.06 | t110: 90.66 / 50.43 / 6.66 |
| DPoser c10, inpainting | 120 | 88.26 | 88.26 | 88.26 | **88.26** | **91.60** | t68: 88.26 / **49.22** / 6.46 |
| DPoser c10, DPoser's method | 120 | **86.15** | **86.17** | **86.17** | 91.50 | 91.50 | t28: 91.50 / 51.13 / 7.04 |

With the same procedure and the same tuning RSGM residual and DPoser c10 reach the same front (88.22 / 49.30 / 6.39
against 88.26 / 49.22 / 6.46). DPoser's own optimization is the best below APD 6.0 cm.

Its front near the threshold, DPoser's method (mean / best of 10 / APD):

| trial | settings | mean | best of 10 | APD |
|---|---|---|---|---|
| 0 | their defaults: lr 0.1, t 0.15 → 0.05, 2 × 100 steps | 86.15 | 52.78 | 5.50 |
| 3 | the same, 3 × 200 steps | 86.17 | 49.99 | 6.02 |
| 28 | lr 0.123, t 0.174 → 0.040, 3 × 100 steps | 91.50 | 51.13 | 7.04 |

Lower APD buys a lower mean (the lowest of the study: 78.46 at APD 3.31). No front point lies between APD 6.02 and
7.04, so the rule picks trial 28: a gap of the front, not a measured loss of the method at APD 6.4. Optuna did not
improve on their defaults in the middle of the front.

**Conclusions.**

1. The comparison of section 10 favoured RSGM (tuning, network calls), not DPoser; with DPoser tuned the same way
   conclusion 2 of 10.6 holds: under inpainting the two priors are equal on validation.
2. DPoser's own optimization keeps a lower mean than inpainting of either prior, about 2 mm at an APD lower by
   0.4 cm (86.17 / 49.99 / 6.02 against 88.2 / 49.3 / 6.4). Inpainting reaches higher APD and a lower best of 10.
3. The inpainting fronts end at r = 5 and do not reach APD below about 6.4; a comparison at equal APD with DPoser's
   method needs a wider space (r up to 10-12, steps up to 800).

**Test** (5000 test poses, K = 10, one run per row; rows of `scripts/evaluate_completion.py` on workstation2, metrics
from the cache; mean / best of 10 / APD, the occlusion cells mean only):

| row | left_leg | legs | arms | trunk | average | validation |
|---|---|---|---|---|---|---|
| DPoser full, its own method (defaults) | **76.3** | 102.9 | **102.2** | 43.4 | 81.19 / 47.22 / 5.36 | |
| DPoser c10, its own method, defaults (t0) | 81.6 | 102.6 | 108.5 | 47.2 | 84.99 / 51.07 / 5.38 | 86.15 / 52.78 / 5.50 |
| DPoser c10, its own method t3 (outside the rule) | 83.2 | 101.6 | 103.7 | 47.8 | **84.10** / 48.59 / 5.80 | 86.17 / 49.99 / 6.02 |
| DPoser c10, its own method t28 (rule) | 90.0 | 108.3 | 110.5 | 50.2 | 89.75 / 49.46 / 6.83 | 91.50 / 51.13 / 7.04 |
| DPoser c10, inpainting t68 (rule) | 81.8 | **98.6** | 120.0 | 43.5 | 85.94 / 47.90 / 6.19 | 88.26 / 49.22 / 6.46 |
| RSGM abl residual, inpainting t81 (rule, section 11) | 81.6 | 102.4 | 114.9 | **43.1** | 85.50 / **46.70** / **6.45** | 88.32 / 49.18 / 6.41 |

The rule points of the two priors under inpainting are equal on the test as on validation (0.4 mm in the mean, one
seed each); DPoser's APD fell more from validation to test (6.46 → 6.19 against 6.41 → 6.45). DPoser's method t28 is
the worst mean, as expected of a point picked by a gap of the front (see above), not a measured loss of the method;
t3, outside the rule, keeps its validation place: 1.4 mm below inpainting in the mean, at a lower APD and a higher best
of 10. Training on c10 costs DPoser's method 3.8 mm against DPoser full (84.99 against 81.19 with their defaults).

**What it does not show.** DPoser c10 was trained with a recipe made for 200k steps; one training seed each; the
compared settings differ in network calls. Numbers are validation, on the poses the settings were chosen on, except
the test above.

## 13. 2026-11-16: Fronts over bands of APD (validation)

**Question.** The rule compares one point per study, at the threshold APD 6.4 cm. Do the fronts compare the same way
elsewhere?

**Measure.** For a study, f(a) is the lowest mean (or best of 10) among its trials with APD at least a: the staircase
of the front. Its average over a band of APD, the integral of f over the band divided by the width, is the hypervolume
of the front in that band in mm, lower is better. Only bands covered by every front compare; outside its points a
front would be extrapolated. The 1000 validation poses of the studies, K = 10;
`explore/evaluation/completion_diagnostics/plot_study_fronts.py --bands` (also the figure of all trials and fronts).

Mean (mm):

| study | trials | APD 5.5-6.4 | APD 6.0-7.0 | APD 6.4-7.5 | APD 6.4-8.5 | APD 7.0-8.5 |
|---|---|---|---|---|---|---|
| RSGM abl residual, inpainting | 120 | **88.22** | 90.08 | 93.08 | 96.97 | 99.23 |
| RSGM abl norm, inpainting | 120 | 89.55 | 90.92 | 93.35 | 97.10 | 99.21 |
| RSGM deep, inpainting | 120 | 90.99 | 91.54 | 93.95 | 97.42 | 99.70 |
| RSGM 1536×6 lik full, inpainting | 120 | 90.66 | 91.58 | 93.95 | 97.89 | 100.17 |
| RSGM small, inpainting (objectives mean and best of 10) | 60 | 92.43 | 92.44 | 93.56 | 97.17 | 99.07 |
| DPoser c10, inpainting | 120 | 88.26 | **88.91** | **91.17** | **95.42** | 97.84 |
| DPoser c10, its own method | 120 | 88.40 | 91.37 | 92.91 | 95.57 | **97.19** |

Best of 10 (mm):

| study | trials | APD 5.5-6.4 | APD 6.0-7.0 | APD 6.4-7.5 | APD 6.4-8.5 | APD 7.0-8.5 |
|---|---|---|---|---|---|---|
| RSGM abl residual, inpainting | 120 | 49.09 | 49.52 | 50.32 | **51.35** | 51.96 |
| RSGM abl norm, inpainting | 120 | 49.66 | 50.06 | 50.70 | 51.76 | 52.34 |
| RSGM deep, inpainting | 120 | 50.18 | 50.30 | 51.19 | 52.03 | 52.69 |
| RSGM 1536×6 lik full, inpainting | 120 | 50.34 | 50.59 | 51.15 | 52.21 | 52.79 |
| RSGM small, inpainting (objectives mean and best of 10) | 60 | 50.81 | 50.81 | 50.93 | 51.63 | **51.95** |
| DPoser c10, inpainting | 120 | **48.87** | **48.97** | **49.64** | 51.36 | 52.29 |
| DPoser c10, its own method | 120 | 50.47 | 51.10 | 51.54 | 51.92 | 52.24 |

In the band 5.5-6.4 the inpainting fronts have no points of their own below about 6.4 (f takes their first point
above), so that column compares the rule points more than the fronts.

**Conclusions.**

1. Under the same inpainting DPoser c10 is slightly better than every RSGM network: 1.2-1.9 mm lower mean in the bands
   from APD 6.0 up, and a lower best of 10 up to APD 7.5. The fronts meet at the threshold 6.4, so the rule points (and
   their test rows) are equal; "the two priors are equal under inpainting" (10.6, 12) holds at that point only.
2. DPoser's own method is not better than inpainting of the same model where both have points: equal in 5.5-6.4
   (2 mm better in 5.5-6.0, 3 mm worse in 6.0-6.4, its gap), 1.7-2.5 mm worse in 6.0-7.5, 0.6 mm better in 7.0-8.5.
   Its advantage is the region below APD 5.5 (down to 78.5 mm at APD 3.3), which inpainting does not reach.
3. Among the RSGM networks residual and norm are ahead of deep and 1536×6 lik full by up to 2.5 mm; with one training
   seed each, residual against norm is within noise. The likelihood weighting, which helped SDS and hurt D-Flow, brings
   inpainting nothing (the last 32 of its 120 trials moved its front by at most 0.35 mm).

**What it does not show.** A band average depends on the band, hence five bands. A front is also a measure of how well
Optuna explored the space: DPoser's method has 15 of its 120 trials in the range of the figure (mean ≤ 105 mm, APD
3-9 cm). One training seed per model. Nothing here is on the test; testing a second point per study (e.g. the lowest
mean at APD ≥ 7.5) is left until the studies on full-data models (`config/completion_tuning/*full*`) are in.

## 14. 2026-11-19: PoseD-Flow with a tuned D-Flow on the test

**Question.** The study `posedflow_dflow_mean_apd` (D-Flow on their pretrained PoseD-Flow, 1000 validation poses,
K = 10; lr, running weight and optimizer steps searched, 25 finished trials) found trial 17 (lr 0.05, running weight
1e-3, 200 steps for all occlusions) better than their protocol (trial 0: lr 0.1, the same otherwise) in all three
metrics: mean 86.34 against 87.31 mm, best of 10 53.04 against 55.59 mm, APD 5.10 against 4.80 cm. Does it hold on
the test?

**Test (5000 poses, K = 10; MPVPE in mm, APD in cm; the row run once).**

| row | mean | best of 10 | APD | left_leg | legs | arms | trunk |
|---|---|---|---|---|---|---|---|
| PoseD-Flow: their checkpoint (their protocol, 300 steps for the arms) | **79.65** | 50.26 | 4.61 | 79.8 | 92.4 | 106.9 | 39.5 |
| PoseD-Flow: their checkpoint, D-Flow t17 | 81.40 | **49.13** | **5.10** | 82.7 | 94.4 | 109.1 | 39.5 |

Pose by pose (t17 minus theirs, 95% interval) the mean MPVPE is worse by 2.85 ± 0.49 mm (left_leg), 2.02 ± 0.44
(legs), 2.15 ± 0.63 (arms), -0.03 ± 0.18 (trunk). On the test t17 is not better: it is another point of the same
front, more diverse and with a lower best of 10, at a higher mean. On validation the same comparison went the other
way (left_leg 85.3 against 87.0, legs 105.9 against 107.6), a swing of about 4.5 mm.

**Not the batch.** The two test rows ran with different batches (`--together`, 250 against 500 poses: 2500 against
5000 hypotheses per call), and the loss of D-Flow is a mean over the batch, so the gradient of a hypothesis halves
with a batch twice as large; Adam ignores that scale only while its eps is negligible. Checked on 100 validation
poses (left_leg, t17 settings, same seeds): the loss halved changes the hidden joints by 0.0112 rad on average, a
control with the loss times 1.0001 by 0.0085 rad, and the error to the ground truth is 0.15317 rad at scale 1 against
0.15356 and 0.15357. The scale moves the result no more than float noise does over 200 steps: the batch is speed only,
as `SPEED` in `hp.evaluation.completion` assumes. The arms differ by their steps (300 against 200); the legs do not.

**By difficulty (test).** The poses of each occlusion split into five equal groups by the mean of both rows' error on
the pose (a difficulty that favours neither row); t17 minus theirs:

| occlusion | difficulty (mm) | Δ mean (mm) | Δ best of 10 (mm) | Δ APD (cm) |
|---|---|---|---|---|
| left_leg | 16-44 | +3.60 ± 0.56 | -0.63 | +0.86 |
| left_leg | 44-58 | +3.29 ± 0.70 | -1.72 | +0.89 |
| left_leg | 58-78 | +1.98 ± 0.86 | -2.32 | +0.77 |
| left_leg | 78-113 | +0.75 ± 1.14 | -2.46 | +0.68 |
| left_leg | 113-435 | +4.64 ± 1.74 | -1.08 | +0.75 |
| legs | 24-58 | +4.27 ± 0.57 | +0.07 | +0.95 |
| legs | 58-73 | +4.48 ± 0.76 | +0.36 | +1.02 |
| legs | 73-93 | -0.16 ± 0.84 | -2.47 | +0.59 |
| legs | 93-123 | -0.62 ± 1.01 | -2.44 | +0.47 |
| legs | 123-383 | +2.13 ± 1.45 | +0.79 | +0.15 |
| arms | 25-66 | +3.45 ± 0.68 | -1.03 | +0.61 |
| arms | 66-85 | +3.60 ± 1.01 | -0.88 | +0.59 |
| arms | 85-108 | +4.18 ± 1.17 | -0.95 | +0.59 |
| arms | 108-145 | +1.60 ± 1.53 | +0.55 | +0.21 |
| arms | 145-376 | -2.08 ± 2.17 | -0.15 | +0.03 |

The APD of t17 is higher in every group; its mean is 3.3-4.5 mm worse on the easier poses, where its best of 10 gains
little or nothing, and about equal on the poses of middle difficulty, where its best of 10 gains 2.3-2.5 mm (legs,
left_leg). The hardest groups do not follow one pattern (left_leg +4.64, arms -2.08, both with wide intervals).

**Hypothesis (not tested by an experiment of its own).** A step of Adam is about lr in size whatever the gradient, so
with half the lr the start of the ODE moves half as far from its random draw in 200 steps, and the 10 hypotheses stay
more spread. Where the visible joints all but fix the hidden ones (easy poses) that spread is error: the mean worsens
and the best of 10 has nothing to gain. Where the hidden part is truly ambiguous (middle poses) the spread covers other
plausible completions: the best of 10 improves at an equal mean. The validation poses are harder than the test poses
(their protocol: 87.3 against 79.7 mm), with more ambiguous ones, so a lower lr pays there and not on the test. A
direct test would be the same comparison by difficulty on the validation poses (the study keeps only the means per
occlusion), or a rule that is checked on a split as hard as the test.

**Conclusions.** Their protocol stays the PoseD-Flow row of the comparison. A point that wins on validation by about
1 mm need not win on the test: the validation and test poses differ in difficulty, and settings that trade mean for
diversity move differently on them. Comparing whole fronts on the test (N points per front, chosen by a rule fixed
before) is less exposed to this than a single winner.

## 15. 2026-11-20: Six fronts on the test

**Question.** Sections 11-14 tested one point per study. Which of the procedures and priors is better over the whole
range of diversity, judged on the test by a rule fixed before it?

**Fronts.** A front is a family (the networks of DPoser, of RSGM, PoseD-Flow's checkpoint; c10 and full training data
together) times a completion method, over all its Optuna studies on the 1000 validation poses (K = 10, objectives mean
and APD; the trials copied between studies left out):

- **DPoser dposer**: DPoser's own optimization on DPoser (c10 and full);
- **DPoser inpaint**: the reverse-process inpainting of section 10 on DPoser;
- **RSGM inpaint**: the same inpainting on RSGM (the ablation networks, 512×4, dposer-like, full and c10);
- **RSGM dposer**: DPoser's optimization (SDS) on RSGM dposer-like full;
- **RSGM dflow**: D-Flow (Heun EDM, 25 steps) on RSGM dposer-like and dposer-like aniso full;
- **PoseD-Flow dflow**: D-Flow on their checkpoint (their ODE), with the studies of lr, running weight and, since
  e58f6fe, the weight w of the guess in the start (`guess_weight`: sqrt(w) guess + sqrt(1 - w) noise, theirs 0.25).

**Rule (a212c54, fixed before the test).** For each of 10 thresholds a = 4.0, 4.5, ..., 8.5 cm the trial with the
lowest mean among those with APD ≥ a (a corner of the staircase of the front; `hp.evaluation.fronts.spanning`); a trial
chosen for several thresholds runs once. Each point runs once on the 5000 test poses
(`scripts/select_test_points.py`, `scripts/evaluate_completion.py --points config/completion_test_points.yaml`). The
fronts compare by the hypervolume of their points over APD 4.0-8.5: the area between the staircase f(a), the lowest
metric among the points with APD ≥ a, and a reference fixed before the test (mean 115 mm, best of 10 65 mm). Band mean
= 115 - HV / 4.5 is the average of the staircase over the band, in mm; where a front has no point with APD ≥ a it gets
no area there, as if its error were the reference. The points of PoseD-Flow were chosen after its w study (a15567a);
those of the other five fronts did not change with it.

**Hypervolume (5000 test poses against 1000 validation poses).**

| front | val band mean | test band mean | val HV best of 10 | test HV best of 10 |
|---|---|---|---|---|
| DPoser dposer | **89.24** | **86.85** | 71.46 | **82.04** |
| RSGM inpaint | 92.40 | 90.05 | 66.27 | 74.78 |
| DPoser inpaint | 91.10 | 91.71 | **71.60** | 72.73 |
| RSGM dflow | 94.03 | 92.91 | 61.44 | 71.64 |
| PoseD-Flow dflow | 95.82 | 93.28 | 49.83 | 59.11 |
| RSGM dposer | 97.84 | 96.69 | 40.74 | 50.14 |

**The points (mean / best of 10 in mm, APD in cm; validation → test).**

| front | point | thresholds | validation | test |
|---|---|---|---|---|
| DPoser dposer | c10 t29 | 4 | 80.61 / 53.20 / 4.22 | 82.97 / 54.60 / 4.16 |
| | full t3 | 4.5-5.5 | 83.69 / 48.73 / 5.66 | 79.59 / 44.95 / 5.63 |
| | c10 t3 | 6 | 86.17 / 49.99 / 6.02 | 84.10 / 48.59 / 5.80 |
| | full t28 | 6.5, 7 | 91.16 / 48.56 / 7.32 | 87.98 / 46.21 / 7.33 |
| | full t47 | 7.5 | 94.68 / 49.99 / 7.82 | 90.74 / 47.09 / 7.77 |
| | full t112 | 8 | 97.42 / 50.83 / 8.36 | 94.73 / 48.44 / 8.35 |
| | c10 t68 | 8.5 | 99.09 / 52.73 / 8.52 | 97.74 / 51.32 / 8.29 |
| DPoser inpaint | full t56 | 4-6 | 86.31 / 47.65 / 6.32 | 84.88 / 45.93 / 6.38 |
| | full t90 | 6.5 | 86.75 / 47.38 / 6.52 | 87.27 / 46.43 / 6.66 |
| | full t89 | 7 | 90.51 / 48.33 / 7.04 | 92.74 / 48.69 / 7.20 |
| | full t49 | 7.5 | 94.62 / 49.92 / 7.51 | 97.97 / 51.17 / 7.66 |
| | c10 t103 | 8 | 100.03 / 53.12 / 8.03 | 103.19 / 55.22 / 7.85 |
| | full t38 | 8.5 | 103.85 / 53.80 / 8.52 | 109.97 / 57.15 / 8.73 |
| RSGM inpaint | abl residual t59 | 4-6 | 88.22 / 49.30 / 6.39 | 85.34 / 46.71 / 6.44 |
| | abl norm t118 | 6.5 | 89.55 / 49.72 / 6.57 | 87.64 / 47.80 / 6.74 |
| | pareto t56 | 7 | 92.49 / 50.82 / 7.00 | 88.56 / 47.52 / 7.04 |
| | 512×4 full t63 | 7.5 | 95.57 / 50.55 / 7.52 | 92.87 / 49.05 / 7.39 |
| | 512×4 full t26 | 8 | 99.45 / 51.96 / 8.03 | 97.45 / 50.71 / 8.04 |
| | 512×4 full t105 | 8.5 | 103.66 / 53.06 / 8.52 | 104.15 / 53.72 / 8.46 |
| RSGM dflow | aniso t29 | 4, 4.5 | 87.71 / 56.26 / 4.56 | 88.44 / 57.10 / 4.70 |
| | aniso t51 | 5, 5.5 | 89.68 / 52.95 / 5.57 | 89.03 / 52.83 / 5.63 |
| | aniso t58 | 6 | 91.12 / 51.26 / 6.18 | 90.91 / 51.05 / 6.30 |
| | aniso t25 | 6.5 | 93.54 / 51.50 / 6.61 | 93.42 / 50.96 / 6.76 |
| | t51 | 7 | 96.28 / 51.22 / 7.26 | 91.40 / 48.35 / 7.03 |
| | aniso t19 | 7.5 | 98.39 / 51.12 / 7.74 | 97.37 / 50.26 / 7.76 |
| | aniso t36 | 8 | 100.51 / 51.75 / 8.06 | 99.82 / 51.08 / 8.12 |
| | t45 | 8.5 | 103.82 / 53.00 / 8.50 | 102.00 / 50.91 / 8.55 |
| PoseD-Flow dflow | w study t14 | 4 | 84.23 / 56.79 / 4.01 | 79.30 / 52.10 / 4.09 |
| | w study t8 | 4.5 | 85.87 / 54.39 / 4.76 | 81.70 / 50.32 / 4.83 |
| | t17 | 5 | 86.34 / 53.03 / 5.10 | 81.40 / 49.13 / 5.10 |
| | w study t12 | 5.5 | 88.45 / 52.58 / 5.53 | 86.04 / 50.09 / 5.67 |
| | w study t4 | 6, 6.5 | 92.00 / 51.18 / 6.52 | 88.65 / 48.63 / 6.51 |
| | w study t19 | 7 | 94.77 / 51.48 / 7.01 | 92.71 / 49.38 / 7.08 |
| | t11 | 7.5, 8 | 104.66 / 56.59 / 8.00 | 105.56 / 55.74 / 8.07 |
| RSGM dposer | t15 | 4, 4.5 | 92.03 / 61.66 / 4.62 | 90.95 / 58.07 / 5.00 |
| | t24 | 5 | 93.74 / 60.03 / 5.20 | 93.70 / 56.69 / 5.66 |
| | t99 | 5.5 | 94.81 / 57.93 / 5.79 | 95.62 / 55.24 / 6.32 |
| | t64 | 6 | 95.46 / 57.16 / 6.10 | 96.93 / 54.64 / 6.73 |
| | t63 | 6.5 | 97.09 / 56.67 / 6.55 | 98.03 / 53.86 / 7.18 |
| | t106 | 7 | 99.02 / 55.92 / 7.16 | 100.92 / 53.81 / 7.84 |
| | t82 | 7.5 | 100.95 / 55.83 / 7.51 | 102.99 / 54.03 / 8.23 |
| | t22 | 8 | 104.27 / 56.23 / 8.18 | 105.82 / 54.29 / 8.84 |
| | t56 | 8.5 | 106.36 / 56.70 / 8.51 | 107.24 / 54.54 / 9.11 |

The w study gave PoseD-Flow five of its seven points, all at the low and middle APD: a smaller weight of the guess
(more noise in the start) is what moves their D-Flow along its front. Neither inpainting front reaches below APD
6.3 cm, so their first point serves the thresholds 4-6.

**Observations.**

1. **The order of the fronts is the same on both splits**, and inside a front the test APD of the points keeps their
   order with one exception (RSGM dflow t51, 91.40 mm at 7.03 cm on the test, below its neighbours). Single points
   move by several mm between the splits (section 14); the fronts move together, five of six down, as the test poses
   are easier.
2. **DPoser's own optimization is the best front** in the mean (by 3.2 mm over the next) and in best of 10, and it
   alone covers APD below 6 cm well with a low mean.
3. **Under the same procedure the priors are on par.** Inpainting: RSGM ahead on the test (90.05 against 91.71 mm,
   HV best of 10 74.78 against 72.73), DPoser ahead on validation (91.10 against 92.40); section 13 found DPoser
   inpainting slightly ahead from APD 6 up. D-Flow: RSGM against PoseD-Flow 92.91 against 93.28 mm on the test, 94.03
   against 95.82 on validation; part of that is the band (PoseD-Flow has no point above 8.07 cm and gets no area in
   8.07-8.5, about 0.9 mm of band mean; over 4.0-8.0 the test order turns, 91.84 against 90.70). Equal in the mean,
   then; in best of 10 RSGM is clearly ahead (71.64 against 59.11 on the test, 61.44 against 49.83 on validation):
   with the same D-Flow its hypotheses cover the truth more often.
4. **SDS on RSGM is the weakest front** (96.69 mm): DPoser's optimization is tuned to DPoser; on RSGM the APD of its points
   also grows from validation to test more than on any other front (by 0.4-0.7 cm from t99 up).

## 16. 2026-11-20: Seven compared rows and hard poses (test)

**Rows.** `explore/evaluation/pose_completion.ipynb` keeps seven rows: their protocols (PoseD-Flow's checkpoint, DPoser
full with its method), the tuned points of section 14 and of the full-data studies (one point per study by the rule
of 10.4, threshold APD 6.4, picked before the fronts: RSGM dposer-like full inpaint t108, RSGM 512×4 full inpaint t16,
DPoser full method t28) and RSGM 512×4 full with D-Flow (Euler 100). Test, 5000 poses, K = 10:

| row | mean | best of 10 | APD | left_leg | legs | arms | trunk |
|---|---|---|---|---|---|---|---|
| PoseD-Flow: their protocol | **79.65** | 50.26 | 4.61 | 79.8 | **92.4** | 106.9 | **39.5** |
| PoseD-Flow: D-Flow t17 | 81.40 | 49.13 | 5.10 | 82.7 | 94.4 | 109.1 | **39.5** |
| DPoser full: its method (their protocol) | 81.19 | 47.22 | 5.36 | **76.3** | 102.9 | **102.2** | 43.4 |
| DPoser full: its method t28 | 87.98 | **46.21** | 7.33 | 87.7 | 110.0 | 106.9 | 47.3 |
| RSGM 512×4 full: D-Flow | 87.85 | 55.71 | 5.00 | 92.8 | 103.9 | 112.0 | 42.7 |
| RSGM dposer-like full: inpaint t108 | 88.07 | 47.23 | 6.76 | 81.8 | 104.1 | 124.0 | 42.4 |
| RSGM 512×4 full: inpaint t16 | 91.95 | 49.33 | 7.16 | 88.7 | 108.5 | 124.6 | 46.0 |

RSGM t108 matches DPoser's protocol in best of 10 (47.23 against 47.22) at a higher APD, and DPoser full inpainting
t90 (87.27 / 46.43 / 6.66, the DPoser inpaint front) at about the same APD; its loss in the mean is the arms (124 mm
against 102-107 for the other procedures), as for every inpainting row.

**Hard poses.** Per row the poses of an occlusion are ranked by their mean MPVPE over the 10 hypotheses, the rank
scaled to (0, 1]; the ranks are averaged over the seven rows and the top 5% (250 of 5000) are "hard for all". Ranks
rather than errors, so that no row's scale decides (`hp.visualization.completion_figure.hard_poses`). "In common":
the share of a row's own top 5% inside the common set. Mean / best of 10 in mm, APD in cm, on the hard poses (all
poses in brackets):

| row | left_leg | legs | arms | trunk |
|---|---|---|---|---|
| PoseD-Flow: their protocol | 195.4 / 118.0 / 10.9 (79.8) | 207.8 / 133.3 / 14.0 (92.4) | 237.6 / 144.8 / 10.5 (106.9) | **91.3** / 66.6 / 2.4 (39.5) |
| PoseD-Flow: D-Flow t17 | 204.9 / 120.2 / 11.7 (82.7) | 213.4 / 138.0 / 14.0 (94.4) | 239.9 / 139.8 / 11.0 (109.1) | 91.9 / 65.9 / 2.4 (39.5) |
| DPoser full: its method | **164.3** / 93.0 / 9.2 (76.3) | 211.5 / 151.6 / 10.8 (102.9) | 216.3 / 147.1 / 8.1 (102.2) | 97.2 / 66.8 / 2.7 (43.4) |
| DPoser full: its method t28 | 173.3 / **92.1** / 12.1 (87.7) | **193.7** / 124.2 / 14.3 (110.0) | **215.6** / **122.2** / 11.6 (106.9) | 101.5 / **62.3** / 3.9 (47.3) |
| RSGM 512×4 full: D-Flow | 220.7 / 122.1 / 13.6 (92.8) | 262.1 / 152.7 / 17.5 (103.9) | 233.1 / 140.5 / 11.1 (112.0) | 95.1 / 66.1 / 2.7 (42.7) |
| RSGM dposer-like full: inpaint t108 | 173.5 / 95.0 / 12.2 (81.8) | 212.4 / **123.5** / 16.5 (104.1) | 242.1 / 123.3 / 14.3 (124.0) | 93.2 / 62.5 / 2.9 (42.4) |
| RSGM 512×4 full: inpaint t16 | 198.3 / 113.4 / 12.1 (88.7) | 218.0 / 130.5 / 16.5 (108.5) | 247.4 / 134.3 / 13.9 (124.6) | 100.7 / 64.6 / 3.5 (46.0) |

Overlap: the poses in the top 5% of exactly all seven rows are 27 (left_leg), 57 (legs), 28 (arms), 82 (trunk); a
row's own top lies 45-80% inside the common set. Pairwise shares of the own tops group by method rather than by
prior: PoseD-Flow with its t17 71-81%, RSGM t108 with t16 59-75%, DPoser with its t28 56-74%; across methods 26-66%,
the lowest DPoser's method against PoseD-Flow on the arms (26%) and against RSGM D-Flow (27%).

**Conclusions.**

1. On hard poses the mean error is 2-2.5 times the average and APD grows with it: the hypotheses spread where the
   visible joints leave the pose open.
2. In the mean on hard poses DPoser's method leads on left_leg, legs and arms, PoseD-Flow on the trunk; in best of 10 the
   diverse rows lead in every occlusion (DPoser t28, RSGM t108), which is what diversity is for.
3. What is hard depends mostly on the procedure: rows of one procedure (two networks, or two settings) agree on
   their hard poses more than rows of two procedures do, with some overlap of the ranges.

The notebook browses them: pick "hard for all" (and "previous hard") steps through the common set of the chosen
occlusion in the order of the mean rank.

## 17. 2026-11-21: Do the training poses cover the held-out poses? (validation and test)

Question: would another composition of the training data help (a selection of the frames by how far they are from
the poses already taken, hard example mining, prioritized sampling)? Such schemes reweight the poses in the data; they
can only help if c10 misses poses that the full split has, or if the hard poses are underweighted rather than absent.

For every query pose the nearest training pose is found in the chordal metric on SO(3)^21 (sum over the joints of
||R - R'||_F^2, the largest inner product of the flattened matrices), once over the whole pose and once over the
joints hidden by each occlusion; the distance is the RMS geodesic angle over those joints of the nearest pose,
sqrt(mean_j theta_j^2), in degrees. Training sets: full (PoseD-Flow's train split, 18.6M poses), c10 (every 10th,
1.86M, the set of the c10 runs) and c100 (every 100th). Queries: the 1000 validation poses of `valid_poses` (seed 0)
and the 5000 test poses of the protocol. `explore/evaluation/data_coverage/coverage.py`, Mac (MPS), about 9.5 hours
(two passes over the full split).

Median / 90% / 99% of the angle to the nearest training pose (deg):

| queries | part | full | c10 | c100 | c10 / full (median) |
|---|---|---|---|---|---|
| valid 1000 | pose | 10.41 / 16.51 / 22.18 | 10.56 / 16.72 / 22.36 | 11.18 / 17.91 / 23.25 | 1.01 |
| valid 1000 | left_leg | 2.22 / 4.44 / 7.63 | 2.47 / 4.90 / 8.15 | 2.96 / 5.83 / 9.24 | 1.11 |
| valid 1000 | legs | 4.40 / 7.83 / 13.08 | 4.67 / 8.05 / 13.25 | 5.24 / 8.98 / 14.77 | 1.06 |
| valid 1000 | arms | 8.90 / 15.34 / 21.73 | 9.12 / 15.62 / 22.04 | 9.89 / 17.16 / 22.87 | 1.02 |
| valid 1000 | trunk | 3.90 / 7.82 / 13.10 | 4.09 / 8.05 / 13.14 | 4.73 / 9.13 / 14.98 | 1.05 |
| test 5000 | pose | 9.54 / 16.26 / 20.47 | 9.75 / 16.52 / 20.59 | 10.45 / 17.63 / 22.29 | 1.02 |
| test 5000 | left_leg | 2.51 / 4.82 / 8.77 | 2.85 / 5.32 / 9.50 | 3.48 / 6.36 / 11.25 | 1.13 |
| test 5000 | legs | 4.87 / 8.94 / 14.19 | 5.14 / 9.39 / 14.84 | 5.89 / 10.40 / 16.48 | 1.06 |
| test 5000 | arms | 7.72 / 13.79 / 18.49 | 7.94 / 14.21 / 19.00 | 8.78 / 15.64 / 20.28 | 1.03 |
| test 5000 | trunk | 3.81 / 7.61 / 11.00 | 4.06 / 7.96 / 11.61 | 4.80 / 8.86 / 12.78 | 1.07 |

Ten times more poses bring the nearest pose 0.1-0.3 deg closer in the median and 0.1-0.7 deg at the 99th
percentile: the distance is set by the other subjects and datasets of the held-out splits, not by how densely the
training sequences are sampled in time. The arms are the least covered part (8-9 deg against 2-5 for legs and
trunk), as they are the occlusion with the largest gap in the mean (sections 15-16).

Hard poses of the test (descriptive, nothing is chosen on them): the top 5% per occlusion by the mean rank over all
71 cached test rows with 5000 poses (`hard_poses` as in section 16, but over every row rather than the seven compared
ones). Median angle of the hidden part to the nearest training pose (deg), and the Spearman correlation of that angle
with the mean rank over all 5000 poses:

| occlusion | hard, full | rest, full | hard, c10 | rest, c10 | Spearman (c10) |
|---|---|---|---|---|---|
| left_leg | 5.18 | 2.46 | 5.59 | 2.78 | 0.56 |
| legs | 10.96 | 4.78 | 11.55 | 5.04 | 0.76 |
| arms | 12.99 | 7.60 | 13.37 | 7.84 | 0.65 |
| trunk | 7.43 | 3.73 | 7.76 | 3.96 | 0.43 |

The hard poses are 1.7-2.3 times farther from the training data than the rest, and the distance predicts the
difficulty (Spearman 0.43-0.76); the full split brings them only 0.3-0.6 deg closer than c10.

**Conclusions.**

1. c10 covers the validation and test poses as well as the full split does (within 1-13% in the median). This agrees
   with full data giving RSGM only 0.5-1 mm: the amount of training poses is not what limits completion.
2. The hard poses of completion are poses whose hidden part has no close configuration in PoseD-Flow's training
   data, in c10 and in the full split alike: completing them is extrapolation, not a pose underweighted in training.
3. So a selection or reweighting of the training poses (frames far from the running mean or from the poses taken,
   hard example mining, prioritized sampling) has nothing to add for completion: it only moves weight among poses that
   are there. Dropped without training. What could help the hard poses is other data (other datasets, augmentation)
   or how the model extrapolates (the network and the noise, the score network report).

**Correction, 2026-11-25** (`2026-11-22_data_weighting.md`, branch `experimental/data-weighting`): conclusions 2 and 3
go too far. Models trained with rarity weights (β 3.3) were tested: with D-Flow the band mean is 3.8 mm lower than with
uniform sampling (91.53 → 87.76 mm, test), with inpainting and SDS there is no difference in the band mean. On training
poses, the rarest fifth has 3.3 times the inpainting error of the most common fifth (91.9 against 27.7 mm), and the
weights lower it by 10.8 mm, so part of the difficulty is underfitting. Extrapolation remains: at the same rarity
validation is harder than training (85.6 against 72.9 mm at 11-15 deg). The weights cost generation (Wasserstein 14.52 →
17.31 deg), so uniform sampling is kept; weighting for D-Flow is left open.

## 18. 2026-11-26: Where the fronts stand over APD (test, the state of this stage)

**State of this stage.** The base RSGM is `fact_RDW` β 0 (512×4 residual, dropout 0.1, width 1024, c10, uniform
sampling, seed 10): the decision of the data weighting report, one model for every protocol. β 3.3 is not a base
model: it gains only in D-Flow and costs generation (Wasserstein 14.52 → 17.31 deg). No new run; the table reads the
test points already run (5000 poses, K = 10), one training seed per network.

A front has no single APD: it is a staircase of 6-9 test points. The columns "≥ a" give the lowest test mean (mm)
among the points of the front with APD ≥ a cm; the band mean is the average of this staircase over 4.0-8.5, and "–"
(no point that diverse) counts as the reference 115 mm there. Points chosen on validation by the rule of section 15;
the `fact_RDW` fronts by the same rule over its own Optuna studies (120 trials inpainting and SDS, 60 D-Flow; data
weighting report, section 3). The rows "RSGM family" are the fronts of section 15 over all RSGM networks; a front of
one network is not directly comparable with a front over several.

| front | band mean | HV best of 10 | APD of points | ≥ 4.0 | ≥ 5.0 | ≥ 6.0 | ≥ 6.5 | ≥ 7.0 | ≥ 7.5 | ≥ 8.0 | ≥ 8.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DPoser, own optimization | **86.85** | **82.04** | 4.16-8.35 | 79.6 | **79.6** | 88.0 | 88.0 | **88.0** | **90.7** | **94.7** | – |
| RSGM `fact_RDW` β 0, inpainting | 90.05 | 76.43 | 6.36-8.55 | 84.6 | 84.6 | **84.6** | 90.1 | 91.1 | 99.2 | 99.2 | **104.8** |
| RSGM `fact_RDW` β 0, D-Flow | 91.53 | 75.76 | 4.34-8.61 | 84.5 | 88.7 | 88.7 | 88.7 | 92.4 | 93.5 | 103.3 | 103.3 |
| DPoser, inpainting | 91.71 | 72.73 | 6.38-8.73 | 84.9 | 84.9 | 84.9 | **87.3** | 92.7 | 98.0 | 110.0 | 110.0 |
| PoseD-Flow, D-Flow | 93.28 | 59.11 | 4.09-8.07 | **79.3** | 81.4 | 88.7 | 88.7 | 92.7 | 105.6 | 105.6 | – |
| RSGM `fact_RDW` β 0, SDS | 96.67 | 49.30 | 4.50-9.10 | 92.0 | 93.0 | 94.3 | 97.5 | 97.5 | 99.5 | 101.9 | 104.9 |
| RSGM family, inpainting | 90.05 | 74.78 | 6.44-8.46 | 85.3 | 85.3 | 85.3 | 87.6 | 88.6 | 97.5 | 97.5 | – |
| RSGM family, D-Flow | 92.91 | 71.64 | 4.70-8.55 | 88.4 | 89.0 | 90.9 | 91.4 | 91.4 | 97.4 | 99.8 | 102.0 |
| RSGM family, DPoser's optimization | 96.69 | 50.14 | 5.00-9.11 | 91.0 | 91.0 | 95.6 | 96.9 | 98.0 | 100.9 | 103.0 | 105.8 |

Sources: the `fact_RDW` rows from `~/data/experiments/data_weighting/test_fast.log` and `test_dflow.log`, the other
rows from the points of section 15.

**Reading.**

1. **Low diversity (APD 4-6 cm) belongs to DPoser's optimization and PoseD-Flow** (79-81 mm). RSGM D-Flow is 4-5 mm
   behind there, and inpainting (RSGM and DPoser alike) does not reach below 6.3 cm.
2. **Around 6-6.5 cm the best point is RSGM inpainting** (`fact_RDW` t99, 84.61 mm at 6.36 cm, best of 10 46.31 mm),
   level with DPoser inpainting (84.88 at 6.38).
3. **From 6.5 to 8 cm DPoser's optimization leads again**, by 2-5 mm over the best RSGM front; RSGM D-Flow is 2.8 mm
   behind at APD ≥ 7.5.
4. **At APD ≥ 8.5 cm only the RSGM fronts and DPoser inpainting have points**, so part of the band mean of DPoser's
   optimization and PoseD-Flow is the reference 115 mm of an empty end, as noted in section 15.
5. One training seed per network, one test run per point: differences below about 1-1.5 mm between fronts are not
   evidence (the seeds of `fact_RDW` differ by up to 1.4 mm in SDS on validation, score report section 21).

**Cost on the test.** The minutes per occlusion that the same runs logged (5000 poses, K = 10, one GPU of a
workstation, `test_fast.log` and `test_dflow.log`; the two logs were not checked to have run on the same GPU under the
same load, so the ratios are an order of magnitude, not a measurement):

| procedure on `fact_RDW` β 0 | test points | min per occlusion | against D-Flow |
|---|---|---|---|
| D-Flow (Heun EDM, 50 network calls per ODE) | 8 | 9.9-14.8 | 1× |
| inpainting | 6 | 0.4-3.7 (most 0.7-1.5) | about 4-37×, typically ~10× |
| SDS (DPoser's optimization) | 9 | 0.6-1.3 | about 10-20× |

Inpainting is about ten times faster than D-Flow on the test; the slowest inpainting point (t99, the most resampling)
only 3-4 times. PoseD-Flow's own D-Flow solves its ODE with 200 network calls instead of 50, so against it the gap is
larger; its time is not in these logs. The "1.4 times faster" of section 10 was measured on the Mac (MPS) before the
IGSO(3) noise moved to the GPU, when inpainting was bound by the noise drawn on the CPU, for one heavy variant against
D-Flow; it does not hold for the test points.

## What it does not show

- One training seed per likelihood run, one Pareto study (on the old 512×4 std model); SDS settings were not retuned
  for the new models.
- D-Flow with the likelihood models ran with Euler 100 only, not with Heun EDM.
- D-Flow on validation for the likelihood models is complete only for 512×4 (the Mac ran out of memory).
- The test of the Heun rows and of the likelihood rows is a single run each; differences below ~1 mm are not
  resolved.
- Section 15: each front point ran once on the test; the hypervolume depends on the band (RSGM D-Flow against
  PoseD-Flow turns between APD 4.0-8.0 and 4.0-8.5) and on how densely Optuna explored a front; 10 points lose
  0.1-1.1 mm of band mean against the whole validation front, the most on the steep fronts (PoseD-Flow, DPoser
  inpainting). The best of 10 hypervolume is that of the points chosen by the mean, not of a front of best of 10.
- The validation poses are harder than the test poses, so single points move between the splits by several mm
  (section 14); the comparison relies on the fronts.
- One training seed per network; the families pool networks of c10 and full training data.
- Section 17: coverage in one metric (the chordal metric of the joint rotations, not joint positions); the hard
  poses are those of the methods compared, so a distance from the data and a weakness of all methods are not
  separated; only PoseD-Flow's training split, no other datasets.

## Where things are

- Coverage (section 17): `explore/evaluation/data_coverage/coverage.py`, the result in
  `~/data/experiments/data_coverage/coverage.pt` (tables again by `--report-only`), the log `coverage.log` next to it.
- Rows of the test: `~/data/experiments/completion/cache/<row>/{<occlusion>.pt, <occlusion>_metrics.pt}`; the rows of
  `scripts/evaluate_completion.py` and of `explore/evaluation/completion_diagnostics/{sds_test,dflow_test_extra}.py`.
- Optuna: `~/data/experiments/completion/optuna/*.db`, configurations in `config/completion_tuning/*.yaml`;
  `python scripts/tune_completion.py --study <name> --summary`.
- Runs: `~/data/experiments/rsgm_posedflow_*_s10`, `~/data/experiments/full/rsgm_posedflow_*_s10`.
- Diagnostics: `explore/evaluation/completion_diagnostics/` (run from the root of the repository with
  `PYTHONPATH=src`; each script says how in its docstring).
- Inpainting (section 10): `src/hp/models/inpainting.py`; validation rows in
  `~/data/experiments/completion/{inpaint,dflow,baselines}_valid_mac.jsonl` by
  `explore/evaluation/completion_diagnostics/{inpaint_valid,dflow_valid,baselines_valid}.py`; `dposer_inpaint` and
  the DPoser variants on the branch `experimental/dposer-inpainting`.
- Inpainting studies and test (section 11): `~/data/experiments/completion/optuna/rsgm_inpaint_*.db`, configurations
  `config/completion_tuning/rsgm_inpaint_*.yaml`; test rows `RSGM <network>: their c10, inpaint t<trial>` in
  `scripts/evaluate_completion.py` and `explore/evaluation/pose_completion.ipynb`, caches in
  `~/data/experiments/completion/cache/`.
- DPoser c10 studies (section 12): `config/completion_tuning/dposer_c10_{method,inpaint}_mean_apd.yaml`, databases in
  `~/data/experiments/completion/optuna/`; the table by `explore/evaluation/completion_diagnostics/study_fronts.py`.
- PoseD-Flow with a tuned D-Flow (section 14): study `~/data/experiments/completion/optuna/posedflow_dflow_mean_apd.db`
  (`config/completion_tuning/posedflow_dflow_mean_apd.yaml`), test row `PoseD-Flow: their checkpoint, D-Flow t17`;
  `explore/evaluation/completion_diagnostics/test_rows_by_difficulty.py` (the table by difficulty) and
  `dflow_loss_scale.py` (the check of the batch).
- Six fronts on the test (section 15): the rule and the measure in `hp.evaluation.fronts` (`spanning`,
  `hypervolume`, `APD_RANGE`, `TEST_POINTS`, `HV_REFERENCE`); the points by `scripts/select_test_points.py` in
  `config/completion_test_points.yaml`, run by `scripts/evaluate_completion.py --points`; the tables by
  `explore/evaluation/completion_diagnostics/test_points_report.py`; the test view of the fronts in
  `explore/evaluation/completion_fronts.ipynb`.
- Compared rows and hard poses (section 16): `explore/evaluation/pose_completion.ipynb` (sections 4-5, the widget
  "hard for all"), `hard_poses` and `hard_overlap` in `src/hp/visualization/completion_figure.py`.
- Fronts over bands (section 13): `explore/evaluation/completion_diagnostics/plot_study_fronts.py --bands`; all
  validation settings and the test rows on their fronts in `explore/evaluation/completion_fronts.ipynb` (a selector of
  the split).
