# Pose completion: RSGM against PoseD-Flow and DPoser

What we did between 2026-11-13 and 2026-11-15 on branch `experimental/completion`: RSGM as the prior of pose completion
in the protocol of PoseD-Flow, two completion procedures (D-Flow and DPoser's score distillation), why RSGM loses to
DPoser in DPoser's procedure, and whether training RSGM with the likelihood weighting or a better ODE solver closes the
gap.

## Summary

All numbers on the 5000 test poses of PoseD-Flow, K = 10, MPVPE in mm averaged over the four occlusions.

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

## What it does not show

- One training seed per likelihood run, one Pareto study (on the old 512×4 std model); SDS settings were not retuned
  for the new models.
- D-Flow with the likelihood models ran with Euler 100 only, not with Heun EDM.
- D-Flow on validation for the likelihood models is complete only for 512×4 (the Mac ran out of memory).
- The test of the Heun rows and of the likelihood rows is a single run each; differences below ~1 mm are not
  resolved.

## Where things are

- Rows of the test: `~/data/experiments/completion/cache/<row>/{<occlusion>.pt, <occlusion>_metrics.pt}`; the rows of
  `scripts/evaluate_completion.py` and of `explore/evaluation/completion_diagnostics/{sds_test,dflow_test_extra}.py`.
- Optuna: `~/data/experiments/completion/optuna/*.db`, configurations in `config/completion_tuning/*.yaml`;
  `python scripts/tune_completion.py --study <name> --summary`.
- Runs: `~/data/experiments/rsgm_posedflow_*_s10`, `~/data/experiments/full/rsgm_posedflow_*_s10`.
- Diagnostics: `explore/evaluation/completion_diagnostics/` (run from the root of the repository with
  `PYTHONPATH=src`; each script says how in its docstring).
