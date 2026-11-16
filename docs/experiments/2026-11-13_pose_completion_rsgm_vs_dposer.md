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

- **2026-11-15, inpainting (validation, section 10)**: completing by the whole reverse process of RSGM, with DPS
  guidance and RePaint resampling, beats D-Flow on all three metrics at once (100 validation poses: 86.68 / 48.19 mm,
  APD 6.42 cm against 90.88 / 52.40 / 6.08), 1.4 times faster; 3.4 mm behind DPoser in the mean, ahead in best of 10
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
  2 mm lower in the mean at a lower APD (86.17 / 49.99 / 6.02 with 3 × 200 steps). Test of the chosen points pending.

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
A fifth study, `rsgm_inpaint_mean_apd_search_lik_full` (1536×6, likelihood weighting, full data), was at 88 of 120
trials when this was written; its rule point so far is trial 71 (400 steps, ζ 0.040, r 5: 90.70 / 50.62 / 6.65), not
better than residual, and it was not run on the test.

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
| RSGM 1536×6 lik full, inpainting (other session) | 88 | 90.70 | 90.70 | 90.70 | 90.70 | 95.15 | t71: 90.70 / 50.62 / 6.65 |
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

**Test.** The rows of the chosen points are in `scripts/evaluate_completion.py`: RSGM t52, t75, t81, t118 (another
session) and DPoser c10 inpainting t68, DPoser's method t28 and, outside the rule, t3; DPoser's method with their
defaults was tested before (row "DPoser: their c10, RSGM budget, DPoser method"). The DPoser rows are not run yet.

**What it does not show.** DPoser c10 was trained with a recipe made for 200k steps; one training seed each; the
compared settings differ in network calls; the 1536×6 lik full study is at 88 of 120 trials. Numbers are validation,
on the poses the settings were chosen on.

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
