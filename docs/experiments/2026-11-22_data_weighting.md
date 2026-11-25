# Sampling rare training poses more often: a trade, not a gain (D-Flow aside)

RSGM is trained on the poses of c10 drawn uniformly. Would drawing rare poses more often (a) improve pose completion,
in particular with the cheap methods that use the score directly (inpainting, SDS), (b) without harming generation?
And could another choice of the training poses (hard examples, D-optimality, active learning) do better than rarity?

Branch `experimental/data-weighting`. One network throughout: `fact_RDW` (512 x 4 with residual connections, dropout
0.1, width 1024; c10, 30k iterations of 4096; 4.4M parameters).

## Summary

- **No free gain for inpainting.** At an unchanged generation quality there is no gain. The Optuna study of the
  weights (24 trials, validation) shows a trade: the stronger the weights, the better best of 10 and the worse
  Wasserstein and precision. At the limit fixed before the study (Wasserstein at most 0.5 deg worse than β = 0) the
  best trial (β 0.68) gains 0.37 mm in best of 10 and loses 0.42 deg in Wasserstein and 0.83 deg in precision against
  the mean of the near-uniform trials; the mean MPVPE does not improve (+0.35 mm).
- **On the test (rule fixed before):** β 3.3 is better than β 0 with D-Flow (band mean 91.53 → 87.76 mm, hypervolume
  of best of 10 75.76 → 81.42), with no difference with inpainting (90.05 → 89.16 mm) and SDS (96.67 → 96.10 mm).
- **Generation pays.** β 3.3 raises Wasserstein from 14.52 to 17.31 deg, precision from 11.83 to 15.08 deg and
  FID_val from 0.036 to 0.074 (two seeds agree). Weak weights (β 0.6-0.9) improve FID_val (0.030) and the
  log-likelihood (+0.4 to +0.7 nats) while Wasserstein and precision get worse.
- **Hard poses are largely common and largely rare.** D-Flow, inpainting and SDS agree on the poses (Spearman of the
  mean rank 0.84-0.90), and rarity explains most of it (0.71-0.77). Rare training poses are underfitted: the weights
  reduce the inpainting error of the rarest training quintile by 10.8 mm, of the rarest validation quintile by 4.4 mm,
  but raise that of the most common training quintile by 21.2 mm.
- **Other choices of the poses are not better than rarity on these data:** leverage and the MaxVol grade γ
  (D-optimality) predict the error worse than rarity; D-optimal weights collapse to 1% of the poses; the loss of the
  model and −log p carry a little beyond rarity; the disagreement of seeds almost nothing.
- **Decision.** One model for everything: β = 0 (uniform sampling). Weighting is closed for inpainting and SDS.
  **D-Flow may profit from it; that question is left open** (a separate model, at the price of generation) and not
  pursued now. Section 17 of the completion report is corrected (below).

## Setup

### Weights

Rarity $r_i$ of a c10 training pose: the RMS angle per joint (deg) to its 5th nearest pose among every tenth c10 pose
(186k), in the chordal metric over the 19 learned joints (`explore/evaluation/data_weighting/rarity.py`). Median 7.2
deg, 90% 14.4, 99% 21.4. The sampling probability is proportional to w_i = clip((r_i / median r)^β, w_min, w_max).
Effective sample size (ESS) relative to the number of poses and the share of the 10% heaviest poses in a batch at
w_min 0.1, w_max 10:

| β | 0 | 0.6 | 1.163 | 1.8 | 2.518 | 3.3 | 4.2 |
|---|---|---|---|---|---|---|---|
| ESS | 1.00 | 0.89 | 0.70 | 0.50 | 0.40 | 0.39 | 0.40 |
| share of the rarest 10% | 0.10 | 0.17 | 0.24 | 0.33 | 0.41 | 0.38 | 0.33 |

### Evaluation

- Completion: the protocol of the [completion report](2026-11-13_pose_completion_rsgm_vs_dposer.md), K = 10, four
  occlusions. Fronts of mean MPVPE and APD by Optuna (120 trials inpainting and SDS, 60 D-Flow) on 1000 validation
  poses, the same search space for β 0 and 3.3; points for the test by the rule of `scripts/select_test_points.py`.
  Hypervolume (HV) over APD 4.0-8.5 cm, reference mean 115 mm and best of 10 65 mm; band mean = 115 − HV / 4.5.
- Generation: the harness of the factorial study, the model's own sampler, validation poses: Wasserstein, recall and
  precision (deg, lower is better), FID to the validation and to the training poses (FID_val, FID_own), log-likelihood
  of 1000 validation poses (nats).
- The choice of β: the grid below at the fixed inpainting point, and D-Flow; β 3.3 was taken for the completion fronts
  as the lowest D-Flow mean of 300 validation poses.
- Rules, all fixed beforehand. First grid (inpainting, fixed point): a weighted run is better if its mean on the
  farthest quintile is lower by 2 mm or more and its mean on all poses is not higher by more than 0.5 mm. Test: β 3.3
  is better in a method if the band mean is lower by 1 mm or more and the HV of best of 10 is not lower.
  Optuna study: on the front of Wasserstein and HV of best of 10, the trial with the best HV among those whose
  Wasserstein is at most 0.5 deg worse than that of β 0 (14.52).

## Results

### 1. The β grid (validation, inpainting at one fixed setting, 1000 poses)

Q1-Q5: quintiles of the distance of the hidden part to the nearest training pose.

| β | mean (mm) | best of 10 (mm) | APD (cm) | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|---|---|
| 0 | 87.40 | 48.58 | 6.30 | 58.04 | 68.97 | 81.95 | 97.15 | 130.89 |
| 0.6 | 87.89 | 48.46 | 6.40 | 59.45 | 68.61 | 83.81 | 97.92 | 129.66 |
| 1.163 | 87.47 | 47.90 | 6.46 | 59.82 | 69.64 | 83.03 | 97.16 | 127.71 |
| 1.8 | 87.95 | 47.80 | 6.60 | 61.59 | 70.32 | 82.86 | 97.99 | 126.99 |
| 2.518 | 88.72 | 48.06 | 6.69 | 62.21 | 71.84 | 84.59 | 97.80 | 127.15 |
| 3.3 | 88.60 | 47.85 | 6.71 | 63.80 | 72.04 | 83.76 | 97.52 | 125.86 |
| 4.2 | 88.53 | 47.80 | 6.63 | 63.55 | 71.36 | 83.30 | 97.13 | 127.28 |

The farthest quintile gains up to 5.0 mm (β 3.3), the nearest loses up to 5.8 mm. By the rule only β 1.163 passes
(far quintile −3.19 mm, all +0.07 mm), by a small margin on one seed. With D-Flow the mean of 300 validation poses
falls from 91.00 mm (β 0) to 88.32 (0.6), 86.51 (1.163), 85.78 (1.8), 84.98 (2.518) and 84.36 (3.3), 85.11 (4.2); APD
falls too (6.22 to 5.14 cm at β 3.3). With SDS (1000 poses) the mean is level (102.4-103.2 mm) and best of 10 improves
from 64.00 to 60.76 mm at β 3.3.

### 2. Generation (validation)

| model | W (deg) | recall (deg) | precision (deg) | FID_val | FID_own | log-likelihood (nats) |
|---|---|---|---|---|---|---|
| β 0, seed 10 | 14.52 | 11.86 | 11.83 | 0.0364 | 0.0086 | 137.07 |
| β 0, seed 11 | 14.55 | 11.87 | 11.88 | 0.0363 | 0.0088 | 136.94 |
| β 0.6, seed 10 | 14.92 | 11.88 | 12.68 | 0.0295 | 0.0118 | 137.51 |
| β 1.163, seed 10 (Mac CPU) | 15.48 | 11.92 | 13.58 | 0.0336 | 0.0248 | 137.71 |
| β 3.3, seed 10 | 17.31 | 12.25 | 15.08 | 0.0737 | 0.0832 | 137.29 |
| β 3.3, seed 11 | 17.37 | 12.26 | 15.12 | 0.0735 | 0.0830 | 137.25 |

Own sampler; with Heun EDM (50 calls) the same order. The β 0 row reproduced on the Mac CPU gives 14.51 / 11.87 /
0.0374 (device noise about 0.01-0.04 deg).

### 3. Test (5000 poses, K = 10)

Test points chosen on validation by the rule. "DPoser", "RSGM" and "PoseD-Flow" are the fronts of the completion
report (other networks) for reference, read from the same cache.

| front | band mean, test (mm) | HV best of 10, test | band mean, validation (mm) | HV best of 10, validation |
|---|---|---|---|---|
| D-Flow, β 0 | 91.53 | 75.76 | 94.01 | 63.61 |
| D-Flow, β 3.3 | **87.76** | **81.42** | 90.91 | 70.40 |
| inpainting, β 0 | 90.05 | 76.43 | 91.39 | 71.41 |
| inpainting, β 3.3 | 89.16 | 83.25 | 91.00 | 75.53 |
| SDS, β 0 | 96.67 | 49.30 | 95.48 | 48.86 |
| SDS, β 3.3 | 96.10 | 53.75 | 93.33 | 55.47 |
| reference: DPoser, its method | 86.85 | 82.04 | 89.24 | 71.46 |
| reference: DPoser, inpainting | 91.71 | 72.73 | 91.10 | 71.60 |
| reference: RSGM, inpainting | 90.05 | 74.78 | 92.40 | 66.27 |
| reference: RSGM, D-Flow | 92.91 | 71.64 | 94.03 | 61.44 |
| reference: PoseD-Flow, D-Flow | 93.28 | 59.11 | 95.82 | 49.83 |

By the rule D-Flow is better with β 3.3 (−3.77 mm); inpainting (−0.89 mm) and SDS (−0.57 mm) show no difference in the
band mean, although the HV of best of 10 is higher in both. A test point of 5000 poses costs 40-59 minutes with D-Flow
(mean 50 at β 0, 57 at β 3.3), 1.5-15 with inpainting (mean 6.5) and 2.4-5.1 with SDS (mean 4.6): D-Flow is about 8
times as expensive as inpainting (`test_dflow*.log`, `test_fast.log`).

### 4. Optuna study of the weights (validation)

`scripts/tune_weighting.py config/weighting_tuning/rdw_inpaint.yaml` on workstation2 (GPU 0): 24 trials, seed 10, each
trains `fact_RDW` with its own β, w_min and w_max (8.8 minutes; six models trained before: 5.5), then the generation
harness and inpainting at six fixed settings on 500 validation poses. Objectives: Wasserstein (min) and HV of best of
10 (max).

| trial | β | w_min | w_max | ESS | W (deg) | HV best of 10 | best of 10 (mm) | mean (mm) | FID_val | precision (deg) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (known) | 0 | 0.1 | 10 | 1.00 | 14.52 | 71.13 | 49.37 | 92.67 | 0.0365 | 11.83 |
| 1 (known) | 0.6 | 0.1 | 10 | 0.89 | 14.92 | 72.50 | 49.21 | 93.21 | 0.0295 | 12.68 |
| **12 (rule)** | 0.68 | 0.25 | 8.7 | 0.87 | 15.00 | 73.46 | 49.04 | 93.29 | 0.0297 | 12.83 |
| 6 | 0.78 | 0.33 | 4.9 | 0.84 | 15.12 | 76.57 | 48.52 | 93.19 | 0.0300 | 12.99 |
| 20 | 0.86 | 0.41 | 3.4 | 0.82 | 15.18 | 74.65 | 48.76 | 92.66 | 0.0299 | 13.05 |
| 14 | 1.27 | 0.15 | 6.4 | 0.67 | 15.75 | 76.77 | 48.57 | 93.60 | 0.0376 | 13.73 |
| 21 | 3.46 | 0.13 | 7.4 | 0.44 | 16.84 | 78.55 | 48.42 | 94.75 | 0.0644 | 14.63 |

The rule picks trial 12. Eight trials are practically uniform (β ≤ 0.21, Wasserstein below 14.65); their HV is 71.26
on average with a standard deviation of 0.95, the noise of one trial. Against this mean trial 12 is +0.42 deg in W,
+0.83 deg in precision, −0.0047 in FID_val, −0.37 mm in best of 10, +0.35 mm in the mean, +2.2 in HV (2.3 standard
deviations of one trial). HV against W is a single curve (Spearman 0.91; residual standard deviation 0.94, the noise):
w_min and w_max do not matter beyond the strength of the weights. Trial 6 lies 2.5 standard deviations above the
curve and outside the limit of Wasserstein (15.12 against 15.02): taken as noise, the rule is not moved for it.
An earlier rule of this line (Wasserstein at most +0.7, precision at most +0.8 deg worse, FID_val at most 0.045) was
stricter in precision: trial 12 (+1.00 deg against the β 0 trial) fails it, as does β 0.6 (+0.85 deg).

### 5. Motion denoising (validation, 100 chunks, settings tuned for another RSGM)

| seed, noise | MPJPE β 0 → 3.3 (mm) | MPVPE β 0 → 3.3 (mm) |
|---|---|---|
| s10, 40 mm | 18.67 → 18.54 | 37.83 → 37.35 |
| s10, 100 mm | 30.86 → 30.85 | 51.35 → 50.74 |
| s11, 40 mm | 18.61 → 18.58 | 37.87 → 37.55 |
| s11, 100 mm | 30.85 → 30.83 | 51.39 → 50.78 |

MPJPE is level (at most −0.12 mm), MPVPE 0.3-0.6 mm better in both seeds. Descriptive: the settings are not tuned for
these models and no test was run.

### 6. Which poses are hard (validation, descriptive)

- D-Flow, inpainting and SDS on 200 validation poses (β 0): Spearman of the mean MPVPE per occlusion 0.63-0.87, of the
  mean rank over the four occlusions 0.84-0.90, beyond rarity 0.62-0.74. The hardest 10% lie to 85-90% among the
  rarest 30% of the poses.
- The rank of inpainting (1000 poses): Spearman with rarity 0.77, with −log p 0.81; rarity and −log p 0.90. Errors of
  two occlusions correlate 0.28-0.63; only 6 of 1000 poses are in the hardest 10% of all four occlusions.
- Motion denoising (SDS): the MPJPE of a frame has Spearman 0.34 (40 mm) and 0.39 (100 mm) with its rarity; 16.2 to
  20.8 mm over the quintiles at 40 mm. Denoising and completion use different poses, so their difficulty cannot be
  compared pose by pose. Generation has no per-pose difficulty: it compares distributions.
- The error correlates with APD (0.46-0.79 between methods): part of the difficulty is the ambiguity of the pose,
  which weights cannot remove.

### 7. Training poses

Inpainting (the fixed point of the grid) on 300 training poses, 60 per quintile of rarity, against the 1000
validation poses grouped by quintile of their own rarity. Mean MPVPE in mm, 95% intervals over the poses, one seed:

| rarity quintile (common → rare) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| train, β 0 | 27.7 | 39.0 | 46.4 | 62.4 | 91.9 |
| train, β 3.3 − β 0 | +21.2 ± 2.4 | +12.7 ± 2.5 | +9.4 ± 2.5 | +1.6 ± 1.9 | −10.8 ± 3.5 |
| validation, β 0 | 56.1 | 71.5 | 83.1 | 96.1 | 130.2 |
| validation, β 3.3 − β 0 | +6.2 ± 1.3 | +3.4 ± 1.2 | +1.8 ± 1.0 | −1.0 ± 1.4 | −4.4 ± 2.0 |

The rarest training poses have an error 3.3 times that of the most common ones; the weights reduce it, on the common
poses they raise it more, on average +6.8 ± 1.7 mm over the training poses and +1.2 ± 0.7 mm over the validation ones
(best of 10: −0.7 ± 0.7 mm). Validation is shifted to rare poses: 560 of its 1000 poses fall into the rarest quintile
of the training rarity and none into the two commonest (median rarity 12.5 against 7.2 deg), which is why the weights
cost less there. At the same rarity validation is still harder: 85.6 against 72.9 mm at 11-15 deg. On the training
poses the error correlates with APD at 0.85-0.93, with rarity at 0.61-0.76.

### 8. Other ways to choose the poses

Spearman with the error of inpainting. Beyond rarity: after regressing the ranks of the rarity out of both.

| signal | 1000 validation poses | beyond rarity | common difficulty of D-Flow, inpainting, SDS (200 poses) | beyond rarity | 300 training poses | vs rarity (valid / train) |
|---|---|---|---|---|---|---|
| rarity | 0.80 | - | 0.77 | - | 0.83 | 1 / 1 |
| loss of β 0, all noise levels | 0.79 | 0.24 | 0.77 | 0.24 | 0.85 | 0.92 / 0.86 |
| loss of β 0, low noise | 0.77 | 0.23 | 0.75 | 0.27 | 0.87 | 0.88 / 0.88 |
| disagreement of three seeds of RSGM 512x4 | 0.68 | 0.03 | 0.65 | 0.08 | 0.49 | 0.84 / 0.51 |
| −log p of β 0 | 0.82 | 0.36 | 0.81 | 0.46 | not measured | 0.90 / - |

The loss and −log p carry some difficulty beyond rarity, but whether that part is reducible was not tested: it
contains the ambiguity of the pose. A model trained on loss weights was not trained.

**D-optimality** (the active set of PACE), linear features of the rotations (entries of D^1: 172; of D^1 and D^2: 647;
plus the pairs of D^1: 917), reference set c100. Error rank by mean over occlusions:

| score | validation | training | beyond rarity (validation) |
|---|---|---|---|
| rarity | 0.77 | 0.83 | - |
| leverage, the three feature sets | 0.70-0.74 | 0.70-0.72 | 0.03-0.13 |
| MaxVol grade γ, the three feature sets | 0.48-0.53 | 0.40-0.50 | −0.03 to 0.06 |

The validation poses are inside the hull of the data for these features: γ ≤ 1.05 for nearly all of them, at most
1.3. The D-optimal sampling weights (multiplicative updates) collapse: ESS 0.08-0.16 after one step and 0.003
to 0.012 after 20, concentrated on outliers (largest leverage 568-806 times the median). At ESS 0.70 the leverage
weights are as strong as the rarity weights (heaviest 10% 0.25 against 0.24 of the weight) but select different poses
(Spearman 0.64-0.67), not better ones. D-optimality is a criterion for a linear model with expensive labels; here the
poses are free and the network is not linear.

## Conclusions

1. Sampling weights move the model along a trade between generation and best-of-10 completion; they do not improve
   both. For inpainting and SDS they are not worth the loss in generation.
2. The gain of D-Flow on the test is real by the rule, at the price of generation and of a separate model. **It is
   left open**: whether it is worth a second model for D-Flow is not decided here, and no further work is planned now.
3. The weights have one effective knob, the strength (ESS); their shape, the rarity's floor and cap do not matter
   within the noise.
4. Rare training poses are underfitted and the weights reduce their error, but part of their difficulty is
   extrapolation (validation harder at the same rarity) and ambiguity (the error follows APD).
5. Choosing poses by another criterion (hard examples, D-optimality, disagreement) has no support in these data
   beyond rarity; not pursued.
6. **Correction of the completion report, section 17.** It concludes that reweighting the training poses has nothing
   to add for completion because hard poses are extrapolation. The test contradicts this for D-Flow, and the training
   poses show that rare poses are underfitted as well: the statement holds for inpainting and SDS in effect (no
   gain without a loss elsewhere), not as a reason in itself.

## What it does not show

- One training seed per weighted model (two for β 0 and 3.3 in generation and denoising); the Optuna study is one seed
  and 500 poses. A gain of 0.4 mm in best of 10 is at the limit of its noise; it was not confirmed on new seeds.
- The study searches inpainting at six fixed settings; weaker weights with tuned settings were not tested.
- The D-Flow gain on the test is one model pair and one β (3.3, chosen on validation D-Flow).
- The difficulty of poses, the training poses and the signals are descriptive, on validation and training poses; a
  gain of the weights is that of rarity weights by construction, so it cannot rank other signals.
- The D-optimality check uses linear features of the rotations of the whole pose, not the ACE descriptors and not the
  hidden part.
- The seeds' disagreement uses RSGM 512x4 (c10), not `fact_RDW`.

## Where things are

- Weights: `~/data/experiments/data_weighting/{rarity.pt,weights_*.pt,weights.json}`,
  `explore/evaluation/data_weighting/rarity.py`; β grid: `evaluate.py` there, results in
  `~/data/experiments/data_weighting/eval/`, `beta_eval2.log`.
- Runs: `~/data/experiments/factorial/rsgm_posedflow_fact_RDW_s10` (β 0), `~/data/experiments/data_weighting/rsgm_posedflow_rdw_*_s10`
  (β 0.6-3.3), seed 11 of β 3.3 `~/data/experiments/pairs_control/rsgm_posedflow_rdw_b3.3_s11`.
- Generation: `~/data/experiments/data_weighting/harness_*.jsonl`, `~/data/experiments/pairs_extra/harness_b{0,3p3}.jsonl`.
  Denoising: `~/data/experiments/pairs_extra/denoising_{b0,b3p3}/results.jsonl`.
- Completion fronts: `~/data/experiments/completion/optuna/rsgm_*_rdw_{b0,b3p3}.db`, test caches `RDW_*` in
  `~/data/experiments/completion/cache/`, the points `config/completion_test_points.yaml`; notebook
  `explore/evaluation/completion_fronts.ipynb` (families of data weighting).
- Study of the weights: `scripts/tune_weighting.py`, `config/weighting_tuning/rdw_inpaint.yaml`, database
  `~/data/experiments/data_weighting/optuna/rdw_inpaint.db` (workstation2 and Mac), the log `tune_weighting_gpu0.log`
  next to it, runs on workstation2 in `~/data/experiments/data_weighting/optuna/runs/rdw_inpaint/`.
- Sections 5-8, the study and the test table: `explore/evaluation/data_weighting/hard_poses.py` (`denoising`,
  `difficulty`, `train-sample`, `train-report`, `signals`, `dopt`, `study`, `test`; `dopt` needs
  `hp.models.so3_harmonics`, on `dev` from 8f5785b).
