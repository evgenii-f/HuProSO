# ACE energy, 2026-11-27: a compact pairwise matrix Fisher basis; the gap is made at small noise

Branch `experimental/ace-descriptors`. Validation numbers only; no test was run. Part of the ACE line: [overview](README.md).

## Summary

- **Question.** The linear energy P+L (products of local rotations of all pairs + joint limits, 20,354 features) is a
  pairwise matrix Fisher (Bingham–Fisher) energy on SO(3)^19 plus the ACE descriptors and SO(3) harmonics. Which of its
  parts are needed, and at which noise levels is its completion gap to the network made?
- **A compact basis keeps generation.** The products, the one-body harmonics D^{1,2,3}(x_j) and the joint limits,
  without the descriptors and the pair harmonics (15,428 features), reach Wasserstein 16.14° and precision 14.00°
  against 16.04° and 13.99° for P+L (rule: at most 0.15° worse; passed). Its score is twice as fast on the Mac CPU
  (0.71 against 1.46 s per 2000 poses).
- **D^1 alone is not enough.** The pure pairwise matrix Fisher form (products + D^1 + limits, 14,022 features) gives
  18.81° and 17.65°: the one-body harmonics of degree 2 and 3 carry the difference, at large noise (s ≈ 0.4–1.7).
- **In completion the compact basis is slightly worse.** Inpainting mean 98.57 against 97.49 mm (paired +1.09
  [−0.54, +2.66]), best of 10 55.19 against 53.52 mm (+1.67). By the rule fixed before the run (both at most 1.5 mm
  worse) it fails on best of 10. P+L stays the linear model for completion.
- **92% of the completion gap is made at s < 0.04.** With the compact energy only at s < 0.04 (the network above) the
  mean is 97.54 mm, a gap of +12.31 mm to the network against +13.35 mm with the energy everywhere; with the energy
  only at s ≥ 0.04 it is 85.43 mm (+0.20 [−0.30, +0.70]). The energy of nearly clean poses is what to fix.
- **Hard joint limits would be worth at most about 1.5 mm** (an estimate; no hard limit was built into the energy,
  whose limits stay the soft joint densities of P+L). 4–6% of P+L's completed poses have a joint outside the training
  ranges, and they are 65–100 mm worse than the rest; dropping them after the fact gives 96.04 mm (P+L), 94.78 mm (MLP
  on P+L), 85.74 mm (network).
- **A ridge growing with the tree distance of a pair does nothing** (α = 1: Wasserstein 16.05° against 16.04°).

## Setup

- **Fits** (`fit.py`, 16 bins, 20,000 samples per bin, ridge 1e-3, `--single`, joint density of 5000 poses with the
  19 bandwidths of P+L): on workstation2 (cuda:1) for the two new bases, on the Mac CPU for the hop ridge (about 40
  min). New options:
  - `--no-descriptors` (53644b9): the harmonic basis without the descriptors of `AceFeatures`; `embedding.owners`
    reads such fits (35b6206);
  - `--hop-ridge ALPHA` (7448f06): the ridge of the 81 products of a pair times h_ik^ALPHA, h_ik its number of edges
    along the tree (1 for parent and child).
- **Rows:**
  - P+L: `energy_b_rich_products_jd_r0.001.pt` (`--wigner 1 2 3 --pairs 1 2 3 --products`, 20,354 features);
  - D^1 + products + limits: `energy_b_min_bf.pt` (`--wigner 1 --products --no-descriptors`, 14,022);
  - compact: `energy_b_min_bf_w123.pt` (`--wigner 1 2 3 --products --no-descriptors`, 15,428);
  - hop ridge: `energy_b_rich_products_jd_hop1.pt` (P+L with `--hop-ridge 1`).
- **Loss:** `fit.py`'s evaluation, 5000 validation poses, one draw per pose, the same draws for every row.
- **Generation:** `generate.py`, 1000 samples, batch 500, the noising and sampler of `fact_RDW` s10; noise of the
  harness about 0.15°. All rows of a table are from one run.
- **Completion:** `complete.py`, the first 200 validation poses, K = 10, four occlusions, inpainting at t81 (DPS
  guidance 0.0493, `replace=False`), on workstation1. Noise bands with `--band` (the energy inside the band, the
  network outside). Paired differences with bootstrap 95% intervals over the poses: `diagnose.py completion`.
- **Joint ranges:** `range_hypotheses.py` on the saved completed poses (rows made with `--rank`): a hypothesis is outside
  if a coordinate of a joint (twist, two of swing) lies outside the 0.05–99.95% quantiles of 400,000 training poses.
- **Rules**, written before the runs: generation at most 0.15° worse than P+L in Wasserstein and precision; completion
  at most 1.5 mm worse in mean and in best of 10; bands: if the energy only at s < 0.04 makes at least 75% of the
  gap, the small noise is to be fixed, else the middle.

## Results: basis

Validation, generation 1000 samples; loss 5000 poses (`generate_min_bf_w123.log`, `generate_hop1.log`, the `.json`
of the fits):

| model | features | Wasserstein (°) | recall (°) | precision (°) | FID_val | APD (cm) | val. loss | loss, 10 bins s < 0.04 |
|---|---|---|---|---|---|---|---|---|
| network `fact_RDW` s10 | 4.4M weights | 14.93 | 13.43 | 12.86 | 0.0390 | 14.55 | 5.951 | 43.61 |
| P+L | 20,354 | **16.04** | **14.23** | 13.99 | **0.0463** | 15.13 | 6.336 | 45.50 |
| products + D^1 + limits | 14,022 | 18.81 | 14.88 | 17.65 | 0.0775 | 19.06 | 7.169 | 46.25 |
| **compact: products + D^{1,2,3} + limits** | 15,428 | 16.14 | 14.39 | 14.00 | 0.0520 | 14.90 | 6.394 | 45.89 |
| P+L, hop ridge α = 1 | 20,354 | 16.05 | 14.25 | **13.98** | 0.0464 | 15.11 | **6.314** | **45.24** |

- The descriptors and the pair harmonics D^l(G_j), D^l(G_i^T G_k) are not needed in generation; the one-body
  harmonics of degree 2 and 3 are. Without them the loss of the three bins at s ≈ 0.2–1.7 rises from 6.21 / 3.88 / 1.85
  to 6.86 / 5.90 / 3.33, and the samples spread (APD 19.06 cm against 15.1).
- The compact basis loses 0.39 of loss at s < 0.04, where completion is decided (below).
- The hop ridge lowers the loss a little at small noise but not the generation, as ridge, bins and smoothing before
  ([products and joint limits](2026-11-25_products_and_joint_limits.md)).
- Speed (`complete.py --time-only`, Mac CPU, one score call with its backward, 2000 poses): P+L 1.46 s, compact 0.71 s.
  Sampling 1000 poses on the Mac: 1.1 against 0.7 min.

## Results: completion and noise bands

Inpainting t81, 200 validation poses, K = 10, MPVPE in mm (`diagnose_completion_compact.log`):

| row | mean | best of 10 | > 200 mm | left leg | legs | arms | trunk | gap to network |
|---|---|---|---|---|---|---|---|---|
| network `fact_RDW` s10 | **85.23** | **47.32** | 5.1% | 83.1 | 102.7 | 113.0 | 42.2 | 0 |
| P+L | 97.49 | 53.52 | 7.9% | 100.0 | 117.7 | 122.7 | 49.6 | +12.26 [+9.89, +14.75] |
| MLP K = 8 on P+L | 95.71 | 51.14 | 7.4% | 97.2 | 115.6 | 121.8 | 48.2 | +10.48 [+8.37, +12.96] |
| compact | 98.57 | 55.19 | 7.2% | 100.8 | 120.3 | 121.7 | 51.4 | +13.35 [+11.38, +15.28] |
| compact at s < 0.04, network above | 97.54 | 54.53 | 6.9% | 100.4 | 119.8 | 118.5 | 51.5 | +12.31 [+10.46, +14.16] |
| compact at s ≥ 0.04, network below | 85.43 | 47.36 | 5.2% | 82.7 | 102.8 | 114.0 | 42.2 | +0.20 [−0.30, +0.70] |

Paired differences (mean over the occlusions, then per occlusion: left leg, legs, arms, trunk):

| difference | mean | left leg | legs | arms | trunk |
|---|---|---|---|---|---|
| compact − P+L | +1.09 [−0.54, +2.66] | +0.83 | +2.63 [+0.05, +5.05] | −0.98 | +1.87 [−0.18, +3.80] |
| compact at s < 0.04 − network | +12.31 [+10.46, +14.16] | +17.33 | +17.08 | +5.58 | +9.26 |
| compact at s ≥ 0.04 − network | +0.20 [−0.30, +0.70] | −0.33 | +0.12 | +0.99 | +0.02 |

- **Compact against P+L:** mean +1.09 mm (within the rule), best of 10 +1.67 mm (55.19 − 53.52, outside the rule by
  0.17 mm); the legs and the trunk lose most. It fails the completion rule.
- **Bands:** the energy only at s < 0.04 makes 12.31 / 13.35 = 92% of the gap (rule: ≥ 75%), in every occlusion; at
  s ≥ 0.04 the linear energy is as good as the network (+0.20 mm). The same as the old energy of 2026-11-24 (122.96 of
  128.16 mm at s < 0.04, [completion and plateau](2026-11-26_completion_and_plateau.md)), and in line with the energy
  of the truth: the energies rank hypotheses well at s ≥ 0.02 and badly at s = 0.002
  ([truth energy](2026-11-26_completion_tuning_and_truth_energy.md)).
- The gap by distance to train and by knee flexion is that of P+L: far third +18.5 mm against 10–12 in the others;
  left knee bent +24.0 mm (`diagnose_completion_compact.log`).

## Results: hypotheses outside the joint ranges

The same completion files, inpainting t81, 200 validation poses, K = 10 (`range_hypotheses.log`):

| prior | hypotheses outside | error inside (mm) | error outside (mm) | mean (mm) | mean, outside dropped (mm) |
|---|---|---|---|---|---|
| network | 1.6–3.4% | 41.7–115.5 | 46.2–140.5 | 85.84 | 85.74 |
| P+L | 3.7–5.8% | 46.1–117.5 | 106.5–219.0 | 97.49 | **96.04** |
| MLP K = 8 on P+L | 3.8–5.0% | 45.4–118.9 | 102.7–209.7 | 95.71 | **94.78** |

(Ranges over the four occlusions. The network row is the run with `--rank`, 85.84 mm, not the 85.23 mm run, whose
poses were not saved.)

- The hypotheses outside are much worse, but few: dropping them, without the ground truth, gains 1.45 mm for P+L and
  0.93 mm for the MLP, about 12% of the gap. A barrier at the limits in the energy could gain about as much and no more.
- Most often outside: SMPL joints 16 (left shoulder), 3 (spine 1), 21 and 20 (wrists), 18 (left elbow); rarely the
  knees. The knee gap is under-bending, inside the ranges.

## Conclusions

1. **The working form of the linear energy is a pairwise Bingham–Fisher energy with richer one-body terms:**
   E_t(x) = v^T W_t v + Σ_j Σ_{l≤3} ⟨C^l_{j,t}, D^l(x_j)⟩ − σ_t Σ_j log p_{t,j}(x_j), v = vec(x) ∈ R^171. The ACE
   descriptors and the pair harmonics are not needed in generation; the pure matrix Fisher one-body term (D^1) is not
   enough. In completion the compact form is 1–2 mm behind P+L.
2. **The linear energy is as good as the network at s ≥ 0.04** in completion; the whole gap is the energy of nearly
   clean poses. More bins at small noise do not address it (32 bins were within the noise of 16 before); the error is in
   what the basis does far from the data.
3. **Next** (not run): fix the energy at s < 0.04 only: weighting rare training poses in the fit of those bins, or a
   term over whole poses (a mixture of matrix Fisher kernels of complete poses) that keeps the energy near the data;
   hard joint limits (not built) would be worth at most about 1.5 mm. The completion of a candidate is judged against P+L on the same
   200 poses.

## What it does not show

- One fit per basis (one seed of the samples), one ridge (1e-3, chosen for P+L); the compact basis was not tuned.
- 200 poses for completion; the best of 10 has no bootstrap interval in the kept scripts (only its difference of means).
- The bands are a hybrid of the energy and the network, not a model; they locate the error, they do not fix it.
- The joint-range check drops hypotheses after the fact; a barrier in the energy would also change the sampling.
- The pseudo-likelihood fit of the pairwise D^1 energy on clean poses (`pseudo_lik.py`, another session) is not part
  of this file.

## Where things are

- Code (branch `experimental/ace-descriptors`): `explore/ace_energy/fit.py` (`--no-descriptors`, `--hop-ridge`),
  `embedding.py` (`owners`), `diagnose.py` (rows "compact…"), `range_hypotheses.py`, `generate.py`, `complete.py`.
- Fits in `~/data/experiments/ace/`: `energy_b_min_bf.{pt,json}` and `min_bf.log`, `energy_b_min_bf_w123.{pt,json}` and
  `min_bf_w123.log` (workstation2, copied to the Mac; also on workstation1), `energy_b_rich_products_jd_hop1.{pt,json}`
  and `rich_products_jd_hop1.log` (Mac).
- Harness: `generate_min_bf.log`, `generate_min_bf_w123.log`, `generate_hop1.log`.
- Completion: `diag6/{inpaint,band_low,band_high}_min_bf_w123.pt`, log `diag6.log` (workstation1, copied to the Mac);
  comparisons `diagnose_completion_compact.log`; joint ranges `range_hypotheses.log`.
