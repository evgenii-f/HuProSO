# ACE energy, 2026-11-25: products of local rotations, joint limits, generation

Branch `experimental/ace-descriptors`. Validation numbers only; no test was run. Part of the ACE line: [overview](README.md).

## Summary

- **2026-11-25: products of local rotations and joint limits**, still linear and in closed form.
  - The products (x_i)_a (x_k)_b of the entries of all 171 pairs of local rotations, with a reference energy made of
    every joint's own noised density, bring generation close to the network: Wasserstein 16.04 against 14.93° (base
    linear energy 18.24°, MLP embedding 16.51°; 1000 samples, validation).
  - Only 5.4% of the generated poses have a joint outside the training range, against 6.7% of the validation poses,
    2.2% of the network's and 16.6% of the old linear energy's.
  - Ridge (1e-4 to 1e-2), noise bins (16 or 32), smoothing over the bins, triples of joints and the distance, angle
    and twist descriptors change the generation by less than about 0.15°, the size of the noise: a plateau near 16.0°.
  - Completion (inpainting) with the new energies was not run. The earlier failure of the linear energy (128.16 mm
    against 85.23 mm) comes from the noise levels below s = 0.04: that band alone adds 37.7 of the 42.9 mm.

Question: what the linear energy lacks, and how near to the network it comes with a richer basis. All numbers are
validation numbers, no test was run.

## Setup

- **Data:** c10, 20,000 training poses per noise bin, 16 bins, the noising of `fact_RDW` s10 as before. Fits in closed
  form with ridge relative to the mean diagonal of GᵀG, on workstation2 (cuda:1; float32 products per batch, float64
  sums and solution).
- **Basis.** On top of the 6503 features (descriptors, D^{1,2,3}(x_j), pair harmonics):
  - *pairs*: D^1(G_iᵀG_k) of the 144 pairs of joints not in `WIGNER_PAIRS` (not parent and child), 1296 features;
  - *products*: (x_i)_a (x_k)_b, 81 for each of the 171 pairs of joints, 13,851 features;
  - *triples*: u_ia u_jb u_kc of the three principal local coordinates of three joints at tree distance at most 3 or 5,
    27 for each of 74 or 289 triples, 1998 or 7803 features.
- **Joint limits** (`joint_density.py`): the reference energy −σ_t Σ_j log p_{t,j}(x_j), with p_{t,j} a mixture of matrix
  Fisher kernels over 5000 training poses, concentration matched to the heat kernel at s + s0_j. The bandwidths s0_j
  (0.001 for Spine2 to 0.025 for the left hand) are chosen on validation poses; one s0 for all joints was chosen
  first (0.005). The expansion is fitted to what the reference leaves. Alone (no expansion) its validation loss is
  8.24, against 26.50 for the zero score (`jd.log`, one s0 for all joints).
- **Per-joint sums.** Gram matrix of products and triples per joint, over the non-zero columns only: the same sums as
  the dense code (checked to 1e-12 for the products), 4.7 min for 16 bins on workstation2 (three ridges) against 19 min
  for the products with the dense code (two ridges).
- **Generation** (`generate.py`): 1000 samples, Euler–Maruyama, 100 steps, the noising of the reference run. Metrics of
  the harness (`hp.evaluation.evaluate`, 2000 or fewer poses, one repeat): Wasserstein, recall, precision against the
  8601 validation poses (degrees per joint), FID_val and APD of PoseD-Flow. Noise: five disjoint sets of 1000 of the
  same samples have a standard deviation of 0.05–0.15° in Wasserstein and precision (a one-off computation, not kept);
  the precision of the network
  itself is 12.86 or 12.60 with another batch size. Differences below about 0.15° are not read.
- **A bug** in `generate.py` (81e51f4) moved the random draws of the first row of every run (recall by about 0.2,
  precision by up to 0.1; Wasserstein not affected). Fixed; the rows below are from the saved samples, scored with the
  same draws (checked by the fork: the order of the models does not matter).
- **Rule** (agreed before the fits): go on with richer bases at Wasserstein ≤ 17.5°; to completion at a gap to the
  network of at most 1° in Wasserstein and in precision; the radial functions matter only if leaving out the
  descriptors costs 0.3° or more.

## Results: basis

1000 samples, batch 500, validation. The gap is the mean over the 10 bins below s = 0.04 of the validation loss minus
the network's on the same draws (5000 poses per bin, s.e. 0.02–0.05, `inspect_energy.py`).

| model | features | Wasserstein (°) | precision (°) | FID_val | APD (cm) | loss gap |
|---|---|---|---|---|---|---|
| network (`fact_RDW` s10) | 4.4M weights | 14.93 | 12.86 | 0.039 | 14.55 | 0 |
| MLP embedding, K = 8 | 6503 basis | 16.51 | 14.39 | 0.047 | 15.16 | 2.36 |
| linear, ridge 1e-6 (the energy of 2026-11-24) | 6503 | 18.24 | 15.90 | 0.088 | 16.33 | 3.11 |
| + joint density, one s0, ridge 1e-4 | 6503 | 17.80 | 15.36 | 0.079 | 15.65 | 3.02 |
| + pairs, ridge 1e-4 | 7799 | 17.78 | 15.58 | 0.084 | 16.49 | 2.31 |
| + products, ridge 1e-4 | 20,354 | 16.86 | 15.18 | 0.047 | 16.47 | 1.44 |
| + pairs and products, ridge 1e-4 | 21,650 | 16.83 | 15.16 | 0.046 | 16.48 | 1.48 |
| **+ products, joint density, ridge 1e-3** | 20,354 | **16.04** | **13.99** | 0.046 | 15.13 | **1.15** |

- The network, the MLP, the base linear energy and "joint density, one s0" are in one run (`generate_baselines.log`);
  the energies with products come from runs with the same batch size. The gaps of these four rows are in
  `low_noise_baselines.log`, the others in the `generate_rich*.log` files.
- The products carry the gain (−1.4° against the base); the pair orientations on top add nothing.
- The joint density alone helps the base little (−0.4°) and on top of the products a lot (−0.8°).
- Ridge of products + joint density (Wasserstein / precision / loss gap): 1e-4 16.09 / 14.07 / 1.55, 3e-4
  16.05 / 14.02 / 1.36, 1e-3 16.04 / 13.99 / 1.15, 3e-3 16.04 / 13.95 / 1.00, 1e-2 16.09 / 13.97 / 0.97, 3e-2
  16.21 / 14.03 / 1.17. The loss has its optimum near 1e-2, the generation is flat from 1e-4 to 1e-2. The base basis
  behaved the other way: 1e-3 and more hurt it (Wasserstein 18.1 at 1e-6 and 1e-4, 18.8 at 1e-3, 22.0 at 1e-2;
  `generate_step1.log`, batch 1000).
- In-sample, fresh train and validation loss of products + joint density at ridge 1e-3 (mean of the 10 bins):
  40.78 / 41.41 / 44.90. The network: 35.60 on fresh train poses, 43.75 on validation poses.

## Results: bins, smoothing, triples, descriptors

| variant of products + joint density, ridge 1e-3 | Wasserstein (°) | precision (°) | loss gap |
|---|---|---|---|
| 8 bins | 16.32 | 14.31 | 1.10 |
| 16 bins (the model above) | 16.04 | 13.99 | 1.15 |
| 32 bins | 15.97 | 13.91 | 1.20 |
| 16 bins, weights smoothed over the bins (¼, ½, ¼, three times, s < 0.17) | 16.10 | 14.01 | 0.90 |
| triples, reach 3 | 15.97 | 13.87 | 1.22 |
| triples, reach 5 | 15.94 | 13.80 | 1.42 |
| without any descriptor (2436 features), batch 1000 | 15.99 | 13.93 | 1.19 |
| full, batch 1000 (the control of the row above) | 15.91 | 13.88 | 1.15 |

- Every row is within the noise of 16.04, except 8 bins.
- The smoothing lowers the loss gap to 0.90, but not the generation; the ridge and the bins do the same: they trade
  the variance of the weights, and neither buys generation.
- Leaving out the distances, the angles, the twist or all of the descriptors changes the Wasserstein by at most 0.08°,
  and the loss gap by at most 0.04 (branch `experimental/ace-radial-ablation`, `fit.py --leave-out`). By the rule, the
  radial functions are not worth changing.

## Results: joint ranges

Share of the generated poses with a joint outside the training range (0.05–99.95% quantiles of twist and swing over
400,000 training poses), mean excess beyond the limit of the values that are outside (`joint_ranges.py`, 1000 samples
each, validation poses for comparison):

| model | poses with a joint outside | mean excess (°) | width ratio of the 99.5% interval |
|---|---|---|---|
| validation poses (8601) | 6.7% | 4.9 | 1.01 |
| network | 2.2% | 5.4 | 0.88 |
| MLP embedding, K = 8 | 8.3% | 20.9 | 1.00 |
| linear, ridge 1e-6 | 16.6% | 23.7 | 1.39 |
| + joint density, one s0 | 11.6% | 13.2 | 1.09 |
| + products | 10.2% | 16.6 | 1.23 |
| **+ products, joint density, ridge 1e-3** | **5.4%** | 8.2 | 0.94 |

- The linear energy's samples break a joint's limits in one or two joints per pose, not all at once. Dropping the
  poses outside the ranges from the old linear energy's samples takes the Wasserstein from 18.24 to 17.44° and the
  precision from 15.93 to 14.91° (`joint_ranges.py --filter`): the limits are about a quarter of its gap, the
  rest is inside them.

## Results: which noise levels matter

- *Completion* (200 validation poses, K = 10, inpainting at t81, `complete.py --band`, files in
  `~/data/experiments/ace/diag/`; mean / best of 10 / APD in mm and cm). Energy = the linear energy of 2026-11-24:

  | score used | mean | best of 10 | APD |
  |---|---|---|---|
  | network | 85.23 | 47.32 | 6.24 |
  | energy everywhere | 128.16 | 68.47 | 10.49 |
  | energy at s < 0.04, network elsewhere | 122.96 | 65.58 | 10.06 |
  | energy at 0.04 ≤ s < 0.4 | 86.70 | 47.98 | 6.38 |
  | energy at s ≥ 0.4 | 85.14 | 47.27 | 6.23 |
  | network, no DPS (guidance 0, replacement) | 101.70 | 54.62 | 7.65 |
  | energy, no DPS | 131.28 | 69.51 | 10.97 |

  The damage is below s = 0.04, not from the guidance's derivative of the score. The limits 0.04 and 0.4 were chosen
  by hand.
- *Generation* with the same linear energy (batch 1000): Wasserstein 16.25° (energy at s < 0.04), 15.58° (0.04–0.4),
  14.98° (above), 18.10° everywhere, 14.91° for the network. The low band matters most in generation as well.

## Results: is the energy a −log p

−E against the log-likelihood of its own probability flow ODE, and against the network's (100 validation poses,
50 midpoint steps, `ode_check.py`; Pearson / slope; mean log p of the own ODE, the network's in brackets):

| energy | s0 | −E, own ODE | −E, network's ODE | mean log p (nats) |
|---|---|---|---|---|
| linear, ridge 1e-6 | 0.002 | 0.980 / 1.07 | 0.923 / 1.45 | 109.9 (121.8) |
| | 0.02 | 0.978 / 0.93 | 0.936 / 1.26 | 87.4 (92.5) |
| | 0.2 | 0.990 / 0.93 | 0.965 / 1.09 | 43.3 (44.5) |
| MLP embedding, K = 8 | 0.002 | 0.971 / 1.01 | 0.947 / 1.34 | 114.6 (121.8) |
| | 0.02 | 0.983 / 0.95 | 0.960 / 1.24 | 89.9 (92.5) |
| | 0.2 | 0.991 / 1.00 | 0.984 / 1.09 | 44.2 (44.5) |
| products + joint density | 0.002 | 0.956 / 1.27 | 0.942 / 1.42 | 118.2 (121.8) |
| | 0.02 | 0.976 / 1.06 | 0.971 / 1.25 | 90.5 (92.5) |
| | 0.2 | 0.985 / 0.96 | 0.988 / 1.08 | 43.6 (44.5) |
| the same, smoothed over the bins | 0.002 | 0.954 / 1.21 | 0.946 / 1.38 | 118.1 (121.8) |

- The new energy is closer to the network in mean log p and in the ranking of the poses than the old one and the MLP
  at s0 ≥ 0.02; at 0.002 it exaggerates the differences (slope 1.3–1.4) and is less consistent with its own ODE.
- Smoothing the weights does not repair the slope, so the independence of the bins is not (alone) the cause. The cause
  is not known.

## Cost

- *Fit:* 4.7 min on the GPU for products + joint density (16 bins, three ridges), 2 min for 8 bins, 8.3 min for 32. The
  network `fact_RDW` trains in 3.4 min (`train_time.json`) on one GPU.
- *One call of the score for 500 poses on the Mac's CPU* (a loaded machine, not a benchmark): network 0.015 s, base
  linear 0.16 s, MLP 0.24 s, products + joint density 0.35–0.39 s, with triples 0.45 s. The code builds all the
  features and differentiates through them; a quadratic form for the products and a closed-form gradient of the
  density are not yet used.

## Conclusions (2026-11-25)

1. **The linear energy lacked couplings of joints, not positions.** Products of local rotations of pairs, the first
   order of coupling, carry the gain; the distances and angles of the points (the atomic descriptors) and their radial
   functions hardly matter in generation.
2. **Joint limits belong in the model.** A reference energy of every joint's noised density brings the share of broken
   joints to that of real poses and the Wasserstein down by 0.8°.
3. **A plateau near 16.0° for this family.** Ridge, bins, smoothing and triples move the loss at low noise but not the
   generation beyond the noise. Loss and generation disagree often (8 bins, smoothing, ridge), so models are chosen by
   generation and loss is only shown next to it.
4. **The gap to the network** is 1.1° in Wasserstein and 1.1–1.4° in precision, at the rule's limit of 1°.
5. **Next:** the completion of the new energies (rule: a gap to the network below 21.5 mm, half of the earlier 42.9 mm,
   counts as a success); then higher order in pairs (D^2 ⊗ D^1 of near pairs), a density of a joint given its parent,
   or a nonlinear embedding on the new basis; speed only after the accuracy.

## What it does not show (2026-11-25)

- One fit and one draw of 1000 samples per row; the noise of a fit was not measured.
- Generation is judged on the same 8601 validation poses for every choice, so the numbers are optimistic for the
  models chosen by them.
- The joint limits are only those of single joints: limits that depend on another joint are in the products at best.
- Completion of the new energies, and the interpretability of the coefficients (couplings per pair of joints, their
  stability across fits), are not checked. The noise bands of the completion were not refitted for the new energies.
- The speed numbers are of unoptimised code on a loaded Mac.

## Where things are

- **2026-11-25** (Mac, `~/data/experiments/ace/` unless noted; copied from workstation2):
  - code on `experimental/ace-descriptors` up to b28b5f6: `explore/ace_energy/{fit.py,joint_density.py,generate.py,
    inspect_energy.py,complete.py,ode_check.py,joint_ranges.py,smooth_bins.py}`;
  - fits: `energy_b_jd_*`, `energy_b_rich_{pairs,products,both}_*`, `energy_b_rich_products_jd_*` (suffix `_r<ridge>`,
    `_b8`, `_b32`, `_t3`, `_t5`, `_smooth{1,3}`), `energy_b_abl_*` (branch `experimental/ace-radial-ablation`);
  - logs: `jd.log`, `rich.log`, `rich_jd.log`, `rich_jd2.log`, `rich_bins_triples.log`, `abl.log`;
  - harness: `generate_step1.log`, `generate_baselines.log`, `generate_rich*.log`, `generate_bins_triples.log`,
    `generate_abl.log`, `low_noise_baselines.log`; samples in `generate_1000/` and `generate_1000_b500/`;
  - ODE checks: `ode_check_*.log`; completion diagnostics: `diag/*.pt`, `diag.log`.
