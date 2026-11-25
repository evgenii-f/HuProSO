# ACE descriptors of a pose: no help as an input of the score network, a linear energy near the network in generation

Branch `experimental/ace-descriptors`. Validation numbers only; no test was run.

## Summary

- **ACE descriptors as the input of the score network do not help (variant A).**
  - The descriptors: distances and angles between the joint positions (Chebyshev × Legendre), plus the twist of the
    bones.
  - With them next to the matrices, `ace_RDW` is worse than its pair `fact_RDW` at the tuned inpainting point of
    `abl_residual`: mean 89.36 against 86.64 mm, best of 10 52.39 against 47.83 mm, APD 5.72 against 6.31 cm.
  - Validation loss 6.30 against 5.85.
  - By the rule fixed before the evaluation this is a negative result: the inpainting mean is worse by more than 1 mm.
- **A linear energy in the same descriptors is a strong score model (variant B).**
  - Score = −∇E_t, E_t = Σ c_p(t) B_p, fitted in closed form per noise level on the CPU (about 15 minutes).
  - With one-body terms of each joint's rotation it reaches a validation loss of 7.23, against 5.95 for `fact_RDW`
    (4.4M parameters) and 26.50 for the zero score: 94% of the network's gain.
  - Without the one-body terms: 10.10 (80%).
- **What the energy needs.**
  - *Smooth descriptors.* The plain angular and twist terms have gradients like 1/r near folded limbs. One such
    validation pose gave a loss of 20,611. As polynomials (solid harmonics, squared distances, unnormalized twist)
    the outlier is gone: mean 17.05 → 10.10.
  - *One-body terms of each joint's own rotation* (its matrix entries and their products) carry the score at large
    noise: at s ≈ 0.4 the loss goes 9.82 → 4.71, against 3.40 for the network.
- **Decision.** Variant A is closed. Variant B is promising as an interpretable, conservative prior. Next: completion
  with the energy as the prior, or more capacity first.
- **2026-11-24: more basis, nonlinear embeddings, completion** (section below).
  - The harmonics of SO(3) as further basis (`hp.models.so3_harmonics`, shared with `experimental/wigner-inputs`) add
    little: D^{1,2,3}(x_j) 7.13, plus the pair harmonics D^l(G_i^T G_k) 7.04. The linear energy has reached a plateau.
  - A per-joint energy with a nonlinear embedding, trained by score matching from the linear fit:
    - Finnis-Sinclair-like, K = 2: 6.71;
    - an MLP of K = 8 densities: 6.54, median 1.66 against the network's 1.65.
  - Completion with the linear energy as the prior (inpainting at t81, 200 poses) is far behind the network:
    128.16 / 68.47 mm, APD 10.49 cm, against 85.85 / 47.21 mm, 6.30 cm.
  - A quarter of the remaining loss gap is one validation pose: a rare deep knee bend (143°), outside the training
    range in 155 of 6503 features, where the polynomial energy gives a spurious force of about 18 at the knee. The
    densities there are ordinary, so the cause is the large, ill-conditioned coefficients off the data, not the
    densities. The next test is a stronger ridge.
- **2026-11-25: products of local rotations and joint limits** (section below), still linear and in closed form.
  - The products (x_i)_a (x_k)_b of the entries of all 171 pairs of local rotations, with a reference energy made of
    every joint's own noised density, bring generation close to the network: Wasserstein 16.04 against 14.93° (base
    linear energy 18.24°, MLP embedding 16.51°; 1000 samples, validation).
  - Only 5.4% of the generated poses have a joint outside the training range, against 6.7% of the validation poses,
    2.2% of the network's and 16.6% of the old linear energy's.
  - Ridge (1e-4 to 1e-2), noise bins (16 or 32), smoothing over the bins, triples of joints and the distance, angle
    and twist descriptors change the generation by less than about 0.15°, the size of the noise: a plateau near 16.0°.
  - Completion (inpainting) with the new energies was not run. The earlier failure of the linear energy (128.16 mm
    against 85.23 mm) comes from the noise levels below s = 0.04: that band alone adds 37.7 of the 42.9 mm.

## Setup

- **Data:** PoseD-Flow's AMASS, every 10th training pose (c10, 1,861,105 poses); validation poses of the same split.
- **Descriptors** (`src/hp/models/ace_features.py`, `AceFeatures`):
  - 25 points: the 22 SMPL joints from the fixed skeleton, the root at the origin, plus the tips of the hands (8 cm)
    and of the head (10 cm).
  - Every point is its own species; the graph is complete.
  - 2-body: T_1..T_4 of each pair distance whose length varies with the pose, scaled by the path length (270 pairs).
  - 3-body: T_0..T_1 × T_0..T_1 × P_1..P_2 of the angle, centre within 2 bones (165 triples).
  - Twist: cos, sin of the twist about the child bone, 18 joints.
  - 2436 in all.
  - `smooth=True`, for the energy: squared distances, solid harmonics (|u||v|)^l P_l, and the twist as
    (w² − u², 2wu), i.e. polynomials in the positions and in x.
- **Variant A** (`inputs="local+ace"` of RSGM): the descriptors z-scored (std floor 0.1, |z| ≤ 20) next to the
  matrices.
  - `ace_RDW` = `fact_RDW` of the factorial (timeres 1024 × 2 blocks, residual, dropout 0.1, isotropic noise, std
    weighting, 30k × 4096) + the input: 6.9M parameters against 4.4M.
  - Trained on workstation2, 4.6 minutes.
  - Evaluated on workstation2 with `fact_RDW` seeds 10 and 11 by the same command: `scripts/evaluate.py completion
    --split valid --together --like "RSGM abl residual: their c10, inpaint t81"`.
  - That is inpainting at the tuned point of `abl_residual` (400 EDM steps, guidance 0.049, resample 5, no
    replacement), 1000 validation poses, K = 10, four occlusions.
- **Rule** (fixed before the evaluation, in `scripts/train_rsgm_posedflow.py` at the ACE configs):
  - `ace_RDW` is better if at least one of:
    - inpainting mean −1 mm at APD ≥ 6.3 cm;
    - hybrid −5 mm;
    - harness Wasserstein −0.5°.
  - And nothing worse beyond the noise: inpainting mean +1 mm, harness +0.2°.
- **Variant B** (`explore/ace_energy/fit.py`, Mac CPU):
  - Model: E_t(x) = Σ_p c_p(t) B_p(x), score = −∇E_t along the tangent directions x_j exp(ε e_a) (the frame of RSGM's
    score).
  - With w(t) = std(t) c(t), RSGM's loss ‖std·s − target‖² is linear least squares in w. It is solved per bin of the
    noise level: 16 bins of equal width in log s over [s(0.002), s(1)], 20,000 samples per bin (1.14M rows), ridge
    1e-6 × the mean diagonal. Between bin centres w is interpolated linearly in log s.
  - Noise and targets are drawn by the noising of `fact_RDW` s10. They reproduce `RSGM.loss` exactly (5.13276 =
    5.13276, checked by `--check`).
  - Evaluation: 5000 validation poses, t uniform on [0.002, 1], one draw per pose. The same draws are used for every
    model.
  - One-body terms (`--one-body 2`): the 9 matrix entries of each joint and their 45 products (spanning D^l, l ≤ 2,
    of SO(3)); 3462 features in all.

## Results

### Variant A: inpainting at the t81 point (validation, 1000 poses, K = 10)

| model | left_leg | legs | arms | trunk | mean (mm) | best of 10 (mm) | APD (cm) |
|---|---|---|---|---|---|---|---|
| ace_RDW s10 | 86.71 | 110.27 | 115.75 | 44.72 | 89.36 | 52.39 | 5.72 |
| fact_RDW s10 | 82.75 | 106.28 | 112.68 | 44.72 | 86.61 | 47.80 | 6.29 |
| fact_RDW s11 | 82.81 | 106.78 | 112.18 | 44.92 | 86.67 | 47.85 | 6.32 |

- `ace_RDW` is worse on every occlusion but the trunk, with narrower hypotheses.
- The two seeds of `fact_RDW` differ by 0.06 mm in the mean, so the difference is not noise.

Validation loss at 30k steps (`history.json` / the training log):

| model | valid loss |
|---|---|
| ace_RDW s10 | 6.30 |
| fact_RDW s10 | 5.85 |
| fact_N s10 | 5.90 |

The harness and the hybrid were not run: the rule was already decided by the inpainting mean.

### Variant B: validation loss of the linear energy

Validation loss (std weighting) on the same draws for every model, mean over 5000 poses and per bin of s (the lower
edge of the bin):

| model | features | mean | median | s 2e-3 | s 8e-3 | s 2e-2 | s 4e-2 | s 8e-2 | s 2e-1 | s 4e-1 | s 8e-1 | s 2e+0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zero score | 0 | 26.50 | 24.31 | 58.94 | 58.50 | 56.23 | 54.16 | 52.39 | 46.81 | 36.97 | 17.63 | 2.25 |
| energy, plain descriptors | 2436 | 17.05 | 8.92 | 53.13 | 39.32 | 31.66 | 112.79 | 18.97 | 13.92 | 13.58 | 11.58 | 1.69 |
| energy, smooth | 2436 | 10.10 | 5.82 | 52.80 | 37.89 | 29.88 | 22.18 | 16.30 | 11.69 | 9.82 | 5.27 | 0.78 |
| energy, smooth + one-body | 3462 | **7.23** | **2.25** | 51.53 | 33.76 | 24.83 | 17.83 | 11.64 | 7.12 | 4.71 | 2.17 | 0.42 |
| fact_RDW s10 | 4.4M params | 5.95 | 1.65 | 42.22 | 28.47 | 19.88 | 14.60 | 9.60 | 5.95 | 3.40 | 1.72 | 0.25 |
| fact_N s10 | 1.2M params | 6.00 | 1.65 | 43.74 | 28.78 | 20.06 | 14.73 | 9.66 | 5.99 | 3.44 | 1.72 | 0.24 |

- Below s ≈ 1e-3 no model, the networks included, is far below the zero score (about 50 against 55).
- **Plain descriptors.** The bin at s ≈ 4e-2 (112.79) is one validation pose with a loss of 20,611; the median there
  is 21.8. Its predicted score came from the angular terms of the triples.
- **Smooth + one-body.** The energy is within 1.2× of the network at middle noise (17.83 against 14.60 at s ≈ 4e-2)
  and within 1.2–1.4× at large noise.

## 2026-11-24: harmonics of SO(3), nonlinear embeddings, completion

### Setup

- **Basis.** `fit.py --wigner 1 2 3 --pairs 1 2 3` adds to the smooth descriptors:
  - D^l(x_j) of every joint's own rotation, 19 × 83 = 1577 features;
  - D^l(G_j) of every joint relative to the root, and D^l(G_i^T G_k) of the 11 pairs of `WIGNER_PAIRS`, 30 × 83 = 2490
    features.

  These come from `hp.models.so3_harmonics`, moved to dev from `experimental/wigner-inputs` (commit 8f5785b). In all
  4013 and 6503 features; fitted as before (16 bins, 20,000 samples per bin, ridge 1e-6) with `--device cuda:0`, on
  workstation1, 32 and 81 s per bin.
- **Per-joint energy with an embedding** (`explore/ace_energy/embedding.py`).
  - Every feature belongs to the joints whose rotations place it (`owners`).
  - Every joint has K densities, each linear in its features with coefficients per bin of log s.
  - E~_t = Σ_j [φ_j^(1) + F(φ_j^(2..K))], where F is either:
    - "fs": sqrt(1 + φ²) − 1 summed over the densities, a smooth Finnis-Sinclair-like embedding; or
    - "mlp": one MLP (64 × 2, SiLU) shared by the joints, of the K − 1 densities, log s / 4 and a learned 8-dim
      vector of the joint.
  - Training starts from the linear fit (φ^(1) = the fit, the rest zero or small, so the energy is the linear fit
    at step 0). It uses Adam with lr 1e-3, cosine schedule, 10,000 steps × 2048 poses, on RSGM's loss with the
    reference run's noising.
  - Runs: workstation2 for FS (0.6 s per step, the GPUs shared), workstation1 for MLP (0.11 s per step).
- **Completion** (`explore/ace_energy/complete.py`): `fact_RDW` s10 with its score replaced by −∇E~_t / std(t).
  Inpainting at the t81 point, the first 200 poses of the 1000-pose validation draw, K = 10, workstation2.
  - The network's numbers on the same 200 poses come from its cached completions (`cache/valid/fact_RDW_s10`).
- **Diagnostics:** `explore/ace_energy/bins.ipynb` with `inspect_energy.py`: loss per bin, the worst poses of a bin,
  and for one pose the error per joint, the score split by feature group, the energy per joint and the skeleton.

### Results

Validation loss on the same draws as before (5000 poses; per bin of s, the lower edge):

| model | parameters | mean | median | s 2e-3 | s 4e-3 | s 2e-2 | s 4e-2 | s 8e-2 | s 2e-1 | s 4e-1 | s 8e-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear, smooth + one-body l ≤ 2 (above) | 3462 × 16 | 7.23 | 2.25 | 51.53 | n/a | 24.83 | 17.83 | 11.64 | 7.12 | 4.71 | 2.17 |
| linear, smooth + D^{1,2,3}(x_j) | 4013 × 16 | 7.13 | 2.00 | 53.96 | 40.25 | 24.78 | 17.92 | 11.60 | 7.08 | 4.31 | 1.93 |
| linear, + D^{1,2,3}(G_j), D^{1,2,3}(G_i^T G_k) | 6503 × 16 | 7.04 | 2.00 | 57.99 | 38.52 | 24.09 | 17.52 | 11.39 | 6.99 | 4.28 | 1.93 |
| FS embedding, K = 2 | 0.27M | 6.71 | 1.73 | 56.32 | 37.37 | 23.11 | 16.72 | 10.85 | 6.64 | 3.89 | 1.78 |
| MLP embedding, K = 8 | 1.1M | **6.54** | **1.66** | 56.38 | 36.76 | 22.22 | 16.13 | 10.52 | 6.44 | 3.70 | 1.74 |
| fact_RDW s10 | 4.4M | 5.95 | 1.65 | 42.22 | 32.61 | 19.88 | 14.60 | 9.60 | 5.95 | 3.40 | 1.72 |

- **More basis.** Diminishing returns: −2.9, −0.1, −0.1.
- **The nonlinear embeddings help at every noise level.** The MLP closes 46% of the linear energy's gap to the
  network (1.09 → 0.59), and its median equals the network's.
- **Where the MLP energy's gap of 0.59 sits** (the share of the poses in a bin × the gap in the bin):

  | noise level | share of the poses | contribution to the gap |
  |---|---|---|
  | above s ≈ 0.8 | 54% | 0.00 |
  | s ≈ 2e-1 – 4e-1 | 32% | 0.16 |
  | s ≈ 4e-3 – 8e-2 | 12% | 0.27 |
  | the bin s ≈ 2e-3 alone | 1% | 0.14 |
  | below | 2% | 0.05 |

**The bin at s ≈ 2e-3** (42 poses, linear fit with 6503 features):
- One pose (2216) has a loss of 457 against 64.5 for the network. The next worst is 102; the median of the bin is
  46.7. Without that pose the bin would be about 48 against the network's 42.
- Its error is at the left knee: a predicted score of about 18 where the target is near 0, spread over all the
  feature groups.
- The pose is real, not made up by the noise: a knee bend of 143° (145° after the noise).
- The densities φ there are ordinary (knee 1.21, the bin's range −0.77 to 3.67). But 155 of its 6503 features lie
  outside the 0.1–99.9% band of 20,000 training poses, mostly D^2 and D^3.
- The same pose sets this bin in every fit, the embeddings included (56–58).

Completion, inpainting at t81, the first 200 validation poses, K = 10:

| prior | left_leg | legs | arms | trunk | mean (mm) | best of 10 (mm) | APD (cm) |
|---|---|---|---|---|---|---|---|
| linear energy, 6503 features | 139.53 | 154.43 | 156.26 | 62.41 | 128.16 | 68.47 | 10.49 |
| fact_RDW s10 | 84.38 | 101.58 | 115.26 | 42.17 | 85.85 | 47.21 | 6.30 |

- An 18% higher loss gives a 42 mm worse mean and hypotheses almost twice as spread. The mean loss over the draws
  does not show what breaks the sampling.
- The likely reason is the same as for pose 2216: the trajectories pass through rare and unphysical poses, where the
  polynomial energy has large spurious forces. Not verified yet.

## 2026-11-25: products of local rotations, joint limits, generation

Question: what the linear energy lacks, and how near to the network it comes with a richer basis. All numbers are
validation numbers, no test was run.

### Setup

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

### Results: basis

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

### Results: bins, smoothing, triples, descriptors

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

### Results: joint ranges

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

### Results: which noise levels matter

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

### Results: is the energy a −log p

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

### Cost

- *Fit:* 4.7 min on the GPU for products + joint density (16 bins, three ridges), 2 min for 8 bins, 8.3 min for 32. The
  network `fact_RDW` trains in 3.4 min (`train_time.json`) on one GPU.
- *One call of the score for 500 poses on the Mac's CPU* (a loaded machine, not a benchmark): network 0.015 s, base
  linear 0.16 s, MLP 0.24 s, products + joint density 0.35–0.39 s, with triples 0.45 s. The code builds all the
  features and differentiates through them; a quadratic form for the products and a closed-form gradient of the
  density are not yet used.

### Conclusions (2026-11-25)

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

### What it does not show (2026-11-25)

- One fit and one draw of 1000 samples per row; the noise of a fit was not measured.
- Generation is judged on the same 8601 validation poses for every choice, so the numbers are optimistic for the
  models chosen by them.
- The joint limits are only those of single joints: limits that depend on another joint are in the products at best.
- Completion of the new energies, and the interpretability of the coefficients (couplings per pair of joints, their
  stability across fits), are not checked. The noise bands of the completion were not refitted for the new energies.
- The speed numbers are of unoptimised code on a loaded Mac.

## Conclusions

1. **Functions of the pose at the input do not help the score network.** That holds for z-scored tangents (score
   network report, section 9) and for these descriptors. The Wigner inputs of `experimental/wigner-inputs` make the
   hypotheses narrower in the same way. The descriptors here even hurt: worse validation loss, narrower and worse
   completions, with 2.5M more parameters in the first layer.
2. **A model linear in the descriptors gets 94% of the network's reduction of the loss.**
   - It is a conservative score, with an energy at every noise level.
   - Its score at a joint is the torque of the "forces" −∂E/∂p on the points of its subtree, plus a twist term.
   - Every coefficient belongs to a distance, an angle, a twist or a joint's orientation.
3. **The parts of such a model, by the loss.**
   - One-body terms of each joint's rotation matter most at large noise (the marginals of the joints).
   - The positions' descriptors matter at middle noise.
   - The descriptors must be smooth (polynomial) for a model fed noisy poses.
4. **More basis is not the lever (2026-11-24); nonlinearity is.** The linear energy reaches a plateau near 7.0 with
   the harmonics of SO(3), while per-joint embeddings lower it to 6.71 (FS) and 6.54 (MLP), with the median at the
   network's.
5. **A good loss is not yet a good prior.** Completion with the linear energy is 42 mm worse than with the network.
   The tails matter, and they come from rare poses off the data, where the polynomial energy's forces are large:
   one such pose makes a quarter of the remaining loss gap.
6. **Next:**
   - a stronger ridge (`fit.py --ridge 1e-6 1e-4 1e-3 1e-2`, one pass), checked by the tails (max and 99th percentile
     of the loss) and by completion;
   - then completion with the MLP energy.

## What it does not show

- Variant A: one seed, c10, one inpainting point, which was tuned for another network; the harness and the hybrid
  were not run.
- Variant B:
  - only the loss; no sampling, completion or likelihood with the energy yet;
  - one fit per setting (the fits are deterministic given the seed, but the 20,000 samples per bin are one draw);
  - the ridge, the bins and the sample size were not varied;
  - the loss of the networks here is on other draws than their `history.json` valid loss (5.95 against 5.85 for
    `fact_RDW`).
- Whether the energy's coefficients are interpretable in practice (sparse, stable across seeds) is not checked.
- 2026-11-24:
  - completion of the energy on 200 poses only, at a point tuned for a network, and only for the linear energy;
  - the embeddings were trained once each, 10,000 steps, without tuning (lr, K, width);
  - the explanation of the completion failure by off-data forces is a hypothesis.

## Where things are

- **Code:** `src/hp/models/ace_features.py` (with `tests/test_ace_features.py`); `inputs="local+ace"` in
  `src/hp/models/rsgm.py`; the configs `ace_RDW`, `ace_N` and the rule in `scripts/train_rsgm_posedflow.py`;
  `explore/ace_energy/fit.py` (variant B; `--check` for its self-checks).
- **Run:** `~/data/experiments/ace/rsgm_posedflow_ace_RDW_s10` on workstation2. On the Mac: its training log and the
  descriptor statistics (`~/data/experiments/ace/train_ace_RDW_s10.log`, `rsgm_ace_statistics.json`).
- **Inpainting:**
  - the caches `~/data/experiments/completion/cache/valid/{ace_RDW_s10,fact_RDW_s10,fact_RDW_s11}`;
  - their rows in `~/data/experiments/ace/inpaint_t81_rows.jsonl`;
  - the log `~/data/experiments/ace/eval_inpaint_t81.log`.
- **Energy fits** (Mac): `~/data/experiments/ace/energy_b.pt` (plain), `energy_b_smooth.pt`,
  `energy_b_smooth_ob2.pt` (with `.json` summaries and `.log` outputs). The commands:
  - `python explore/ace_energy/fit.py --plain --per-bin 20000 --out ...` (the plain fit was made before `--plain`
    existed, with the then default);
  - `--per-bin 20000 --out ...`;
  - `--per-bin 20000 --one-body 2 --out ...`.
- **2026-11-24:**
  - linear fits with harmonics:
    - `energy_b_smooth_w123` on workstation1 (`--wigner 1 2 3 --device cuda:0`);
    - `energy_b_smooth_w123_p123` (`--pairs 1 2 3` as well) on workstation1 and the Mac.
  - embeddings: `energy_fs.*` (`embedding.py --init energy_b_smooth_w123_p123.pt`, workstation2) and
    `energy_mlp_k8.*` (`--embedding mlp --k 8`, workstation1), both also on the Mac.
  - completion: `~/data/experiments/ace/complete_linear_200.log` on workstation2 (`complete.py --fit
    energy_b_smooth_w123_p123.pt --poses 200 --device cuda:0`).
  - the shared harmonics: `src/hp/models/so3_harmonics.py`, `tests/test_so3_harmonics.py`.
- **2026-11-25** (Mac, `~/data/experiments/ace/` unless noted; copied from workstation2):
  - code on `experimental/ace-descriptors` up to b28b5f6: `explore/ace_energy/{fit.py,joint_density.py,generate.py,
    inspect_energy.py,complete.py,ode_check.py,joint_ranges.py,smooth_bins.py}`;
  - fits: `energy_b_jd_*`, `energy_b_rich_{pairs,products,both}_*`, `energy_b_rich_products_jd_*` (suffix `_r<ridge>`,
    `_b8`, `_b32`, `_t3`, `_t5`, `_smooth{1,3}`), `energy_b_abl_*` (branch `experimental/ace-radial-ablation`);
  - logs: `jd.log`, `rich.log`, `rich_jd.log`, `rich_jd2.log`, `rich_bins_triples.log`, `abl.log`;
  - harness: `generate_step1.log`, `generate_baselines.log`, `generate_rich*.log`, `generate_bins_triples.log`,
    `generate_abl.log`, `low_noise_baselines.log`; samples in `generate_1000/` and `generate_1000_b500/`;
  - ODE checks: `ode_check_*.log`; completion diagnostics: `diag/*.pt`, `diag.log`.
