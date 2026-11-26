# ACE descriptors of a pose: no help as an input of the score network, a linear energy near the network in generation

Branch `experimental/ace-descriptors`. Validation numbers only; no test was run. Part of the ACE line: [overview](README.md).

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
- **2026-11-24: more basis, nonlinear embeddings, completion** (section 2026-11-24 below).
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
