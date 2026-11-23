# ACE descriptors of a pose: no help as an input of the score network, a strong linear energy

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
4. **Next (not decided):**
   - completion with the energy as the prior, to see what 1.2× in the loss at middle noise costs in mm;
   - or more capacity first: D^3 one-body terms, products of neighbouring joints' one-body terms, triples within 3
     bones.

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
