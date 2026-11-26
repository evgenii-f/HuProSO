# ACE energy, 2026-11-26: an MLP embedding on the products basis improves generation

Branch `experimental/ace-descriptors`. Validation numbers only; no test was run. Part of the ACE line: [overview](README.md).

## Summary

- **Question.** Does a nonlinear embedding on top of the products + joint limits basis (P+L) give what more linear
  features did not ([completion and plateau](2026-11-26_completion_and_plateau.md))?
- **Generation: yes.** The MLP embedding (K = 8, started from the P+L fit) reaches Wasserstein 15.31° against 15.93° of
  P+L and 14.86° of the network `fact_RDW` s10 (1000 samples, one run of the harness with all rows). The difference,
  0.62°, is four times the noise of the harness (0.05–0.15°); the precision goes 13.98 → 13.41.
- **Validation loss** 6.336 → 6.090 (network 5.951): 64% of the fit's gap to the network closed.
- **The control** (the same coefficients trained by Adam without an embedding) stays at the fit: loss 6.290,
  Wasserstein 15.94°. The gain comes from the nonlinearity, not from more training of the coefficients.
- **Completion: in the same direction, below the rule.** Inpainting 95.71 against 97.49 mm for P+L (200 poses, K = 10):
  −1.8 mm [−3.5, −0.1], below the 2 mm written before the runs; against the control −1.3 [−2.9, +0.2]. The gain is where
  the gap was: poses far from the training data (gap to the network 20.3 → 17.0 mm) and bent knees (25.7 → 17.0 mm).
- **Ranking.** Its own 10 hypotheses ranked by its energy at s = 0.02 give 82.44 mm (P+L 89.39, t5 87.18), below the
  network's mean of K (85.84 mm on the same Mac run).
- **Decision (with the user).** A success in generation; the line continues with the MLP embedding on P+L, not with the
  linear energy. Completion is reported as below its rule.

## Setup

- **Model** (`explore/ace_energy/embedding.py`, unchanged from 2026-11-24 except that the joint density of the fit is
  kept, e9cd138): per joint K densities φ_j^(k), linear in the joint's features with coefficients per bin of log s;
  Ẽ_t = Σ_j [φ_j^(1) + MLP(φ_j^(2..K), log s / 4, e_j)] + the fixed joint density of the fit. One MLP (64 × 2, SiLU)
  shared by the joints, e_j a learned 8-dim vector of the joint, the last layer zero at the start, so the energy starts
  as the fit (`--check`: 6.336 = the fit's 6.336).
- **Basis:** the fit `energy_b_rich_products_jd_r0.001.pt` (P+L, 20,354 features; products of all 171 pairs of local
  rotations, D^{1,2,3}(x_j), pair harmonics, descriptors, joint density of 5000 poses), 4.6M coefficients with K = 8.
- **Rows** (one setting each, not tuned; the settings of the MLP of 2026-11-24):
  - A: `--embedding mlp --k 8`;
  - B (control): `--embedding fs --k 1`, no embedding: the linear energy, its coefficients trained by Adam from the
    fit.
  - Both: Adam lr 1e-3, cosine, 10,000 steps × 2048 training poses, RSGM's loss with the noising of `fact_RDW` s10;
    workstation2, GPU 1, 0.26 s (A) and 0.14 s (B) per step.
- **Rule written before the runs:** A is a success if inpainting is at least 2 mm better than P+L 1e-3 with a 95%
  interval that excludes 0, and better than B. Loss, the harness and the ranking are shown next to it.
- **Completion:** one ready point per method, as for the linear energies: inpainting at t81 (DPS guidance 0.0493,
  RePaint 5), SDS at t15, no-DPS; the first 200 validation poses of the completion studies, K = 10, with `--rank 2e-3
  2e-2 2e-1`. Paired differences with a bootstrap over the poses (`diagnose.py completion`).
- **Harness:** `generate.py`, 1000 samples, batch 500, the network, P+L, A and B in one run.

## Results: training

Validation loss (5000 poses, the draws of fit.py), mean and per bin of s (`mlp_pl_k8.log`, `lin_pl_adam.log`).

| model | mean | s 2e-05 | 2e-04 | 2e-03 | 2e-02 | 2e-01 | 2e+00 |
|---|---|---|---|---|---|---|---|
| network `fact_RDW` s10 | 5.951 | – | – | – | – | – | – |
| P+L 1e-3 (the fit, step 0 of both) | 6.336 | 55.51 | 56.36 | 50.47 | 21.23 | 6.21 | 0.25 |
| B: linear, Adam | 6.290 | 55.38 | 56.22 | 50.57 | 21.11 | 6.15 | 0.23 |
| A: MLP, K = 8 | **6.090** | 55.36 | 56.31 | 49.81 | 20.54 | 5.91 | 0.23 |

- A lowers the loss mostly at s ≥ 2e-3; it was still falling at step 10,000 (6.093 at 9000, the end of the cosine).
- Both rows first rise to 6.565 at step 1000 (lr 1e-3 from the first step) and then fall.

## Results: generation

1000 samples, one run with all rows (`generate_mlp.log`), degrees per joint against the 8601 validation poses.

| model | Wasserstein | recall | precision | FID_val | APD (cm) |
|---|---|---|---|---|---|
| network `fact_RDW` s10 | 14.86 | 13.51 | 12.79 | 0.0337 | 14.88 |
| P+L 1e-3 | 15.93 | 14.26 | 13.98 | 0.0360 | 15.19 |
| B: linear, Adam | 15.94 | 14.23 | 13.97 | 0.0371 | 15.14 |
| **A: MLP, K = 8** | **15.31** | **13.82** | **13.41** | **0.0285** | 15.31 |
| training poses in the place of the samples | 15.15 | 13.46 | 13.08 | – | – |

- A closes 58% of P+L's gap to the network in Wasserstein, 0.62° beyond the noise; B does not move.

## Results: completion

200 validation poses, K = 10, MPVPE in mm, mean of K (`diagnose_completion.log`).

| prior | inpainting | SDS | no-DPS | inpainting, lowest energy of 10 at s = 0.02 |
|---|---|---|---|---|
| network | 85.23 | 89.43 | 101.70 | 81.10 (ranked by P+L) |
| P+L 1e-3 | 97.49 | 108.03 | 109.56 | 89.39 |
| B: linear, Adam | 97.02 | 106.55 | 108.82 | 88.91 |
| **A: MLP, K = 8** | **95.71** | 107.87 | **108.11** | **82.44** |

Paired differences (bootstrap 95% interval over the poses; per occlusion left_leg, legs, arms, trunk):

| a − b | all | left_leg | legs | arms | trunk |
|---|---|---|---|---|---|
| A − P+L, inpainting | −1.78 [−3.51, −0.13] | −2.81 | −2.02 | −0.91 | −1.35 |
| A − B, inpainting | −1.30 [−2.88, +0.24] | −1.33 | −2.09 | −0.33 | −1.47 |
| B − P+L, inpainting | −0.47 [−0.91, −0.05] | −1.48 | +0.07 | −0.58 | +0.11 |
| A − t5, inpainting | −0.48 [−2.26, +1.27] | −0.49 | −1.96 | +2.13 | −1.61 |
| A − P+L, SDS | −0.17 [−2.66, +2.37] | +3.97 | +3.90 | −6.62 | −1.92 |
| A − P+L, no-DPS | −1.45 [−2.61, −0.22] | −4.11 | +1.18 | −1.21 | −1.66 |

- **By the rule A does not pass in completion**: −1.8 mm with an interval that excludes 0, but less than 2 mm, and not
  significantly better than B.
- **Where it gains**, gap to the network by thirds of the distance to the nearest training pose (near, middle, far) and
  by the flexion of the left knee in left_leg (straight, middle, bent):

| prior | near | middle | far | legs: far | left knee: bent |
|---|---|---|---|---|---|
| P+L 1e-3 | 8.6 | 8.0 | 20.3 | 28.3 | 25.7 |
| B: linear, Adam | 8.0 | 8.1 | 19.4 | 29.0 | 23.4 |
| A: MLP, K = 8 | 6.5 | 8.1 | **17.0** | **22.4** | **17.0** |

- Against a baseline without information (random training limbs in place of the hidden ones) A keeps 86–89% of the
  network's gain in every occlusion, P+L 84–88%.
- Inpainting stays the best method for every prior.

## Conclusions

1. **The nonlinear embedding is the lever the linear basis was not.** The same basis, the same training, without the
   embedding (B) gives nothing; with it (A) the loss drops by 0.25 and generation by 0.62°.
2. **Generation is a success, completion is in the same direction but below the rule** (−1.8 mm against 2 mm). The
   gain in completion is concentrated where the linear energy failed: far poses and bent knees.
3. **As a judge of hypotheses A is better than P+L:** its own ranked hypotheses (82.44 mm) come near the network ranked
   by P+L (79.32 mm). The network's hypotheses ranked by A were not measured.
4. **Next:** the code made faster first ([speed](speed.md)), then a joint Optuna study of the embedding and the basis
   (K, width, depth, lr, steps; basis among the ready fits), chosen by Wasserstein on validation, then completion of the
   winner once.

## What it does not show

- One training per row, one setting, not tuned; the loss was still falling at the end.
- One run of the harness (1000 samples), though with all rows in it, so the comparison of rows shares its noise.
- 200 completion poses at points tuned for a network; one run per method.
- The ranking level s = 0.02 was chosen on the same 200 poses.
- The network's hypotheses ranked by the MLP energy were not measured.

## Where things are

- **Code:** `explore/ace_energy/embedding.py` (the joint density of the fit kept: e9cd138; `--check`),
  `explore/ace_energy/inspect_energy.py` (`load_energy` restores it), `explore/ace_energy/diagnose.py` (rows `MLP K8`,
  `linear Adam`).
- **Runs** (workstation2, copied to the Mac, `~/data/experiments/ace/`):
  - `energy_mlp_pl_k8.{pt,json}`, `mlp_pl_k8.log` (`embedding.py --init energy_b_rich_products_jd_r0.001.pt
    --embedding mlp --k 8 --steps 10000 --batch 2048 --device cuda:0`);
  - `energy_lin_pl_adam.{pt,json}`, `lin_pl_adam.log` (`--embedding fs --k 1`, the rest the same);
  - completion `diag5/{inpaint,sds,nodps}_{mlp_pl_k8,lin_pl_adam}.pt` (`complete.py --poses 200 --batch 100 --rank
    2e-3 2e-2 2e-1`, with `--method dposer` or `--no-dps`), log `diag5.log`;
  - harness `generate_mlp.log`, samples in `generate_mlp/`;
  - the comparison `diagnose_completion.log` (`python explore/ace_energy/diagnose.py completion`).
