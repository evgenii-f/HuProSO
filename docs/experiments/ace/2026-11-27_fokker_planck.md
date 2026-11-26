# ACE energy, 2026-11-27: a Fokker-Planck term makes the noise levels agree, and completion worse

Branch `experimental/ace-descriptors`. Validation numbers only; no test was run. Part of the ACE line: [overview](README.md).

## Summary

- **Question.** On hard poses the ACE energies do not tell good hidden legs from bad at s = 0.002, while they do at
  s ≥ 0.02 ([tuning and the energy of the truth](2026-11-26_completion_tuning_and_truth_energy.md)). Denoising score
  matching trains every noise level only near the data noised at that level, so nothing ties the levels far from it.
  Does a term that ties them by the heat equation fix the small levels?
- **The violation is real and large.** E_s = −log p_s must obey dE/d log s = s/2 (ΔE − |∇E|²) + const(s) (the
  generator of IGSO(3) is Δ/2). Near the data both energies (P+L, MLP K8) obey it within 0.5–2 nats per unit of log s
  for s ≥ 0.02 and 4–10 nats below; at poses noised at s' = 1 and scored at s = 0.002 they miss it by about 400 nats.
- **The term works on what it measures.** The linear energy P+L trained 10,000 steps with the term (weight 1.5e-3, set
  before the run): the violation far from the data falls 3–5 times at s ≤ 0.005 (−420 → −143 at s = 0.002), the
  validation loss is unchanged (6.284 against 6.290 for the same training without the term, row B).
- **It does not fix the energy, and completion is worse.** The test of the holes does not move (at s = 0.002, worst
  20% of poses, legs hidden: 0.88 of the wrong hypotheses below the truth against 0.84); on pose 189 the splits sit
  80–90 nats below the truth (B: about 60). Inpainting at t81: **104.14 mm against 97.02 for B, +7.1 mm [+5.7, +8.7]**,
  worse in all four occlusions, with a higher APD (8.29 against 7.39 cm).
- **Decision.** The term in this form is closed and recorded as a negative result (the user: record it, think further
  about the linear model). Agreement of the levels is necessary for a correct energy, not sufficient, and points of
  isotropic noise miss the region where the sampler goes wrong.

## Setup

- **The equation.** RSGM noises every joint by IGSO(3) with the Brownian time s (isotropic in `fact_RDW` s10:
  `joint_scale` is None); its density is the heat kernel e^{sΔ/2}, Δ with the eigenvalue −l(l+1) on D^l. So
  ∂_s p = Δp / 2 and, for E_s = −log p_s = Ẽ_s / std(s), R = dE/d log s − s/2 (ΔE − |∇E|²) is a constant a(s) per level
  (a score does not fix the constant of an energy).
- **Measurement** (`explore/ace_energy/fp_residual.py`): central differences in float64, ΔE and ∇E along the 57
  directions x_j exp(h e_a) (h = 1e-3), d/d log s by ±0.02. Checked on the IGSO(3) density itself (`--toy`): residual
  0.002 against terms of 28 at every s. R minus its mean near the data, at points of five kinds: "near" (200 training
  poses noised at s), "far 0.2" and "far 1" (the same poses noised at s' = 0.2 or 1, scored at s), and the legs of the
  MLP's and the network's completions on the 40 validation poses where the MLP was worst (a diagnostic only).
- **The term** (`explore/ace_energy/fokker_planck.py`, `embedding.py --fp-weight`): Huber (δ = 5 nats) of R − a(s),
  a(s) learned per bin of log s (started at the mean near residual, its own learning rate 1.0); ∇E from the scaled
  score, ΔE by Hutchinson with central differences (2 probes, h = 0.2 √s within [1e-3, 0.05]), d/d log s by ±0.02;
  in float32 it matches the exact residual (256 probes: within 0.5 nats). Points per step: 256 training poses of the
  batch noised at s, log-uniform over the levels of the loss, and the same poses noised at s' log-uniform in [s, 2].
  The weight 1.5e-3 makes the term about 10% of the loss at the start (term 416, loss 6.3), set before the run.
- **Rows.** Both continue the linear fit P+L (`energy_b_rich_products_jd_r0.001.pt`) with `--embedding fs --k 1`,
  10,000 steps × 2048 poses, Adam 1e-3 cosine, seed 0: B without the term (`energy_lin_pl_adam.pt`, 2026-11-26) and
  the row with the term (`energy_lin_fp.pt`, workstation2, 0.33–0.67 s per step on a shared GPU). The term draws its
  points from the generator of the batches, so the batches of the two rows differ after the first step.
- **A CPU check before** (1000 steps × 512 poses, lr 1e-4, 128 points): the loss as without the term (6.387 against
  6.391), the far violation 5–35% smaller, the test of the holes unchanged.
- **Rule written before the runs:** inpainting at t81 better than B by at least 2 mm with a paired 95% interval
  excluding 0; the residual, the test of the holes and the loss per bin next to it.

## Results: the violation

R − (its mean near the data), nats per unit of log s, mean (rms); 200 points per kind (`fp_residual.log` for the
fits, `fp_residual_gpu.log` for B and the row with the term). MLP K8 before any term:

| s | near (rms) | far 0.2 | far 1 | MLP's legs, worst poses | network's legs, same poses |
|---|---|---|---|---|---|
| 0.0005 | 9.7 | −115 (158) | −571 (647) | +0.0 (18.7) | −2.6 (14.7) |
| 0.002 | 4.0 | −62 (90) | −412 (461) | +5.8 (18.1) | +0.7 (6.4) |
| 0.005 | 4.9 | −35 (49) | −332 (371) | +1.7 (9.7) | −0.8 (5.7) |
| 0.02 | 1.8 | −13 (16) | −185 (198) | +0.1 (2.2) | −0.4 (2.2) |
| 0.05 | 1.4 | −8 (10) | −87 (93) | −0.3 (2.1) | −0.4 (1.7) |
| 0.2 | 0.5 | 0.0 (0.6) | −9 (12) | −0.3 (0.5) | −0.3 (0.5) |
| 1 | 0.5 | −0.1 (0.6) | 0.0 (0.4) | −1.8 (2.0) | −1.9 (2.0) |

- The violation vanishes where a point becomes data for its level (far 0.2 at s ≥ 0.2, far 1 at s = 1): the levels
  are trained only near their own data. P+L gives the same picture.

B against the row with the term:

| s | near rms, B / term | far 0.2, B / term | far 1, B / term | MLP's legs (rms), B / term |
|---|---|---|---|---|
| 0.0005 | 6.1 / 3.8 | −85 / +31 | −551 / −114 | 14.8 / 17.2 |
| 0.002 | 4.2 / 2.1 | −58 / +4 | −420 / −143 | 17.6 / 20.2 |
| 0.005 | 1.8 / 1.7 | −37 / +7 | −352 / −104 | 9.0 / 12.9 |
| 0.02 | 1.4 / 1.1 | −15 / −2 | −181 / −105 | 1.8 / 2.6 |
| 0.2 | 0.6 / 0.5 | 0.0 / +0.1 | −13 / −14 | 1.7 / 1.4 |

- The term removes most of the violation at its own points; at the legs of the hard completions it does not.

## Results: the holes and completion

Share of the hypotheses whose energy lies below the truth's, legs hidden (`gt_energy.py --shares`; "own": the fixed
completions of P+L, as these energies have none of their own), mean over the 200 poses and over the worst 20%:

| energy | s | own below the truth | worst 20% | network's below the truth | worst 20% |
|---|---|---|---|---|---|
| B | 0.002 | 0.69 | 0.84 | 0.68 | 0.73 |
| with term | 0.002 | 0.70 | 0.88 | 0.67 | 0.72 |
| B | 0.02 | 0.59 | 0.59 | 0.65 | 0.75 |
| with term | 0.02 | 0.59 | 0.59 | 0.65 | 0.75 |

Pose 189 at s = 0.002 (`gt_energy_gpu.log`): with the term the truth −145 nats, the splits of P+L −225 … −235;
B: truth −418, splits −467 … −479.

Inpainting at t81, the first 200 validation poses, K = 10 (`diag5/inpaint_lin_pl_adam.pt`,
`diag6/inpaint_lin_fp.pt`):

| occlusion | B, MPVPE mm | with term | B, APD cm | with term |
|---|---|---|---|---|
| left leg | 98.54 | 106.12 | 8.45 | 9.62 |
| legs | 117.73 | 124.26 | 9.98 | 11.12 |
| arms | 122.11 | 126.33 | 8.40 | 8.95 |
| trunk | 49.69 | 59.84 | 2.72 | 3.46 |
| **mean** | **97.02** | **104.14** | 7.39 | 8.29 |

- Paired over the poses: +7.12 mm [+5.69, +8.71]. The hypothesis of lowest energy at s = 0.02 gives 90.65 mm.

## Conclusions

1. **The levels of the ACE energies disagree far from the data, by hundreds of nats at small s.** This is the measured
   form of the trouble at small noise: denoising score matching never constrains a level away from its own data.
2. **Making them agree is not enough.** The heat equation does not fix a small level from the larger ones (the
   backward problem is ill-posed); the term satisfies it mostly by reshaping the energy where its points are. Points of
   isotropic noise in 57 dimensions do not reach coherent wrong poses like both legs spread, so the region where the
   sampler goes wrong is untouched, and the energy got flatter where it matters (higher APD, completion +7 mm).
3. **Closed in this form.** Left open: the same term at points the sampler itself visits (its trajectories on training
   poses) — not run; a smaller weight; the MLP.

## What it does not show

- One weight, one seed, one point design; the MLP was not tried.
- No harness of generation for the row with the term.
- The test of the holes uses fixed completions of P+L, not the energy's own.

## Where things are

- Code: `explore/ace_energy/fp_residual.py`, `explore/ace_energy/fokker_planck.py`, `embedding.py --fp-*`,
  `gt_energy.py --energy`.
- `~/data/experiments/ace/`: `fp_residual{,_toy,_cpu,_gpu}.{log,pt}`; `energy_lin_fp.{pt,json}`, `lin_fp.log`;
  `energy_lin_fp{,0}_cpu.{pt,json}`, `lin_fp{,0}_cpu.log` (the CPU check); `gt_energy_gpu{,_high}.{pt,log}`,
  `gt_energy_cpu.{pt,log}`; `diag6/inpaint_lin_fp.pt`, `diag6_inpaint_lin_fp.log`.
