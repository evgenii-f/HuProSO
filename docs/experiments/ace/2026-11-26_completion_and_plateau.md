# ACE energy, 2026-11-26: completion of the new energies, where the gap to the network is, a plateau

Branch `experimental/ace-descriptors`. Validation numbers only; no test was run. Part of the ACE line: [overview](README.md).

## Summary

- **2026-11-26: completion of the new energies, the gap, a plateau**.
  - Inpainting with products + joint limits: 97.49 mm against 85.23 mm for the network (200 poses, K = 10): a gap of
    12 mm instead of 42.9 mm. The products carry it (−31 mm); limits, descriptors, squares and 100k poses per bin
    change it by less than 1 mm, triples by −1.3 mm. Inpainting beats SDS and no-DPS for every prior.
  - The energy ranks hypotheses well: the network's 10 hypotheses, the one of lowest energy chosen, give 79.32 mm.
  - The gap is far from the data: 90% of the loss gap is in the third of the poses farthest from train, and in a few
    deeply bent knees. Against a baseline without information the energy keeps 84–88% of the network's gain in every
    limb.
  - Couplings of distant joints are weaker and less aligned than the network's (cosine 0.33–0.45 at tree distance 10),
    and functions of a pair carry only 17–35% of them. Third joints along the tree, collective channels over all
    joints, child-given-parent densities and 5× more data do not close the gap on validation: the linear family is on
    a plateau.

Questions:

1. how the parts of the basis act on completion;
2. which completion method suits an energy;
3. where the remaining gap to the network lies;
4. whether many-body terms over all joints or more data close it.

All numbers are validation numbers; no test was run.

## Setup

- **Completion** (`complete.py --out`, workstation2 and the Mac):
  - the first 200 validation poses of the completion studies, K = 10, the four occlusions;
  - inpainting at the point t81 of `abl_residual` (DPS guidance 0.0493, RePaint 5), the reverse process of `fact_RDW`
    s10 with the score of the energy;
  - no-DPS (`--no-dps`: replacement of the visible joints) and SDS (`--method dposer`, the point t15) as well;
  - not tuned for the energies.
- **Ranking** (`--rank 2e-3 2e-2 2e-1`): the energy of each of the K completed poses at three noise levels s; the
  hypothesis of lowest energy is chosen without the ground truth. The network's own hypotheses (a Mac run of
  `--network`) are ranked with the energy P+L 1e-3.
- **Rows**: the fits of 2026-11-25, named by their parts. "P+L" is products + joint limits with ridge 1e-3, "t5"
  adds the triples within tree distance 5. "No descriptors" is P+L without the distance, angle and twist
  descriptors (branch `experimental/ace-radial-ablation`). "Squares" adds the quadratic forms of one joint times the
  coordinates of another. "100k" is P+L fitted on 100,000 training poses per bin instead of 20,000.
- **Paired differences**: of the mean error over the K hypotheses, per pose (mean over the 4 occlusions), with a
  bootstrap 95% interval over the 200 poses. Rule fixed before: a part of the basis "matters" when its interval
  excludes 0 and the difference exceeds 2 mm.
- **Distance to train**: the mean joint angle between a validation pose and its chordally nearest training pose (all
  1.86M c10 poses). The "far third" is the third of the poses with the largest distance.
- **Analyses**: `explore/ace_energy/diagnose.py` (`completion`, `loss`, `couplings`, `pairbasis`; CPU) and
  `explore/ace_energy/collective_test.py` (the residual tests).

## Results: completion by the parts of the basis

Inpainting, 200 validation poses, K = 10, MPVPE in mm; gap to the network paired (`diagnose_completion.log`).

| row | mean | best of 10 | > 200 mm | left_leg | legs | arms | trunk | gap to the network |
|---|---|---|---|---|---|---|---|---|
| network `fact_RDW` s10 | **85.23** | 47.32 | 5.1% | 83.1 | 102.7 | 113.0 | 42.2 | – |
| joint limits only | 193.00 | 92.91 | 39.3% | 246.1 | 245.5 | 182.4 | 98.0 | +107.8 [102.9, 112.9] |
| old base (`energy_b_smooth_w123_p123`) | 128.16 | 68.47 | 17.0% | 139.5 | 154.4 | 156.3 | 62.4 | +42.9 [39.0, 47.1] |
| base + limits (one s0), 1e-4 | 126.39 | 67.09 | 16.9% | 138.9 | 152.2 | 155.8 | 58.7 | +41.2 [37.6, 44.8] |
| base + pairs, 1e-4 | 114.02 | 59.69 | 12.8% | 121.2 | 137.6 | 141.4 | 55.9 | +28.8 [25.5, 32.3] |
| products, 1e-4 | 97.04 | 54.57 | 7.8% | 98.3 | 118.3 | 122.9 | 48.6 | +11.8 [9.1, 14.8] |
| P+L 1e-4 | 97.96 | 53.81 | 8.2% | 99.0 | 119.1 | 124.0 | 49.8 | +12.7 [10.1, 15.5] |
| P+L 1e-3 | 97.49 | 53.52 | 7.9% | 100.0 | 117.7 | 122.7 | 49.6 | +12.3 [9.9, 14.8] |
| P+L 1e-2 | 100.00 | 53.85 | 8.2% | 106.1 | 118.6 | 124.9 | 50.4 | +14.8 [12.5, 17.1] |
| no descriptors, 1e-3 | 98.01 | 54.36 | 7.9% | 100.2 | 118.4 | 124.0 | 49.5 | +12.8 [10.5, 15.2] |
| squares, 1e-3 | 98.03 | 53.98 | 8.1% | 100.9 | 118.9 | 122.3 | 50.2 | +12.8 [10.3, 15.6] |
| t5, 1e-3 | **96.19** | **53.09** | 7.4% | 97.7 | 117.6 | 119.6 | 49.8 | +11.0 [8.7, 13.4] |
| 100k, 1e-3 | 97.15 | 53.31 | 7.8% | 100.4 | 117.1 | 122.3 | 48.9 | +11.9 [9.5, 14.5] |

The products of local rotations carry the gain: 128 → 97 mm. Every other part changes the mean by less than the
products' 31 mm by an order of magnitude.

| a − b | all | left_leg | legs | arms | trunk |
|---|---|---|---|---|---|
| products − old base | −31.1 [−34.4, −27.8] | −41.2 | −36.1 | −33.3 | −13.8 |
| base + pairs − old base | −14.1 [−16.9, −11.4] | −18.4 | −16.8 | −14.8 | −6.6 |
| base + limits − old base | −1.8 [−4.1, +0.5] | −0.6 | −2.3 | −0.4 | −3.7 |
| P+L 1e-4 − products 1e-4 (limits) | +0.9 [−0.2, +2.0] | +0.6 | +0.8 | +1.0 | +1.2 |
| P+L 1e-2 − P+L 1e-3 | +2.5 [+1.6, +3.4] | +6.1 | +1.0 | +2.2 | +0.8 |
| no descriptors − P+L 1e-3 | +0.5 [−0.2, +1.2] | +0.2 | +0.8 | +1.3 [+0.5, +2.1] | −0.1 |
| squares − P+L 1e-3 | +0.6 [−0.3, +1.4] | +0.8 | +1.2 | −0.4 | +0.6 |
| t5 − P+L 1e-3 | −1.3 [−2.4, −0.2] | −2.3 | −0.1 | −3.0 [−4.9, −1.3] | +0.3 |
| 100k − P+L 1e-3 | −0.3 [−0.8, +0.1] | +0.4 | −0.6 | −0.4 | −0.7 |

- **Products: −31 mm**, the only part that matters by the rule. The pairs D^1(G_iᵀG_k) are half of it (−14 mm) and
  are contained in the products.
- **Joint limits** help generation (2026-11-25) but not completion (+0.9 mm, interval includes 0). The observed
  joints already keep the hidden ones near the data.
- **The descriptors** (distances, angles, twists) can be dropped: +0.5 mm overall. Only in the arms is it +1.3 mm,
  significant but below 2 mm.
- **Triples (t5)** give −1.3 mm, significant, but below the 2 mm rule; mostly in the arms (−3.0 mm).
- **A stronger ridge hurts**: 1e-2 is +2.5 mm worse, +6.1 mm in left_leg. A ridge of 1e-3 against 1e-4 is level.
- **100k poses per bin** instead of 20k: −0.3 mm, not significant.
- The products alone with ridge 1e-4 (97.04 mm) and P+L 1e-3 (97.49 mm) are level. **Gap to the network: 11–13 mm**,
  against 42.9 mm for the old base. That is below the success rule of 21.5 mm set on 2026-11-25.

## Results: the completion method

Paired change of the mean error against inpainting with the same prior (mm, 200 poses, K = 10).

| prior | inpainting | no-DPS − inpainting | SDS − inpainting |
|---|---|---|---|
| network | 85.23 | +16.5 [+14.4, +18.7] | +4.2 [+2.2, +6.3] |
| old base | 128.16 | +3.1 [+0.3, +6.0] | – |
| P+L 1e-3 | 97.49 | +12.1 [+9.7, +14.5] | +10.5 [+6.7, +14.7] |
| t5 | 96.19 | – | +13.6 [+9.3, +18.2] |

- **Inpainting wins for every prior.** SDS loses more with the energies (+10.5 to +13.6 mm) than with the network
  (+4.2 mm). Its errors have heavier tails: 12.2% of the hypotheses above 200 mm for P+L, against 7.9% with inpainting.
- The step of the guidance was not varied for the energies, except two probes with P+L (0.02 and 0.1 instead of
  0.0493), both worse (`diag2/g0.02_products.pt`, `g0.1_products.pt`).

## Results: ranking the hypotheses by the energy

The hypothesis of lowest energy against the mean of K (mm, inpainting; paired change in brackets).

| hypotheses of | ranked by | s = 0.002 | s = 0.02 | s = 0.2 | mean of K |
|---|---|---|---|---|---|
| network (Mac run) | P+L 1e-3 | **79.32** (−6.5 [−8.6, −4.3]) | 81.10 (−4.7) | 83.01 (−2.8) | 85.84 |
| P+L 1e-3 | itself | 94.43 (−3.1) | 89.39 (−8.1) | 90.76 (−6.7) | 97.49 |
| t5 | itself | 94.81 (−1.4) | 87.18 (−9.0 [−11.4, −6.7]) | 88.45 (−7.7) | 96.19 |
| P+L SDS | itself | 102.00 (−6.0) | 97.58 (−10.5) | 98.67 (−9.4) | 108.03 |

- The energy is a good judge of poses: ranking lowers the error of every prior's hypotheses by 3–13 mm.
- The network's hypotheses ranked by the energy (79.32 mm) are better than the energy's own, ranked (87.18 mm). The
  energy is better at judging poses than at making them. The network ranked is 6.5 mm ahead of the network unranked
  (the Mac run, 85.84 mm, matches the reference run 85.23 within its noise: +0.6 [−0.5, +1.7]).
- For the energy's own hypotheses s = 0.02 ranks best, for the network's s = 0.002.

## Results: where the gap is

Gap to the network (mm, inpainting, mean of K), by the distance of a pose to its nearest training pose (thirds:
5.5–8.7°, 8.7–11.1°, 11.1–19.3°) and, in the occlusion left_leg, by the flexion of the left knee (thirds: 3–15°,
15–32°, 33–128°).

| row | near | middle | far | legs: near | middle | far | left knee: straight | middle | bent |
|---|---|---|---|---|---|---|---|---|---|
| old base | 31.0 | 33.0 | 65.1 | 39.3 | 37.8 | 78.5 | 39.5 | 50.1 | 80.2 |
| products 1e-4 | 7.0 | 5.9 | 22.7 | 5.7 | 5.8 | 35.6 | 8.5 | 11.3 | 26.2 |
| P+L 1e-3 | 8.6 | 8.0 | 20.3 | 8.6 | 8.3 | 28.3 | 10.9 | 14.4 | 25.7 |
| P+L 1e-2 | 12.2 | 10.9 | 21.3 | 12.2 | 10.5 | 25.3 | 16.8 | 20.4 | 31.9 |
| t5 | 7.5 | 6.9 | 18.6 | 8.0 | 6.8 | 30.1 | 12.3 | 12.9 | 18.8 |
| 100k | 8.5 | 7.9 | 19.5 | 7.9 | 8.2 | 27.3 | 11.2 | 14.6 | 26.2 |

- **The gap is in the poses far from the training data**: 19–23 mm in the far third against 6–9 mm in the others; in
  the legs occlusion 27–36 against 6–9 mm.
- **The gap grows with knee flexion**: the energies under-bend the knee.
- Per-limb gaps in mm mislead without a baseline. Against one without information (the hidden joints of random
  training poses: 193.2, 195.6, 191.7 and 90.4 mm in left_leg, legs, arms, trunk), P+L 1e-3 keeps 84–88% of the
  network's gain in every occlusion (t5 84–92%, the old base 44–58%). The legs are not worse modelled than the arms;
  the poses with hidden legs are harder for both priors.

## Results: the loss gap by distance to train and by knee flexion

`diagnose.py loss`: 4000 validation draws per bin, the 10 bins with s ≤ 0.04, mean over the bins; the gap of a draw is
the sum over joints of the energy's squared error minus the network's (`diagnose_loss.log`). The knee columns are the
mean gap of one knee, by its flexion in the clean pose.

| fit | gap, all | far third | the rest | knee 0–18° | 18–66° | > 66° | share of the > 66° gap in its worst 2% |
|---|---|---|---|---|---|---|---|
| P+L 1e-3 | 1.163 | 3.150 | 0.153 | −0.002 | 0.058 | 0.344 | 71% |
| t5 | 1.458 | 3.951 | 0.191 | −0.002 | 0.054 | 0.348 | 70% |
| P+L 1e-2 | 0.957 | 2.234 | 0.307 | −0.011 | 0.043 | 0.247 | 75% |
| 100k | **0.914** | 2.587 | **0.063** | −0.009 | 0.046 | 0.308 | 76% |

- **90% of the loss gap is in the far third** (P+L 1e-3: 3.150 / 3 of 1.163). On the other two thirds the energy is
  within 0.15 of the network.
- **Bent knees**: the gap of a knee is zero when straight and 0.34 beyond 66°, and 2% of the bent knees carry 70–76%
  of it: a few rare poses, as pose 2216 of 2026-11-24.
- **Loss and completion disagree** between fits: t5 has the larger loss gap (1.458) and the better completion
  (96.19 mm), ridge 1e-2 the smaller gap far from train (2.234) and the worse completion (+2.5 mm). 100k lowers the gap
  most (−0.25), mostly in the far third, and moves completion by −0.3 mm.

## Results: couplings of joints along the tree

`diagnose.py couplings`: the blocks ∂s_j/∂x_i (3 × 3, s the scaled score, central differences of 0.03 rad) at 240
noisy validation poses, by the distance of i and j along the tree; the ratio of the rms of a model's blocks to the
network's, and the cosine between them (`diagnose_couplings.log`). A second network (`fact_N` s10, another
architecture) shows how far two good score models agree.

| tree distance | pairs | network rms | P+L 1e-3 ratio / cos | t5 | old base | network `fact_N` |
|---|---|---|---|---|---|---|
| s = 0.005: 1 | 16 | 0.856 | 0.92 / 0.88 | 0.93 / 0.87 | 0.71 / 0.62 | 1.08 / 0.94 |
| 2 | 19 | 0.613 | 1.00 / 0.83 | 1.02 / 0.83 | 0.95 / 0.73 | 1.07 / 0.91 |
| 5 | 24 | 0.287 | 0.71 / 0.53 | 0.79 / 0.51 | 0.35 / 0.10 | 1.09 / 0.67 |
| 8 | 15 | 0.412 | 0.82 / 0.75 | 0.80 / 0.75 | 0.09 / 0.33 | 1.09 / 0.83 |
| 10 | 4 | 0.294 | 0.55 / 0.44 | 0.53 / 0.45 | 0.06 / 0.02 | 1.08 / 0.65 |
| s = 0.05: 1 | 16 | 0.225 | 0.99 / 0.79 | 0.98 / 0.79 | 0.91 / 0.54 | 1.04 / 0.91 |
| 2 | 19 | 0.149 | 0.97 / 0.70 | 1.00 / 0.71 | 1.08 / 0.56 | 1.05 / 0.86 |
| 5 | 24 | 0.094 | 0.70 / 0.46 | 0.79 / 0.46 | 0.41 / 0.09 | 1.03 / 0.68 |
| 8 | 15 | 0.129 | 0.84 / 0.70 | 0.81 / 0.70 | 0.12 / 0.23 | 1.04 / 0.80 |
| 10 | 4 | 0.083 | 0.60 / 0.33 | 0.55 / 0.33 | 0.05 / 0.04 | 1.00 / 0.58 |

(all distances 1–10 are in the log; the rows shown are representative.)

- The products give the energy couplings between distant joints, which the old base lacked (cosine 0.02–0.33 at
  distance 5–10, now 0.33–0.75).
- Near joints are coupled almost as the network couples them (ratio 0.92–1.00, cosine 0.70–0.88). With distance both
  the strength (0.55–0.60 at distance 10) and the direction (cosine 0.33–0.45) fall.
- Two networks agree better (cosine 0.58–0.65 at distance 10) but not fully: part of the far couplings is not fixed by
  the data. Triples do not change the couplings.

`diagnose.py pairbasis --third`: can a function of the two joints alone carry the network's coupling of a pair? The
network's symmetrised block at s = 0.02 is regressed on the products of the derivatives of D^1 of the two joints (81
columns, the form the products of fit.py give), with D^2 (1156 columns), and with each third joint's three principal
local coordinates times the 81 columns; 4500 validation poses fit, 1500 test, test R^2 (`diagnose_pairbasis.log`;
3 random pairs per distance).

| tree distance | D^1 × D^1 | + D^2 | best third joint | third joints on the path | off the path | all third joints |
|---|---|---|---|---|---|---|
| 1 | 0.38–0.73 | +0.01 to +0.04 | +0.02 to +0.06 | – | +0.01 | +0.07 to +0.15 |
| 2 | 0.60–0.92 | +0.00 to +0.03 | +0.01 to +0.08 | +0.00 to +0.08 | +0.00 to +0.01 | +0.02 to +0.14 |
| 5 | 0.26–0.33 | +0.03 to +0.04 | +0.04 to +0.05 | +0.02 to +0.03 | +0.01 to +0.02 | +0.14 to +0.17 |
| 8 | 0.28–0.35 | +0.01 to +0.04 | +0.03 to +0.05 | +0.01 to +0.02 | +0.01 to +0.03 | +0.13 to +0.19 |
| 10 | 0.17–0.26 | +0.02 | +0.04 to +0.05 | +0.01 to +0.02 | +0.02 | +0.15 to +0.18 |

- **A pair basis explains 38–92% of the network's coupling of joints at distance 1–2 and 17–35% of distant
  ones.** A higher degree of the pair (D^2) adds only 0–4 points.
- **No single third joint carries the rest**, and the joints on the tree path between the two are no better than the
  others. All 17 third joints together add 0.07–0.19. The coupling of distant joints in the network depends on the
  whole pose, spread over many joints; it is not a chain along the tree.
- So clusters along the kinematic tree (triples by tree distance, t5) are not the missing piece, as their completion
  (−1.3 mm) and couplings (unchanged) show.

## Results: many-body terms over all joints (residual tests)

A test without a new fit (`collective_test.py`, as `pair_test.py` of 2026-11-25): the residual of P+L 1e-3 (target
minus its scaled score) is regressed on the gradients of candidate features on 16,000 training draws per bin, and the
drop of the loss is measured on 4000 validation draws per bin; the 10 bins below s = 0.04, mean over the bins. Rule
fixed before: a candidate is worth a fit when it lowers the validation loss gap by at least 0.2 (of 1.131; 3.062 in
the far third). Ridge 1e-3, 1e-2, 1e-1 relative to the mean diagonal; the best of the three is shown.

| candidate | features | drop on train | drop on validation | drop in the far third |
|---|---|---|---|---|
| collective channels, cubic, K = 20 main components | 1540 | 0.154 | −0.054 | −0.119 |
| the same with K = 20 random directions (control) | 1540 | 0.123 | +0.003 | +0.006 |
| cubic, K = 12 | 364 | 0.041 | −0.015 | −0.036 |
| quartic, K = 10 | 715 | 0.065 | −0.037 | −0.115 |
| pair mixtures of parent and child, bandwidth ×1 | 16 | 0.118 | −0.143 | −0.382 |
| the same, ×3 | 16 | 0.064 | −0.063 | −0.172 |
| the same, ×10 | 16 | 0.013 | −0.014 | −0.043 |

- **Collective channels** A_r = Σ_j Σ_a W_{r,ja} u_{ja} (u the three principal local coordinates of a joint, W the main
  components of the 57 numbers, every channel of unit variance) and their products A_r A_s A_t couple all joints at
  once, a linear analogue of a global aggregation. They fit the training residual (drop 0.15) and make validation
  worse (−0.05, far third −0.12): **negative**.
- **The density of a child given its parent** (log p(x_i, x_k) − log p(x_i) − log p(x_k), mixtures over 5000
  training poses): the same, worse on validation the sharper the mixture: **negative**.
- What the energy lacks on validation is not in these directions. What they learn on train is the training poses
  themselves.

## Results: more data (100,000 poses per bin)

The fit P+L 1e-3 on 100,000 training poses per bin instead of 20,000 (workstation2, `rich_jd_100k.log`).

- Loss gap to the network, s ≤ 0.04: see the loss table above; it is the one place where the fit improves by more
  than the noise.
- Generation (harness, workstation2 GPU, 1000 samples, batch 500; `generate_100k.log`):

| model | Wasserstein | recall | precision | FID_val | APD (cm) |
|---|---|---|---|---|---|
| network `fact_RDW` s10 | 14.86 | 13.51 | 12.79 | 0.0337 | 14.88 |
| P+L 1e-3, 20k | 15.93 | 14.26 | 13.98 | 0.0360 | 15.19 |
| P+L 1e-3, 100k | 15.92 | 14.25 | 13.99 | 0.0364 | 15.26 |
| P+L 1e-4, 100k | 15.94 | 14.21 | 14.05 | 0.0356 | 15.28 |
| training poses in the place of the samples | 15.15 | 13.46 | 13.08 | – | – |

- Generation and completion (−0.3 mm) are level with 20k by the 0.15° and 2 mm rules. The loss and the tasks
  disagree again, as in 2026-11-25 and as for the MACE-like graph network of `experimental/wigner-inputs` (best loss,
  worse completion).

## Conclusions (2026-11-26)

1. **The products of local rotations make the linear energy a usable completion prior.** Inpainting with P+L is 97.49
   mm against the network's 85.23 mm (old linear energy 128.16 mm): a gap of 12 mm, below the 21.5 mm rule. The
   products carry all of the gain (−31 mm); joint limits, the descriptors and squares do not matter for completion,
   triples give −1.3 mm.
2. **Inpainting is the method for an energy.** SDS and replacement without DPS are 10–14 mm worse.
3. **The energy is a better judge than a generator.** It lowers the error of the network's own hypotheses to 79.32 mm
   when it picks one of the 10 (s = 0.002), 6.5 mm below the network's mean of K and 8 mm below the energy's own best
   pick (87.18 mm). A network to propose and an energy to choose is the strongest combination seen here (validation, 200
   poses, the noise level chosen on them).
4. **The gap is far from the data.** 90% of the loss gap and most of the completion gap are in the third of the poses
   farthest from train, and in a few deeply bent knees. Relative to a baseline without information the energy keeps
   84–88% of the network's gain in every limb.
5. **A plateau for the linear family.** Couplings of distant joints are weaker and less aligned than the network's;
   functions of a pair cannot carry them, and neither third joints along the tree, collective channels over all
   joints, densities of a child given its parent, nor 5× more data close the gap on validation. Loss and tasks disagree
   again (t5, ridge 1e-2, 100k), so the loss alone does not choose.
6. **Next**, to be discussed: the energy as a ranker of the network's hypotheses (a completion protocol of its own,
   tuned on validation, then one test); the mixing of the energy and the network scores deferred earlier
   (`~/data/experiments/ace/mix_check`); a nonlinear embedding on the products basis. The linear basis is closed.

## What it does not show (2026-11-26)

- 200 poses, one run per row, the inpainting point of a network; the energies' own completion settings were not tuned
  (two guidance probes only).
- The parts of the basis were compared on the same 200 poses that judge them; with 15 rows some differences near the
  rule are chance.
- The ranking was measured on the same 200 poses and the noise level of the ranking was chosen on them.
- The residual tests probe directions in a fixed fit; a joint refit with the candidates can differ, though the train
  drop bounds what it can give.
- One network for the couplings, a second one only for their noise; the couplings are finite differences of 0.03 rad.

## Where things are

- **2026-11-26** (Mac, `~/data/experiments/ace/`):
  - code: `explore/ace_energy/diagnose.py` (`completion`, `loss`, `couplings`, `pairbasis --third`; its logs
    `diagnose_{completion,loss,couplings,pairbasis}.log`), `explore/ace_energy/collective_test.py`
    (`collective_test.log`, `collective_test.pt`; run with `--train 16000 --valid 4000 --ridge 1e-3 1e-2 1e-1 --candidates cubic:20:pca
    cubic:20:random cubic:12:pca quartic:10:pca pairs:1 pairs:3 pairs:10`);
  - completion files of `complete.py --out` (workstation2 unless noted): `diag/` (network, old base, no-DPS),
    `diag2/` (P+L 1e-3 and t5 with inpainting, SDS and no-DPS, two guidance probes; `inpaint_network_rank.pt` and
    `sds_network_rank.pt`, the network's hypotheses ranked by P+L, Mac), `diag3/` (the 8 rows of the basis parts),
    `diag4/inpaint_products_jd_100k.pt`; logs `diag2.log`, `diag3.log`;
  - the 100k fits `energy_b_rich_products_jd_100k_r{0.001,0.0001}.pt` (`rich_jd_100k.log`) and their harness
    `generate_100k.log`, samples in `generate_100k/`.
