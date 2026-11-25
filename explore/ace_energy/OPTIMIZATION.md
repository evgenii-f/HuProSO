# Notes for optimising the linear ACE energy

What the basis could lose, with the measurement behind it, for the time when the model is made faster or smaller. Each
note says what was measured, what it does not say, and what is still untested. Background and all numbers of the line:
`docs/experiments/2026-11-23_ace_descriptors.md`. The options `--leave-out` and `--mirror` of `fit.py` used here are on
the branch `experimental/ace-radial-ablation` (558c8fb, ec0a108).

Model of notes 1-3: products + joint density, basis 20354, c10, 20000 poses per bin, 16 bins, ridge 1e-3 (validation;
Wasserstein W and precision in degrees per joint, the low-noise gap is the loss above the network's at s < 0.04,
standard error 0.02). Rows of one note share the noise of their samples; rows of different notes do not.

## 1. The descriptors can go (2436 of 20354 features)

1000 samples, 2026-11-25, `energy_b_abl*`.

| left out | features | W | precision | low-noise gap |
|---|---|---|---|---|
| nothing | 0 | 15.91 | 13.83 | 1.148 |
| distances (Chebyshev of point pairs) | 1080 | 15.99 | 13.91 | 1.162 |
| angles of point triples | 1320 | 15.90 | 13.83 | 1.160 |
| twist | 36 | 15.90 | 13.82 | 1.149 |
| all of them | 2436 | 15.99 | 13.93 | 1.188 |

- All of it costs 0.08 in W, nearly all from the distances; the rule was 0.3. So there is no reason to change the radial
  functions either. (Precision of the first row rescored without the draw bug of the first row.)
- Not measured: the speed without the descriptors, and completion without them.

## 2. Columns that carry no information

- 249 columns are exact duplicates: G_j = x_j for the three children of the root (LeftUpLeg, RightUpLeg, Spine0), so
  D^l(G_j) = D^l(x_j), l = 1, 2, 3 (9 + 25 + 49 = 83 per joint). Found on the harmonics; products not checked.
- The smooth twist (36) is linear in the entries of x_j, that is in D^1(x_j).
- The energy is the same family of functions without them; G^T G is singular with them and ridge repairs it (a
  duplicate pair has half the effective ridge). Removing them changes the layout of the features: old fits need a
  version or a flag.
- D^1(G_i^T G_k) of the other 144 pairs (`--all-pairs 1`, 1296 features) adds nothing on top of the products
  (W 16.83 with both, 16.86 with the products; no joint density, ridge 1e-4).

## 3. The odd half of the weights is the less needed one

The weights split by the left-right mirror into an even and an odd part, w = P_s w + P_a w, P_a = (I - R) / 2, with
R the map of the features under x -> M x_mirror M (exact on the 17918 features without the descriptors, a signed
permutation; 8855 odd directions). `--mirror KAPPA` multiplies the ridge of the odd part by KAPPA. No descriptors,
2000 samples on the GPU, 2026-11-25, `energy_b_mir_*`, `mir.log`, `mir_eval.log`.

| ridge of the odd part | share of the weight norm that is odd | W | precision | low-noise gap |
|---|---|---|---|---|
| x 1 (free) | 51% | 15.54 | 13.60 | 1.188 |
| x 10 | 23% | 15.54 | 13.59 | 1.061 |
| **x 100** | **8.5%** | **15.54** | **13.58** | **1.108** |
| x 1e7 (equal left and right) | 0% | 15.72 | 13.80 | 1.654 |
| x 100 on random directions (control) | (61% odd) | 15.69 | 13.72 | 1.202 |

- The odd part can be shrunk from half of the weight norm to 8.5% for nothing, while the same penalty on as many random
  directions costs 0.15. So the odd directions are the less needed ones. It is a saving and not a gain: no variant
  generates better than the free one.
- It cannot be dropped: exact tying costs 0.18 in W and 0.47 in the gap. The data are not mirror symmetric
  (`docs/experiments/2026-10-20_rsgm_mirror_augmentation.md`).
- Evaluation is not faster by this alone: all features are still computed. Ideas, none tested: drop the odd weights of
  small size after a fit with x 100; keep only the odd weights of the blocks that need them; store the even part
  as half of the coefficients; solve the fit in the even and odd blocks.
- The joint density was not tied. One ridge (1e-3), one draw of samples.

## Costs known so far

- Fit on an RTX-class GPU (workstation2): about 20 s per bin with 6 variants (one solution each) of the 20354 basis,
  30 s with 10. The sums of the products are taken per joint (`Gram`).
- Inpainting with the energy is 25-32 times slower than with the network (ace session, 200 poses, 20-25 min against
  0.8 min); the share of the basis, of DPS and of the joint density in this is not profiled.
