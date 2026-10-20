# Mirror augmentation of RSGM: a negative result

Does left-right mirror augmentation of the training poses make RSGM generalize better to unseen subjects? No: it makes every metric worse, by 8-10 times the training noise. The poses of AMASS are not mirror symmetric, and the test subjects share the asymmetry of the training subjects, so a prior should model it rather than average it away.

The code (`hp/data/augmentation.py`, the `data.mirror` option of the training script, tests) and the notebook of the experiment (`explore/training/rsgm_mirror_experiment.ipynb`) are on the branch `experimental/data_mirroring`; they are not in `dev`. The runs are in `~/data/experiments/rsgm_c10_t34_{plain,mirror}_s{10,11}`, the cached evaluations in `~/data/experiments/rsgm_mirror_experiment/`.

## Motivation

The train/valid gap of the priors is large (flow: 53 nats at the end of its training, `explore/evaluation/training_curves.ipynb`), and neighbouring frames are nearly identical: the data lacks different people and motions, not frames. The mirror image of a pose is a valid pose, and a mirrored training pose is about as far from the training data as a pose of an unseen subject (nearest training pose: 10.3° per joint, against 11.1° for test poses and 4.0° for other training poses), so mirroring looked like a way to add diversity.

## The augmentation

Mirroring the body in the sagittal plane (x is the lateral axis of SMPL) swaps the left and right joints and maps every local rotation R to M R M with M = diag(-1, 1, 1). Per joint this is the conjugation Q R Qᵀ with the rotation Q = diag(1, -1, -1) (by π about x), an isometry of SO(3): the uniform measure, the geodesic distances and the heat kernel are invariant, but a learned density is not, unless the data or the model is symmetric. Each training pose was mirrored with probability 0.5; validation and test poses never.

Checks of the implementation:
- the forward kinematics of a mirrored pose are the mirrored joint positions up to the asymmetry of the SMPL template: 8 mm per joint on average (20 mm at most) on 2,000 real test poses, against 16 mm of asymmetry in the rest pose; mirroring in the y or z plane instead gives errors of meters;
- on the training data, the mean rotation of a left joint and the mirrored mean rotation of its right partner differ by 2-7° (the spread of a joint is 10-25°; the wrong planes give 10-80°), so the mirror is right for every joint, also those whose rotation moves no joint position (head, hands);
- mirroring twice gives the pose back exactly, distances between poses are preserved.

## Setup

2 x 2: with and without the augmentation, two seeds each (10, 11). Every 10th frame of the official train split (1,482,799 poses; validation 132,168), the best trial of round 1 of the hyperparameter search (trial 34: MLP 1024 x 4, silu, lr 1.48e-3, β_f 8.08, Adam, batch 512, EMA 0.999, step schedule), 100k steps, ~24 min per run on mps. Evaluation with the harness of the model zoo on the full test split (117,679 poses), RSGM likelihoods with 200 ODE steps.

## Results

| run | log-likelihood | Wasserstein (°) | recall (°) | precision (°) | valid loss (end) |
|---|---|---|---|---|---|
| plain s10 | 139.09 ± 0.16 | 17.31 | 11.53 | 13.11 | 5.745 |
| plain s11 | 139.24 ± 0.13 | 17.28 | 11.56 | 13.13 | 5.733 |
| mirror s10 | 137.72 ± 0.12 | 17.78 | 11.91 | 13.59 | 5.792 |
| mirror s11 | 137.53 ± 0.12 | 17.74 | 11.93 | 13.57 | 5.773 |
| contrast (real poses of other subjects) | – | 17.49 | 11.75 | 13.49 | |

- **Effect of the augmentation:** -1.5 nats of log-likelihood, 0.4-0.5° worse on all sample metrics (worse than the contrast), higher validation loss, on both seeds.
- **Training noise** (seed 11 - seed 10, measured here for the first time): 0.15-0.2 nats and 0.02-0.04°. The effect is 8-10 times larger. (So differences of a few tenths of a nat between single runs, e.g. AdamW vs. Adam, +0.36, are at the edge of the noise.)
- The plain runs are the best stable RSGM so far (139.1-139.2; t23 on all frames: 138.2).

## Why: the asymmetry is real

Before the experiment, the trained priors were found to be strongly asymmetric: the mirror images of 1,000 test poses get 50 nats (flow, 20k steps on all frames) and 79 nats (RSGM t23) less log-likelihood than the poses themselves, on 100% of the poses, although they are only 1.2° per joint farther from the training data. That could have been an idiosyncrasy of the training subjects; the experiment says it is not: the test poses are of other subjects (Transitions, SSM) and still prefer the orientation of the training data, and a model forced towards symmetry loses likelihood on them. The augmentation moves density to mirrored ("left-handed") poses that do not occur in the test data, and its samples are farther from the test poses.

Whether the asymmetry comes from the motions themselves (handedness) or from systematic effects of the SMPL-X fitting does not matter for a prior: it is in the test data too.

## Open

- Section 4 of the notebook (the log-likelihood of the mirror images of the test poses under the four runs, and the rare poses of `docs/rsgm_rare_poses.json`) was interrupted. It would confirm that the mirrored models are close to symmetric (the augmentation did what it should) and show whether the rare poses gain from it.
- A smaller mirror probability would trade less of the asymmetry for less diversity; given the size of the effect it is not expected to help.
