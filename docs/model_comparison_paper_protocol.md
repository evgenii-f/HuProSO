# RSGM, PoseRFM and the HuProSO3 flow, evaluated with the protocol of the HuProSO3 paper

Log-likelihood, recall and precision of our unconditional pose priors, computed as in the HuProSO3 paper (Dünkel, Salzmann, Pfaff: Tab. 1, 6, 7). The numbers come from `explore/evaluation/model_comparison_loglkhd_recall_precision.ipynb`; its reports are in `cache/model_comparison_loglkhd_recall_precision/`.

## Summary

- **On the same data, RSGM and PoseRFM beat the HuProSO3 flow by 2-6 nats of test log-likelihood**, more than the spread between training seeds of the flow (~0.3-0.6 nats).
- **RSGM full t23 AdamW is the only model that beats real data on both sample metrics.** On recall (3.60) and precision (4.26) it beats the yardstick of real train poses (3.68 / 4.45). Its precision matches the paper's (4.24).
- **PoseRFM has the best density but the weakest samples:** the highest log-likelihood (139.4) and the worst precision of our models (4.70).
- **The flows overfit most:** the highest train log-likelihood, the lowest test log-likelihood.
- **The paper's absolute numbers cannot be reached on our data.** Even real train poses do not reach the paper's test recall, so compare the models with each other, not with the paper.
- **The tuned models have an advantage.** RSGM and PoseRFM in their `_t` variants were tuned with Optuna on our data; the flow runs with the settings of the paper.

## Setup

- **Task.** Unconditional prior on SO(3)^19 (the 19 dynamic SMPL body joints), official AMASS split. Test: Transitions + SSM, all frames (117,679 poses). Train: all frames (14.8M poses).
- **Protocol** (`hp/evaluation/huproso3_protocol.py`), the same sizes and units as the paper:
  - **recall**: 1k dataset poses against 100k model samples, the distance of each dataset pose to its nearest sample;
  - **precision**: 1k model samples against 100k dataset poses, the distance of each sample to its nearest dataset pose;
  - **distance**: the geodesic angle summed over the joints (radians, Eq. 9 of the paper), or the MPJPE after forward kinematics (mm, Tab. 6). Reported as mean [median];
  - **log-likelihood**: the mean over 5,000 dataset poses, in nats, relative to the uniform distribution on SO(3)^19 (the scale of Tab. 7).
- **The same poses for every model.** The dataset poses are drawn with a generator seeded by (seed, split, repeat), independent of the model, so every model is evaluated on the same 3 × 5,000 poses for the log-likelihood and the same recall and precision queries and pools. Differences between models are therefore paired.
- **Repeats.** Each number is the mean over 3 independent draws of the dataset poses. The ± is the std over these draws: it measures the noise of the data sample, **not** the variation between training seeds.
- **Yardstick.** 100k real train poses (other subjects) take the place of a model. A model that reproduced the training distribution exactly would score about the same.

The models (one training run each; "best": the kept checkpoint with the lowest validation loss of the EMA weights, `hp.training.best_checkpoint`):

| model | trained on | hyperparameters | weights evaluated |
|---|---|---|---|
| RSGM c30 | every 30th frame | of its paper (De Bortoli et al., SO(3) experiment) | final, 100k steps × 512 |
| RSGM full t23 AdamW | all frames | Optuna trial t23, AdamW | best: 50k of 100k steps × 512 |
| PoseRFM full | all frames | of its paper | best = final: 50k steps × 4,096 |
| PoseRFM full t24 @30k | all frames | Optuna trial t24 | checkpoint 30k × 4,096 (the run broke down at 33.9k; 30k is also its best) |
| flow c30 | every 30th frame | of the HuProSO3 paper | final, 2k steps × 2,000 |
| flow full | all frames | of the HuProSO3 paper | best: 7k of 20k steps × 2,000 |
| flow authors (`id_0900`) | the authors' AMASS | of the HuProSO3 paper | the released checkpoint |

## Caveats

- **The data release.** Our AMASS is the SMPL-X **G** release; the authors most likely used the **N** release. Their checkpoint does not reproduce the paper's numbers on our data: log-likelihood 133.0 against 137.6, precision 4.89 against 4.24. On *our* train split (unseen data for it) it scores 146.6, against 184.7 in the paper.
- **A floor for recall.** The paper's test recall (3.44) is below what real train poses reach on our data (3.68). No model can be expected to reach it here.
- **The log-likelihood is on one scale but computed in different ways:**
  - **flow**: exact change of variables;
  - **RSGM**: the probability flow ODE from t = 2·10⁻⁴ to 1 with 200 midpoint steps and the exact divergence. The steps make it ~0.06 nats too high (measured on RSGM c30). The density at t = 1 is taken as uniform; with β_f = 9-10 of our runs the l = 1 mode is down to 0.7-1.1%, so this bias is negligible;
  - **PoseRFM**: the same ODE backwards from t = 1 − 10⁻³, from an exactly uniform source. The convergence in the number of steps is **not checked**;
  - both ODE models give the density of the data slightly smoothed by the path (at most 0.18° per joint for PoseRFM).
- **Training seeds.** Each model is one training run. Three runs of flow full (seeds 10, 11, 12) differ by 0.3-0.6 nats in validation NLL. For RSGM and PoseRFM the spread between seeds is unknown. Differences within a model family (below ~1 nat) are therefore not established.
- **Tuning.** The `_t` models were tuned with Optuna on our data; the flows were not tuned. Untuned RSGM (c30) and PoseRFM (full) are included for a fairer comparison.

## Results

### Test split

| model | LL ↑ | recall [rad] ↓ | precision [rad] ↓ | recall [mm] ↓ | precision [mm] ↓ |
|---|---|---|---|---|---|
| *paper: HuProSO3* | *137.6* | *3.44 [2.95]* | *4.24 [3.71]* | *45.4 [40.2]* | *61.2 [52.1]* |
| *paper: 6D NF* | – | *3.66 [3.16]* | *4.50 [4.00]* | *46.3 [39.9]* | *65.8 [57.8]* |
| *paper: GAN-S* | – | *3.76 [3.34]* | *4.51 [4.23]* | *50.6 [48.4]* | *68.7 [64.2]* |
| PoseRFM full t24 @30k | **139.38** ± 0.09 | 3.66 [3.26] | 4.70 [4.58] | 43.5 [37.8] | 58.4 [53.5] |
| RSGM full t23 AdamW | 138.39 ± 0.08 | **3.60** [3.22] | **4.26** [4.08] | **43.2** [36.3] | **52.4** [47.3] |
| PoseRFM full | 137.62 ± 0.02 | 3.61 [3.23] | 4.70 [4.56] | 43.4 [37.6] | 58.6 [52.7] |
| RSGM c30 | 137.00 ± 0.04 | 3.70 [3.30] | 4.45 [4.27] | 43.9 [37.5] | 55.1 [49.7] |
| flow c30 | 135.23 ± 0.06 | 3.61 [3.19] | 4.49 [4.20] | 43.2 [36.9] | 56.0 [48.2] |
| flow full | 134.79 ± 0.07 | 3.60 [3.18] | 4.43 [4.18] | 43.3 [36.4] | 54.5 [47.5] |
| flow authors (`id_0900`) | 133.04 ± 0.05 | 3.72 [3.33] | 4.89 [4.58] | 45.0 [38.8] | 63.9 [55.3] |
| real train poses (yardstick) | – | 3.68 [3.32] | 4.45 [4.22] | 43.2 [37.4] | 52.5 [46.1] |

The std of recall and precision over the repeats is 0.02-0.06 rad and 0.2-1.2 mm.

### Train split and the generalization gap

| model | LL train | LL test | gap | implausible train LL (> 250) | recall / precision train [rad] |
|---|---|---|---|---|---|
| *paper: HuProSO3* | *184.7* | *137.6* | *47.1* | – | *2.93 / 3.90* |
| flow full | 171.96 | 134.79 | 37.2 | 0.79% | 3.10 / 3.40 |
| PoseRFM full t24 @30k | 173.05 | 139.38 | 33.7 | 0.15% | 2.97 / 3.46 |
| flow c30 | 167.87 | 135.23 | 32.6 | 0.56% | 3.18 / 3.56 |
| PoseRFM full | 162.21 | 137.62 | 24.6 | 0 | 3.10 / 3.56 |
| RSGM full t23 AdamW | 160.53 | 138.39 | 22.1 | 0 | 3.20 / 3.18 |
| RSGM c30 | 152.97 | 137.00 | 16.0 | 0 | 3.39 / 3.44 |
| flow authors (`id_0900`) | 146.60* | 133.04 | – | 0 | 3.41 / 3.89 |

\* Our train split is unseen data for the authors' checkpoint.

## Observations

- **Test recall barely separates the models.** All models lie within 3.60-3.72 rad, around the yardstick (3.68). Recall is bounded by the gap between the train and the test subjects (Transitions are poses that train hardly has), not by the models. The log-likelihood and precision are the informative metrics.
- **Against the flow on the same data**, RSGM full t23 AdamW gains 3.6 nats of log-likelihood and 4% precision over flow full, and 5.4 nats and 13% over the authors' checkpoint.
- **PoseRFM: a good density, poor samples.** Both PoseRFM models have the best or near-best log-likelihood, but their precision (4.70 rad, 58 mm) is the worst of our models and below the yardstick. The density of real poses is learned well, while a part of the samples is unrealistic. The likely cause is the sampler (100 midpoint steps), not the field; not verified.
- **The flows overfit most.** They reach the highest train log-likelihood but the lowest test log-likelihood. Only the flows (and, slightly, PoseRFM t24 just before its breakdown) give implausible train log-likelihoods above 250: density spikes on training poses. The paper's own gap is larger still (47 nats). RSGM has the smallest gap (16-22 nats).
- **More data did not help the flow, and every N-th frame may be generally better than all frames.**
  - Flow c30 beats flow full on test log-likelihood (135.2 against 134.8), with equal recall and similar precision. It saw fewer training poses (2k × 2,000 against 7k × 2,000 up to the best checkpoint of flow full) and overfits less (gap 32.6 against 37.2).
  - Possible reason: neighbouring frames are near duplicates (up to 120 fps). All frames add little information but weight long, slow sequences more; subsampling decorrelates the batches and acts as a regularizer.
  - For RSGM the effect is neutral: test log-likelihood 137.0 (c30) against 137.2 (full, same hyperparameters; `rsgm_huproso_comparison.ipynb`). So it is not a general rule yet; the stride study (`prior_c10` … `prior_c100`, `stride_comparison`) is the place to confirm it.
- **Mean and median tell different stories.** The test median of RSGM and PoseRFM is ~5 nats above the mean (143.2 against 138.4 for RSGM AdamW), that of the flows ~3 nats. RSGM has a heavier lower tail: a few rare test poses get a very low density while typical poses get a higher one. By the median, RSGM leads the flows by ~5 nats, by the mean by 3.6.
- **The train split does not change the test numbers.** Train = every 30th frame and train = all frames give the same test metrics (e.g. RSGM c30: 3.70 / 4.45 / 137.00 in both). Train recall moved by +0.1 rad for every model while train precision and log-likelihood stayed; the cause is not known.
- **RSGM c30 → RSGM full t23 AdamW (+1.4 nats)** changes the data, the architecture, the optimizer and the checkpoint at once. It cannot be attributed to any one of them.

## Open questions and next steps

- Check the convergence of PoseRFM's log-likelihood in the number of ODE steps (200 / 400 / 800 on ~1,000 test poses), especially for t24 @30k.
- Rerun PoseRFM's precision with more sampling steps: is the sampler the cause?
- Paired per-pose log-likelihood differences with standard errors (the per-pose values of the first repeat are in the reports, `likelihood.values`).
- Seed replicas of RSGM and PoseRFM on all frames (there are `rsgm_c10_t34_*_s10/_s11` for c10).
- Look at the 1% of test poses with the lowest log-likelihood under RSGM.
- Ablate RSGM c30 → t23 AdamW: data, architecture, optimizer, checkpoint.
- Confirm the subsampling effect with the stride study.
