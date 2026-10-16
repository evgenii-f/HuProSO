# Hyperparameter tuning of RSGM

How the hyperparameters of RSGM (the Riemannian score-based generative model on SO(3)^19, `src/hp/models/rsgm.py`) were tuned, what came out, and what it does and does not show. The search is in `explore/training/rsgm_hyperparameter_tuning.ipynb`; the studies are in `~/data/experiments/rsgm_hpo/optuna.db` (view them with `optuna-dashboard sqlite:///$HOME/data/experiments/rsgm_hpo/optuna.db`).

## Summary

- Two rounds of Optuna (TPE with a median pruner), 50 trials in 5.1 hours on an Apple GPU (mps), each trial 30k training steps on every 30th frame of AMASS.
- The objective, the validation log-likelihood, went from **131.74** (the default configuration) to **138.08** (round 1) and **138.29** (round 2). The gain of round 2 is within the noise: the search has reached a plateau for this model family and budget.
- The main lever is **capacity**: every good configuration has width 1024 (the default has 512). Then a **higher learning rate** (1e-3 ... 1.6e-3 instead of 2e-4), silu, and β_f between 6 and 9.
- On the test split, at the full budget (100k steps, all frames), the best round-1 configuration (trial 23) improves the default RSGM from 137.15 to **138.16** nats, and beats the real-data contrast on Wasserstein, recall and precision. The gain on the test (+1.0) is much smaller than at 30k steps on the validation poses (+5.6): the default catches up with a longer training.

## Setup

**Data.** Every 30th frame of our local AMASS (SMPL-X G release, `~/data/AMASS_rotations_c30`), official split (`ORIGINAL_AMASS_SPLITS`): 497,707 training poses. Every 30th frame was chosen because it costs nothing in quality (the HuProSO3 flow scores the same on every 10th and every 30th frame, RSGM nearly the same on every 30th frame and on all frames: 136.97 and 137.15) and keeps the data small.

**Objective.** The mean log-likelihood (nats, relative to the uniform distribution on SO(3)^19) of 2,000 fixed validation poses, 500 from each valid dataset (HumanEva, HDM05, SFU, MoSh), under the EMA weights after 30,000 training steps, with 200 steps of the probability flow ODE.
- The validation **loss** is not usable as an objective: its value depends on the hyperparameters themselves (the noise schedule, `loss_t_min` and the weighting change what is averaged), so it does not compare between configurations.
- The validation poses are **stratified**: HDM05 is 79% of the valid split and would otherwise dominate.
- **200 ODE steps**: with 100 the likelihood is ~0.3 nats too high, and the error depends on the model, which could favor some configurations; with 200 it is ~0.06.
- The **test split** was not used in the search.

**Search.** TPE sampler (multivariate, 10 random start-up trials), median pruner (8 start-up trials) that checks the likelihood of 1,000 validation poses (100 ODE steps) at 10k and 20k steps. Fixed in both rounds: `eps` = 2e-4 (it defines what log p means), β_0 = 0.001, the input (rotation matrices and t), gradient clipping 1.0.

Two features were added to the code for the search: a cosine learning-rate schedule with warm-up (`TrainConfig.scheduler = "cosine"`, `warmup_steps`) and the likelihood weighting of the loss (`RSGM(weighting="likelihood")`: the loss weighted by β(t), a bound on the negative log-likelihood after Song et al. 2021, implemented by importance sampling of t). Both are off by default.

## Round 1: a broad search

40 trials (35 finished, 5 pruned), 4.1 hours. The default configuration was the first trial.

| parameter | range | default |
|---|---|---|
| `lr` | 5e-5 ... 2e-3 (log) | 2e-4 |
| `batch_size` | 256, 512, 1024 | 512 |
| `width` × `depth` of the MLP | 256 / 512 / 1024 × 3 ... 6 | 512 × 5 |
| `activation` | sin, silu | sin |
| `beta_f` | 6 ... 16 | 10 |
| `loss_t_min` | 1e-4 ... 1e-2 (log) | 1e-3 |
| `ema_decay` | 0.999, 0.9995, 0.9999 | 0.999 |
| `scheduler` | step, cosine | step |
| `weighting` | std, likelihood | std |

The five best (objective after 30k steps):

| trial | objective | width × depth | lr | β_f | loss_t_min | parameters |
|---|---|---|---|---|---|---|
| 34 | 138.08 | 1024 × 4 | 1.48e-3 | 8.08 | 1.7e-3 | 3.4 M |
| 36 | 137.90 | 1024 × 5 | 1.62e-3 | 7.43 | 1.4e-3 | 4.4 M |
| 39 | 137.86 | 1024 × 4 | 9.4e-4 | 8.37 | 1.9e-3 | 3.4 M |
| 32 | 137.84 | 1024 × 4 | 1.37e-3 | 8.12 | 5.5e-3 | 3.4 M |
| 35 | 137.84 | 1024 × 4 | 1.54e-3 | 7.82 | 1.4e-3 | 3.4 M |
| 0 (default) | 131.74 | 512 × 5 | 2e-4 | 10 | 1e-3 | 1.2 M |

All five use silu, batch 512, EMA 0.999, the step schedule and the std weighting.

Mean objective by value (finished trials):

| parameter | values (n, mean) |
|---|---|
| width | 256: 5, 124.2 · 512: 2, 124.2 · **1024: 28, 136.4** |
| depth | 3: 15, 135.3 · **4: 8, 136.6** · 5: 9, 134.1 · 6: 3, 120.0 |
| batch | 256: 6, 126.7 · **512: 25, 136.4** · 1024: 4, 129.8 |
| EMA | **0.999: 25, 136.0** · 0.9995: 4, 132.1 · 0.9999: 6, 126.9 |
| scheduler | cosine: 6, 123.1 · step: 29, 136.2 |
| weighting | likelihood: 6, 130.5 · std: 29, 134.7 |
| activation | silu: 21, 134.1 · sin: 14, 133.8 |

At width 1024 the objective rises with the learning rate (correlation with log lr +0.73) and falls with β_f (−0.74); the best trials pressed against the upper end of the width range, the upper end of the lr range and the lower end of the β_f range.

## Round 2: around the best of round 1

10 new trials (5 finished, 5 pruned), 1.0 hour, in a new study warm-started with the 12 round-1 trials that lie in its space (their values added without retraining).

| parameter | round 2 |
|---|---|
| `width` | 1024 / 1536 / 2048 |
| `depth` | 3 ... 5 |
| `lr` | 6e-4 ... 4e-3 (log) |
| `beta_f` | 4 ... 10 |
| `loss_t_min` | 1e-3 ... 1e-2 (log) |
| fixed | silu, batch 512, EMA 0.999, step, std |

| trial | objective | width × depth | lr | β_f | loss_t_min |
|---|---|---|---|---|---|
| 12 | **138.29** | 1024 × 5 | 1.60e-3 | 6.00 | 1.4e-3 |
| 21 | 138.16 | 1024 × 5 | 1.06e-3 | 6.97 | 2.3e-3 |
| 19 | 138.12 | 1024 × 5 | 1.46e-3 | 8.78 | 3.2e-3 |
| 7 (round-1 trial 34) | 138.08 | 1024 × 4 | 1.48e-3 | 8.08 | 1.7e-3 |
| 20 | 138.07 | 1024 × 5 | 1.51e-3 | 8.96 | 2.7e-3 |

- **Depth** at width 1024: 3 → 137.2, 4 → 137.8, 5 → 138.1 on average (correlation +0.81).
- **Learning rate**: three of the five pruned trials had lr ≈ 2e-3 or more; the optimum is ~1e-3 ... 1.6e-3.
- **β_f**: 5.2 ... 9 work alike (5.24 gave 137.9); 4.4 and 5.0 were pruned.
- **Width**: TPE never proposed 1536 or 2048 (all warm-start trials had 1024, and the sampler stayed there). Whether wider networks help is still open.

## From the search to the test

The search ranks configurations after 30k steps on validation poses. The real question is the test split at the full budget. Evaluation with the harness of the model zoo (`explore/evaluation/rsgm_huproso_comparison.ipynb`) on the full official test split (Transitions + SSM, 117,679 poses); RSGM likelihoods with 200 ODE steps; angles in degrees per joint; ± over 3 evaluation repeats (the sampling of the evaluated poses, not training seeds).

| model | trained on | log-likelihood | Wasserstein | recall | precision |
|---|---|---|---|---|---|
| RSGM, trial 23 of round 1 | all frames, 100k steps | **138.16 ± 0.17** | **17.16** | **11.53** | **12.99** |
| RSGM, default | all frames, 100k steps | 137.15 ± 0.11 | 17.59 | 11.72 | 13.53 |
| RSGM, default | every 30th frame, 100k steps | 136.97 ± 0.11 | 17.58 | 11.71 | 13.48 |
| HuProSO3 flow, retrained by us | every 30th frame, 2k steps | 135.18 ± 0.18 | 17.89 | 11.55 | 13.86 |
| HuProSO3 flow, released `id_0900` | the authors' AMASS | 132.96 ± 0.19 | 19.01 | 12.00 | 14.97 |
| contrast: real poses of other subjects | | – | 17.49 | 11.75 | 13.49 |

- The tuned configuration (trial 23: MLP 1024 × 3, silu, lr 1.06e-3, β_f 8.99) is better than the default on every metric, and its samples are closer to the test poses than real poses of other subjects are (all three sample metrics below the contrast).
- The gain shrinks from the search to the test: +5.6 nats at 30k steps on the validation poses, +1.0 at 100k on the test. A short budget favors configurations that learn fast (large learning rates, short EMA); the default catches up with longer training. The 30k-step differences between the top configurations (a few tenths of a nat) are not expected to survive.
- The best configuration of round 2 (trial 12: 1024 × 5, lr 1.6e-3, β_f 6.0) is being trained at the full budget (`explore/training/rsgm_train.ipynb`, run `rsgm_tuned_r2t12`).

## Caveats

1. **One training seed per configuration.** The ± above are evaluation noise; the spread between training runs of one configuration is unknown. A difference between configurations has to be judged against it: the optional final step of the notebook retrains the best configurations with three seeds.
2. **The 30k-step budget** biases the search towards fast learners (see above). Parameters that act late (EMA 0.9999, cosine decay) are disadvantaged.
3. **Under-explored options.** The cosine schedule and the likelihood weighting were only tried with small networks in the random start-up of round 1 (one trial each at width 1024: cosine with EMA 0.9999 → 132.0, likelihood weighting → 136.3, the third best at the time). TPE then stopped proposing them. No conclusion about them is justified.
4. **Parameter importances (fANOVA) are unreliable** here: with ~35 correlated trials they rank β_f first (0.57-0.66 depending on the run), because in the random start-up the small widths came with small β_f. The grouped statistics show that the width is what matters.
5. **Small β_f** leaves the end of the noising not quite uniform: the likelihood is then underestimated (~0.2 nats at β_f = 6, ~0.006 at 10) and sampling starts slightly off. The search preferred small β_f despite that bias.
6. **Only RSGM was tuned.** A comparison with the HuProSO3 flow (retrained with 2k steps, not tuned) is fair only for the default RSGM; a tuned RSGM should be reported separately, or the flow tuned too.

## Next steps

- Train the best round-2 configuration at the full budget and compare it on the test split with trial 23 (paired comparison on the same poses).
- Test the open boundaries directly rather than with TPE: width 2048 and depth 6 at width 1024.
- Retrain the final choice (and the default) with three seeds to know the training noise.
- Ablations at width 1024 for the cosine schedule and the likelihood weighting.

## Reproducing

```bash
pip install optuna optuna-dashboard
```

Run `explore/training/rsgm_hyperparameter_tuning.ipynb` top to bottom: the studies are resumable, so with the existing database it only loads them and shows the results; `RUN_FINAL = True` runs the final retraining with seeds (hours). Train a configuration at the full budget with `explore/training/rsgm_train.ipynb` (the best one is printed by the notebook as `TUNED` overrides) and evaluate it with `explore/evaluation/rsgm_huproso_comparison.ipynb`.
