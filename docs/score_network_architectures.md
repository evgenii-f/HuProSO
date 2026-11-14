# Score and flow networks of the pose priors, and a DPoser-like score network for RSGM

Why this note: in pose completion RSGM loses to DPoser with DPoser's own optimization (score distillation at small
noise), also when the optimization is copied one to one and only the denoiser is swapped (the "hybrid" diagnostic,
see below). The heat kernel, the noise geometry and the noise of a single step are ruled out; what remains is how the
network behaves off the data at small noise, which training does not pin down and the architecture does. This note
puts the three networks side by side and proposes the one to try.

## 1. The three networks

### RSGM (ours, `hp/models/rsgm.py`)

```
features = [x_1 ... x_J (9 numbers each, the local rotation matrices)                       9J
            (inputs="local+global": the root-relative rotations of the forward kinematics)  +9J
            t                                                                                 +1
            (time_features=K: sin, cos of pi 2^k t, k < K)                                    +2K
            (a condition: the features of the condition MLP)]                                +F
h = MLP(features): Linear -> act, repeated len(hidden) times, then Linear -> 3J
score(x, t) = h / std(t)          (std trick, std = sqrt(1 - exp(-s(t)))), tangent coordinates in the frame of x
```

- **Time enters once**, as one scalar next to 171 rotation numbers. Every later layer sees t only through what the
  first layer made of it. Small times (t = 0.002 ... 0.03, where completion works) differ by hundredths in that input.
- No normalization, no residual connections, activation silu (sin in the JAX original).
- `small`: hidden 4 x 512, 0.9M parameters. `search`: 6 x 1536, 12.2M.
- Trained with `weighting="std"` (t uniform on [0.002, 1]), 30k steps x 4096, lr 2e-3, AdamW; chosen by an Optuna
  search on the validation log-likelihood.

### PoseRFM (PoseD-Flow, `hp/models/posedflow.py`, their tMLP of manifm)

```
for every layer:  h <- ConcatLinear(h, t) = W h + b + w t        (the time in EVERY layer)
                  h <- TimeSwish(h, t) = h sigmoid(beta(t) h)      (a time-dependent activation, beta(t) a small MLP)
4 layers of 512 on the 21 rotation matrices; the output is the velocity (ambient, projected onto the tangent space)
```

- The same size as RSGM `small`, but **every layer is conditioned on t**, additively and through the activation.

### DPoser (`hp/models/dposer.py`, their TimeFC)

```
temb = SiLU(Linear(sinusoidal_embedding(999 t, 512)))                       a shared time embedding
h    = Dropout(SiLU(GroupNorm(Linear(x) + Linear(temb))))                   x: the z-scored axis-angle, 3J
2 residual blocks:  h <- h + [Dropout(SiLU(GroupNorm(Linear(h) + Linear(temb)))) twice]
out  = Linear(h) / sigma_table(t)    -> the noise prediction; score = -out / std(t)
```

- **Every layer gets its own projection of a rich time embedding** (a 512-dim sinusoidal embedding of 999 t, which
  resolves small times finely), GroupNorm(32) in every layer, residual blocks, dropout 0.1.
- hidden 1024, 2 blocks, embed 512: 7.2M parameters. Trained 200k steps x 5120, lr 2e-4 (our full run); the
  c10 run with RSGM's budget (30k x 4096) is nearly as good in completion (left leg 81.6 mm vs 76.3).
- The input is z-scored per coordinate, so every joint is seen in units of its own spread.

### Side by side

| | RSGM small | PoseRFM | DPoser |
|---|---|---|---|
| parameters | 0.9M | ~0.8M | 7.2M |
| input | rotation matrices, 9J | rotation matrices, 9J | z-scored axis-angle, 3J |
| time conditioning | one scalar at the input | every layer (ConcatLinear + TimeSwish) | every layer (projected 512-dim sinusoidal embedding) |
| resolution of small t | t itself (0.002 vs 0.03) | t itself, but in every layer | sin/cos of 999 t at many frequencies |
| normalization | none | none | GroupNorm in every layer |
| residual | no | no | yes (2 blocks of 2) |
| dropout | no | no | 0.1 |
| output | score / std(t), tangent coordinates | velocity | noise prediction |
| loss weighting | std (likelihood option) | flow matching | DSM with t ~ U(1e-5, 1) |

## 2. What the completion diagnostics say

On validation poses, one-step denoising, RSGM small (std) against DPoser (full), noise matched by its mean angle:

- **Clean poses plus noise**: RSGM denoises as well or better at every noise level and for every joint group.
- **Implausible poses** (the hidden joints at the start of score distillation): the expected step of RSGM points less
  toward the truth (cosine with the direction to the truth about 0.1 lower in the coordinates the optimizer uses).
- **Along the 200 steps of the optimization** (identical for both): RSGM descends slower in the first phase (2-3 deg
  behind at step 100) and drifts away in the second (the smallest noise): +3.2 deg on the left leg against +0.7.
- **The likelihood weighting** (about 34% of training at t <= 0.025 instead of 2%) halves the drift, not the slow first
  phase: the hybrid improves by 3-16 mm and stays 13-29 mm behind DPoser. The c10 run overfits (validation loss rises
  after ~5k steps, train/valid gap 20 vs 25): at small noise the optimal score of finite data points at the nearest
  training pose.

- **A larger RSGM** (`search`, 6 x 1536, 12.2M, std weighting, c10): the hybrid improves by 7-25 mm over the small
  one (98.3 / 122.4 / 130.3 / 46.1 mm against 123.7 / 143.7 / 137.1 / 49.1; DPoser 79.4 / 103.3 / 110.6 / 43.5), and
  the drift on the left leg nearly vanishes (+0.7 deg, as DPoser). The slow first phase stays (2.3 deg behind at step
  100). So the network matters, and capacity alone fixes the drift but not the pull from implausible poses.

Both strong baselines condition every layer on time; ours does not. The two failures (slow pull from implausible
poses, drift at the smallest noise) are both about how the score changes with t at small t, which a scalar t at the
input represents poorly.

## 3. Proposed: a time-conditioned residual score network on SO(3) ("timeres")

The model and its training stay those of RSGM (the IGSO(3) forward process, denoising score matching against the
exact heat kernel, the std trick, the samplers, the likelihood); only the network changes, behind a `net` option:

```
input   x_j as rotation matrices (9J) [+ the root-relative rotations of the forward kinematics, 9J]
noise   c = log(s(t)) / 4          (the log of the Brownian time: small times are spread out; EDM's c_noise)
temb    = SiLU(Linear(sinusoidal_embedding(c, E)))                             shared, E = 256
h       = SiLU(Norm(Linear(input) + Linear_t(temb)))
B residual blocks:  h <- h + Dropout(SiLU(Norm(Linear(h) + Linear_t(temb)))) twice
out     = Linear(h) -> 3J;  score = out / std(t)   (the std trick as now, so the DSM target is unchanged)
```

- **Time in every layer**, from an embedding of log s (the noise level), which separates t = 0.002 and t = 0.02 as
  clearly as t = 0.2 and t = 1.
- **Normalization**: LayerNorm (no batch statistics; GroupNorm(32) is the same up to the grouping). Residual blocks.
- **Size**: hidden 1024, B = 2, about 5-7M parameters, close to DPoser; a `small` variant (512, B = 2) to separate the
  architecture from the size.
- **Dropout** 0.1 as DPoser: a regularizer against the memorization at small noise that the likelihood weighting
  showed.
- **Output last layer initialized to zero**, so the score starts at 0 (a common choice for residual diffusion nets).

Not changed on purpose: the rotation-matrix input (the geometry of RSGM), the std trick and the target, so that the
comparison isolates the network.

## 4. Experiment plan

| run | network | weighting | data | question |
|---|---|---|---|---|
| A | timeres 1024 x 2 | std | c10 | does time in every layer fix completion (both phases)? |
| B | timeres 1024 x 2 | likelihood | full | the network and the training together |
| C | timeres 512 x 2 | std | c10 | the architecture or the size? |

Evaluation, all on validation poses: the hybrid (DPoser's optimization with the RSGM denoiser), the trajectories of
the 200 steps, one-step denoising, and the validation log-likelihood (so that the density is not lost on the way);
then D-Flow and score distillation on the test set with the settings chosen on validation.

Training budget as RSGM (30k x 4096) for A and C, so that a change is the network's; DPoser trains 200k steps, so a
longer run is the next question if A helps.

## 5. Results (c10, std weighting, 30k x 4096)

Hybrid on 500 validation poses (DPoser's optimization, the denoiser swapped), MPVPE mean / APD / best of 10 (mm, cm):

| denoiser | params | left_leg | legs | arms | trunk | valid loss (best) |
|---|---|---|---|---|---|---|
| DPoser | 7.2M | **79.4** / 6.31 / **39.7** | **103.3** / 7.16 / **66.2** | **110.6** / 5.61 / **74.1** | **43.5** / 1.96 / **24.0** | – |
| RSGM MLP small | 0.9M | 123.7 / 10.15 / 60.3 | 143.7 / 11.74 / 85.3 | 137.1 / 8.03 / 88.1 | 49.1 / 2.30 / 25.9 | 5.871 |
| RSGM MLP small, likelihood | 0.9M | 108.0 / 8.97 / 52.2 | 132.6 / 10.57 / 78.5 | 128.3 / 7.58 / 82.8 | 46.4 / 2.10 / 26.1 | (other scale) |
| RSGM MLP large | 12.2M | 98.3 / 8.04 / 49.5 | 122.4 / 9.19 / 74.5 | 130.3 / 7.45 / 83.8 | 46.1 / 1.80 / 27.3 | 5.861 |
| RSGM timeres 512 x 2 | 1.9M | 97.9 / 7.96 / 49.9 | 123.2 / 8.98 / 77.0 | 121.2 / 5.78 / 84.7 | 44.5 / 1.83 / 26.0 | 5.936 |
| RSGM timeres 1024 x 2 | 5.8M | 98.7 / 7.65 / 50.6 | 121.3 / 8.71 / 76.6 | 121.1 / 6.26 / 80.9 | 44.2 / 1.76 / 26.1 | **5.854** |

Hidden-joint error along the 200 steps (deg; step 100 -> 199):

| denoiser | left_leg | arms |
|---|---|---|
| DPoser | 10.8 -> 11.5 | 21.7 -> 22.1 |
| RSGM MLP small | 13.8 -> 17.0 | 24.1 -> 29.1 |
| RSGM MLP large | 13.1 -> 13.8 | 23.8 -> 26.6 |
| RSGM timeres 512 x 2 | 13.4 -> 13.9 | 23.1 -> 24.1 |
| RSGM timeres 1024 x 2 | 13.4 -> 13.7 | 23.3 -> 24.6 |

- **The time in every layer works**: 1.9M parameters match the 12.2M MLP (better on the arms by 9 mm), the drift at
  the smallest noise is gone (as DPoser), the spread of the hypotheses is DPoser's.
- **Size does not help further**: 5.8M is the same as 1.9M in completion, but it has the best validation loss of all
  RSGMs (the density does not suffer from the new network; the small one is slightly worse).
- **The slow first phase stays** (about 2.6 deg on the left leg at step 100): the rest of the gap to DPoser
  (18-20 mm on the legs, 10 mm on the arms, 1 mm on the trunk). Neither the time conditioning nor the size moves it.
  Candidates: the hybrid is DPoser's own optimization in its own coordinates (its schedule, lr and weights are tuned
  for DPoser, and the start of the hidden joints is a typical input of DPoser at large t but an unusual one for RSGM
  at small t); the input (rotation matrices against z-scored axis-angle).
