# Score and flow networks of the pose priors, and a DPoser-like score network for RSGM

Why this note: in pose completion RSGM loses to DPoser with DPoser's own optimization (score distillation at small
noise), also when the optimization is copied one to one and only the denoiser is swapped (the "hybrid" diagnostic,
see below). The heat kernel, the noise geometry and the noise of a single step are ruled out; what remains is how the
network behaves off the data at small noise, which training does not pin down and the architecture does. This note
puts the three networks side by side and proposes the one to try.

## Summary

Updated 2026-11-16 (sections 6-12). Validation numbers only; no test was run for this line of work.

- **Skip connections or LayerNorm in the score network remove the drift of SDS at the smallest noise** and improve the
  hybrid (DPoser's optimization with the RSGM denoiser) by 15-17 mm on the legs and 10-14 mm on the arms over a plain
  MLP of the same size (section 6). Which of the two is needed, and whether they add up, one training seed does not
  resolve: both together were not better than either alone.
- **The time in every layer, a sinusoidal time embedding, log s as the time input and a zero-initialized last layer
  have no measurable effect** (within about ±5 mm, the size of the differences between nearly identical runs). The
  claim of section 5 that the time conditioning is what helps is wrong: timeres helped through its residual blocks and
  LayerNorm.
- **The slow first phase of SDS does not move** with any network: 13.2-14.3 deg on the left leg at step 100 for every
  RSGM with the isotropic noise, 10.8 deg for DPoser. It moves only with the noise (section 10).
- **DPoser's size and training budget do not close the gap** (section 7): a 7.3M network trained 200k steps x 5120 on
  all frames is in the hybrid where the 1.2M networks with residual or LayerNorm are (94.8 / 124.0 / 125.5 / 42.6 mm
  against DPoser's 79.4 / 103.3 / 110.6 / 43.5), and in D-Flow it is not better than the 512x4 MLP (94.35 / 52.46 mm
  against 93.33 / 52.43, section 8).
- **The input representation does not move it either** (section 9): z-scored tangent coordinates around each joint's
  mean next to the matrices (as DPoser's input) leave the first phase at 13.2 and 14.2 deg; the hybrid on the left leg
  is 106.0 / 108.0 mm, above the networks with residual or LayerNorm. With DPoser's network and budget
  (`dposer_like_z`) it changes nothing either, and it breaks the exact likelihood of 4% of the poses.
- **Per-joint noise is what was missing** (section 10): with the Brownian time c_j s(t), c_j proportional to the
  squared spread of the joint (every joint at the same signal-to-noise ratio, as DPoser's noise in z-scored
  coordinates), the first phase moves from 13.2-14.3 to 12.1-12.3 deg (seed 10 and 11 agree), and it adds up with
  residual or LayerNorm. `abl_norm_aniso` (1.2M, 30k steps, c10) reaches 87.0 / 105.2 / 108.0 / 44.0 mm in the
  hybrid against DPoser's 79.4 / 103.3 / 110.6 / 43.5: better on the arms, 1.9 mm behind on the legs, 7.6 mm on the
  left leg. The APD is lower than without it (5.8-6.8 against 7.3-9.5 cm). It does not carry over to the completion
  protocols: in D-Flow the mean drops by 1-4 mm with less diversity (APD 5.5-5.8 against 6.4-6.5 cm, best of 10 1-2
  mm worse), and in inpainting at equal APD it is worse (2-3 mm on c10, 0.2-1.1 mm on full data; the completion
  report, section 14).
- Decision: per-joint noise helps score distillation at small noise (the first phase), not the whole reverse
  process; the best completion of RSGM stays inpainting with isotropic networks.

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

- **timeres works** (the attribution to the time in every layer is wrong, see section 6: it is the residual blocks or
  LayerNorm): 1.9M parameters match the 12.2M MLP (better on the arms by 9 mm), the drift at the smallest noise is
  gone (as DPoser), the spread of the hypotheses is DPoser's.
- **Size does not help further**: 5.8M is the same as 1.9M in completion, but it has the best validation loss of all
  RSGMs (the density does not suffer from the new network; the small one is slightly worse).
- **The slow first phase stays** (about 2.6 deg on the left leg at step 100): the rest of the gap to DPoser
  (18-20 mm on the legs, 10 mm on the arms, 1 mm on the trunk). Neither the time conditioning nor the size moves it.
  Candidates: the hybrid is DPoser's own optimization in its own coordinates (its schedule, lr and weights are tuned
  for DPoser, and the start of the hidden joints is a typical input of DPoser at large t but an unusual one for RSGM
  at small t); the input (rotation matrices against z-scored axis-angle).

## 6. 2026-11-15: Ablation of the network

Which component of timeres helps? Every component can be switched off (`hp.models.score_nets.TimeResNet`, RSGM's
`net_*` options). `abl_base` has all of them off: a plain MLP of 1 + 2 x 2 layers of 512 with t as one number at the
input, lr 2e-3 as the MLPs (about 1.2M parameters). Every `abl_<component>` turns on exactly one component of
timeres_small; `res_norm` turns on two. Configs in `scripts/train_rsgm_posedflow.py` (`ABL_BASE`, `ABLATION`,
`deep`, `deep_nores`, `res_norm`); all c10, weighting std, 30k steps x 4096, seed 10, trained on workstation1
(about 17 min each on an RTX 4090).

Hybrid on 500 validation poses (MPVPE mean, mm), validation loss of the run, and the hidden-joint error of the
trajectory at steps 100 and 199 (deg, 300 validation poses):

| run | left_leg | legs | arms | trunk | valid loss best / last | left_leg 100 -> 199 | arms 100 -> 199 |
|---|---|---|---|---|---|---|---|
| DPoser (control) | **79.4** | **103.3** | **110.6** | **43.5** | – | 10.8 -> 11.5 | 21.7 -> 22.1 |
| abl_base | 112.2 | 138.7 | 133.6 | 48.2 | 5.884 / 5.885 | 13.6 -> 15.8 | 24.1 -> 28.2 |
| + time in every layer | 117.7 | 136.2 | 132.9 | 47.2 | 5.894 / 5.905 | 14.3 -> 16.6 | 24.0 -> 27.9 |
| + sinusoidal embedding | 114.6 | 140.0 | 136.0 | 47.0 | 5.902 / 5.905 | 13.6 -> 16.0 | 24.2 -> 28.9 |
| + log s instead of t | 117.3 | 143.8 | 134.2 | 47.7 | 5.890 / 5.897 | 13.9 -> 16.4 | 24.1 -> 28.4 |
| + zero init | 114.0 | 139.6 | 134.9 | 47.7 | 5.884 / 5.888 | 13.8 -> 15.7 | 24.5 -> 29.1 |
| + lr 1e-3 | 129.7 | 151.0 | 141.2 | 51.6 | 5.927 / 5.927 | 13.8 -> 18.2 | 24.0 -> 29.1 |
| + dropout 0.1 | 108.9 | 125.4 | 127.5 | 48.5 | 6.027 / 6.027 | 13.7 -> 14.8 | 23.3 -> 24.8 |
| + residual | 96.6 | 121.7 | 123.3 | 45.4 | 5.906 / 5.913 | 13.2 -> 14.1 | 23.4 -> 24.8 |
| + LayerNorm | 97.4 | 121.4 | 119.5 | 44.4 | 5.903 / 5.903 | 13.2 -> 13.6 | 23.4 -> 24.5 |
| + residual + LayerNorm (`res_norm`) | 102.8 | 130.0 | 128.2 | 46.7 | 5.908 / 5.927 | 14.0 -> 14.6 | 24.0 -> 26.2 |
| timeres_small (all on, dropout 0.1, lr 1e-3) | 97.9 | 123.2 | 121.2 | 44.5 | 5.936 | 13.4 -> 13.9 | 23.1 -> 24.1 |
| deep: 256 x 8 blocks, all on | 99.6 | 121.8 | 120.4 | 45.0 | 5.945 / 5.951 | 13.6 -> 14.3 | 23.1 -> 24.3 |
| deep without residual | 248.4 | 261.2 | 192.2 | 93.7 | 20.996 / 20.996 | 27.2 -> 30.6 | 35.5 -> 40.4 |

- Two groups: every network with residual and/or LayerNorm (5 runs) is at 97-103 mm on the left leg and drifts by
  0.4-1 deg in the second phase; every network without them (6 runs) is at 109-130 mm and drifts by 2-2.4 deg. This
  gap of 10-20 mm is larger than any difference inside a group.
- Inside a group the differences are of the size of the noise of one training seed: `abl_base` and `+ zero init`
  are nearly the same network and differ by 1-2 mm; `res_norm` is 5-9 mm worse than each of its components alone,
  which an additive effect cannot explain. The validation loss of `res_norm` rises at the end (5.908 -> 5.927).
- The validation loss does not predict completion: residual and LayerNorm are slightly worse in it than `abl_base`
  (5.90 against 5.88) and much better in completion. Dropout helps completion a little and costs density (6.03).
- Depth (256 x 8) adds nothing to 512 x 2; without skip connections the 17-layer network does not train.
- The first phase (step 100) is 13.2-14.3 deg for every network.

## 7. 2026-11-15: DPoser's size and training budget (`dposer_like`)

Config `dposer_like`: timeres with DPoser's TimeFC size (1024 x 2 blocks, a 512-dim time embedding, dropout 0.1,
7.3M parameters; DPoser 7.2M) and DPoser's training (200k steps x 5120, lr 2e-4, cosine with 5k warm-up, EMA
0.9999) on all frames (`--rotations ~/data/AMASS_rotations_posedflow`), as DPoser's full run; LayerNorm instead of
GroupNorm, rotation matrices and IGSO(3) as every RSGM. Trained on workstation1 (about 4.2 h on an RTX 4090).

| denoiser | left_leg | legs | arms | trunk | valid loss best / last | left_leg 100 -> 199 | arms 100 -> 199 |
|---|---|---|---|---|---|---|---|
| DPoser (control) | **79.4** | **103.3** | **110.6** | 43.5 | – | 10.8 -> 11.5 | 21.7 -> 22.1 |
| RSGM dposer_like | 94.8 | 124.0 | 125.5 | **42.6** | 5.986 / 6.010 | 13.4 -> 13.6 | 24.0 -> 25.5 |

The same hybrid and trajectory as section 6: `dposer_like` is in the group of the networks with residual or
LayerNorm, the first phase is unchanged. Eight times the training samples, ten times the data and DPoser's size do
not close the gap. Its validation loss is worse than the c10 runs', as with dropout in section 6.

## 8. 2026-11-15: D-Flow on validation (partial)

D-Flow (PoseD-Flow's lr 0.1 and running weight 1e-3, Heun on the EDM times with 50 network calls, the best RSGM
setting of [the completion report](2026-11-13_pose_completion_rsgm_vs_dposer.md)) on 1000 validation poses, K = 10,
MPVPE mean (mm) / APD (cm) / best of 10 (mm), `dflow_valid.py` on workstation2 (RTX 6000 Ada):

| model | left_leg | legs | arms | trunk | mean / best of 10 |
|---|---|---|---|---|---|
| RSGM 512x4 MLP, c10, 30k | 95.2 / 7.65 / 46.2 | 113.0 / 8.76 / 67.1 | 117.8 / 7.12 / 69.4 | 47.4 / 2.06 / 27.0 | **93.33** / 52.43 |
| RSGM dposer_like, full, 200k | 94.7 / 7.56 / 46.3 | 116.1 / 9.24 / 67.9 | 119.8 / 7.33 / 69.1 | 46.8 / 2.03 / 26.6 | 94.35 / 52.46 |

In D-Flow, too, DPoser's size and budget give nothing over the 512x4 MLP. The validation numbers are higher than the
test numbers of the completion report (512x4, Heun EDM 50: 88.32 / 49.92 on the test): other poses; compare rows
within a table only. The other rows of the plan (PoseD-Flow on validation, the 512x4 seeds 11 and 12, the networks of
section 6) were not run to the end; the plan and its rule for the test are below.

## 9. 2026-11-16: z-scored tangent input (`abl_z`, `res_norm_z`)

Question: does the score network fail at small noise because it sees raw rotation matrices, in which joints of small
spread (the spine, the shoulders) barely change their 9 numbers? DPoser sees z-scored axis-angle.

`inputs="local+z"` (`hp/models/rsgm.py`) gives the network, next to the 9J matrix entries, z_j = log(M_j^T x_j) /
sigma_j (3J numbers): M_j is the Karcher mean of joint j over the 1,861,105 training poses of c10, sigma_j the std of
each coordinate of the log (`z_statistics`, kept as buffers and model arguments). The matrices stay, so the input is
continuous where the log jumps (angle pi from M_j, at large t). The spread ranges from 0.04-0.08 rad (Spine2) to 0.5
rad (elbows, knees), so z magnifies the spine about 20 times relative to the knees. Diffusion, score parametrization,
loss and weighting are unchanged. Configs `abl_z` = `abl_base` + z and `res_norm_z` = `res_norm` + z: each pair
differs by the input only (30k x 4096, seed 10, c10, std weighting; workstation1, RTX 4090, 21 min each; 1.20M
parameters, 29k more than without z).

The rule, written before the runs: the input helps if the left-leg error at step 100 moves from about 13.4 deg
clearly toward DPoser's 10.8 deg and the hybrid on the left leg drops clearly below the 97-103 mm of the networks with
residual or LayerNorm.

Hybrid on 500 validation poses, K = 10, MPVPE mean (mm) / APD (cm) / best of 10 (mm); validation loss of the run;
hidden-joint error of the trajectory at steps 100 and 199 (deg, 300 validation poses); `eval_runs.sh` on the Mac:

| run | left_leg | legs | arms | trunk | valid loss best / last | left_leg 100 -> 199 | arms 100 -> 199 |
|---|---|---|---|---|---|---|---|
| DPoser (control) | **79.4** | **103.3** | **110.6** | 43.5 | – | **10.8 -> 11.5** | **21.7 -> 22.1** |
| abl_base | 112.2 / 9.52 / 54.8 | 138.7 / 11.52 / 80.0 | 133.6 / 8.09 / 84.2 | 48.2 / 2.22 / 26.1 | 5.884 / 5.885 | 13.6 -> 15.8 | 24.1 -> 28.2 |
| abl_z | 106.0 / 8.83 / 51.7 | 127.3 / 10.20 / 76.1 | 130.0 / 7.29 / 85.1 | 46.0 / 2.13 / 25.3 | 5.903 / 5.903 | 13.2 -> 14.7 | 23.8 -> 27.0 |
| res_norm | 102.8 / 7.48 / 54.1 | 130.0 / 9.26 / 80.4 | 128.2 / 6.65 / 86.5 | 46.7 / 1.63 / 28.8 | 5.908 / 5.927 | 14.0 -> 14.6 | 24.0 -> 26.2 |
| res_norm_z | 108.0 / 7.48 / 57.3 | 127.5 / 8.91 / 79.6 | 126.8 / 6.57 / 83.4 | 47.1 / 1.66 / 29.1 | 6.043 / 6.043 | 14.2 -> 15.2 | 23.8 -> 25.6 |
| RSGM dposer_like (full, 200k) | 94.8 / 7.30 / 49.3 | 124.0 / 8.75 / 78.3 | 125.5 / 6.74 / 83.0 | **42.6** / 1.69 / 25.3 | 5.986 / 6.010 | 13.4 -> 13.6 | 24.0 -> 25.5 |

Neither part of the rule holds: the first phase stays in the band of every other network (13.2-14.3 deg), and the
hybrid on the left leg is 106.0 and 108.0 mm. Against `abl_base` the input gains 2-11 mm on every occlusion and less
drift; against `res_norm` the sign changes from occlusion to occlusion (left leg 5 mm worse, legs and arms 1-3 mm
better), within the spread of one seed seen in section 6. The validation loss of `res_norm_z` is worse (6.043 against
5.908). The representation of the input is not the cause of the slow first phase.

Added the same day: `dposer_like_z` (`dposer_like` + z, full data, 200k x 5120, 2.97 h on workstation1) against
`dposer_like`, same hybrid and trajectory:

| run | left_leg | legs | arms | trunk | valid loss best / last | left_leg 100 -> 199 | arms 100 -> 199 |
|---|---|---|---|---|---|---|---|
| RSGM dposer_like (full, 200k) | 94.8 / 7.30 / 49.3 | 124.0 / 8.75 / 78.3 | 125.5 / 6.74 / 83.0 | 42.6 / 1.69 / 25.3 | 5.986 / 6.010 | 13.4 -> 13.6 | 24.0 -> 25.5 |
| dposer_like_z | 94.2 / 7.23 / 49.5 | 122.3 / 8.98 / 75.0 | 124.5 / 6.50 / 82.7 | 42.7 / 1.71 / 25.1 | 6.065 / 6.066 | 13.2 -> 13.3 | 24.1 -> 25.6 |

The rule fixed before it (left leg at step 100 at most 12.5 deg and the hybrid on the left leg at most 90 mm) does
not hold: the two differ by 0-2 mm and 0.2 deg.

The z input breaks the exact likelihood. Validation log-likelihood of the 500 poses of the quick check (EMA weights,
nats relative to uniform on SO(3)^19, `loglik_valid.py` on the Mac):

| run | mean +- s.e.m. | median | min / 1% / 5% | poses below 0 |
|---|---|---|---|---|
| abl_base | 135.88 +- 0.81 | 136.6 | 72.8 / 97.2 / 106.5 | 0 |
| abl_z | 130.35 +- 5.34 | 138.4 | -2019 / -14.0 / 102.5 | 6 |
| dposer_like | 136.58 +- 0.78 | 137.1 | 80.7 / 98.6 / 107.4 | 0 |
| dposer_like_z | -1.13 +- 59.46 | 140.2 | -20595 / -2163 / 89.5 | 19 |

The median is as good or better, but a few percent of the poses get thousands of nats wrong (and `abl_z` a maximum of
766): the log jumps at angle pi from M_j, the probability flow passes there at large t, and the exact divergence there
is meaningless. A smooth feature (vee(M_j^T x_j - x_j^T M_j) / 2 = sin(angle) axis) would avoid it; not done, since the
input does not help completion.

## 10. 2026-11-16: Per-joint noise (`abl_aniso`, `abl_residual_aniso`, `abl_norm_aniso`)

Question: is the slow first phase the isotropic noise? DPoser adds its noise to z-scored coordinates, so in radians
the noise of a joint is proportional to its spread, and at every noise level every joint has the same
signal-to-noise ratio. RSGM's IGSO(3) noise has one angle for all joints: at a t where the knees (spread about 0.5 rad)
are still clear, the spine and the shoulders (0.04-0.2 rad) are drowned.

`joint_scale` = c (`hp/models/rsgm.py`): joint j diffuses with the Brownian time s_j(t) = min(c_j s(t), 20), beta_j =
d s_j / dt, so x_t | x_0 is still IGSO(3) per joint and the target of the loss stays exact; the std trick, the
weighting std_j(t)^2, the samplers and the likelihood work joint by joint, and the network still gets one t.
c_j = sigma_j^2 / min_k sigma_k^2 with sigma_j the RMS spread of joint j around its Karcher mean (`fit_joint_scale` of
the z statistics of section 9): 1 (Spine2) to 42.8 (right elbow), about 35 for the knees. The cap at 20 (uniform to
exp(-20)) keeps the IGSO(3) numerics finite, which fail above about 30 (c_j s(1) reaches 160). In the hybrid the RSGM
time is chosen with s(t_r) = mean_j s_j^DPoser / c_j and the noise of joint j is IGSO(3) of c_j s(t_r): close to
DPoser's own noise of every joint; for c = 1 it is the old matching (`abl_base` reproduces its numbers exactly).

Configs `abl_aniso` = `abl_base` + c, `abl_residual_aniso` and `abl_norm_aniso` = `abl_residual` and `abl_norm` + c;
each pair differs by the noise only (30k x 4096, c10, std weighting). `abl_aniso` seed 10 was trained with the CPU
noise of the loss (workstation1, 21 min); seed 11 and the other two with the noise drawn on the GPU (branch
`experimental/igso_gpu`, 11-14 min each on a loaded workstation1; section 11).

The rule, written before the runs (as section 9): the left-leg error at step 100 moves from about 13.4 deg clearly
toward 10.8 deg, and the hybrid on the left leg drops clearly below the 97-103 mm of the networks with residual or
LayerNorm.

Hybrid on 500 validation poses, K = 10, MPVPE mean (mm) / APD (cm) / best of 10 (mm); validation loss of the run;
hidden-joint error of the trajectory at steps 100 and 199 (deg, 300 validation poses); `eval_runs.sh` on the Mac:

| run | left_leg | legs | arms | trunk | valid loss best / last | left_leg 100 -> 199 | arms 100 -> 199 |
|---|---|---|---|---|---|---|---|
| DPoser (control) | **79.4** | **103.3** | 110.6 | 43.5 | – | **10.8 -> 11.5** | **21.7 -> 22.1** |
| abl_base | 112.2 / 9.52 / 54.8 | 138.7 / 11.52 / 80.0 | 133.6 / 8.09 / 84.2 | 48.2 / 2.22 / 26.1 | 5.884 / 5.885 | 13.6 -> 15.8 | 24.1 -> 28.2 |
| abl_residual | 96.6 / 7.27 / 50.9 | 121.7 / 8.72 / 76.6 | 123.3 / 6.37 / 83.0 | 45.4 / 1.86 / 26.2 | 5.906 / 5.913 | 13.2 -> 14.1 | 23.4 -> 24.8 |
| abl_norm | 97.4 / 7.89 / 47.8 | 121.4 / 9.10 / 74.6 | 119.5 / 6.04 / 81.8 | 44.4 / 1.79 / 26.0 | 5.903 / 5.903 | 13.2 -> 13.6 | 23.4 -> 24.5 |
| RSGM dposer_like (full, 200k) | 94.8 / 7.30 / 49.3 | 124.0 / 8.75 / 78.3 | 125.5 / 6.74 / 83.0 | **42.6** / 1.69 / 25.3 | 5.986 / 6.010 | 13.4 -> 13.6 | 24.0 -> 25.5 |
| abl_aniso, seed 10 | 93.4 / 6.68 / 51.4 | 116.3 / 8.02 / 74.4 | 121.6 / 6.12 / 82.2 | 47.8 / 2.06 / 26.5 | 1.539 / 1.539 | 12.3 -> 13.9 | 23.1 -> 25.5 |
| abl_aniso, seed 11 | 93.0 / 6.83 / 49.6 | 114.1 / 7.92 / 71.6 | 121.7 / 5.92 / 82.1 | 47.3 / 1.97 / 26.8 | 1.549 / 1.549 | 12.2 -> 13.5 | 23.1 -> 25.4 |
| abl_residual_aniso | 89.6 / 6.03 / 50.4 | 107.6 / 6.66 / 71.3 | 115.7 / 4.96 / 82.6 | 46.6 / 1.74 / 28.7 | 1.550 / 1.551 | 12.3 -> 12.8 | 22.7 -> 23.1 |
| abl_norm_aniso | 87.0 / 5.81 / 49.3 | 105.2 / 6.40 / 70.6 | **108.0** / 4.87 / 75.0 | 44.0 / 1.61 / 26.5 | 1.556 / 1.556 | 12.1 -> 12.3 | 22.4 -> 22.6 |

Both parts of the rule hold. The first phase leaves the band of all isotropic networks (13.2-14.3 deg) for the first
time: 12.1-12.3 deg, about half of the way to DPoser, and the two seeds of `abl_aniso` agree within 0.1 deg and
0.4-2 mm. The noise adds up with residual and LayerNorm (7-16 mm on the left leg, the legs and the arms in each pair);
with LayerNorm the drift of the second phase nearly vanishes (12.1 -> 12.3 deg) and the hybrid is within 2 mm of
DPoser on the legs and better on the arms. The APD drops with the noise (left leg 5.8-6.8 against 7.3-9.5 cm): part of
the gain in the mean may be less diversity. The validation losses with c are not comparable to those without (the
weighting std_j^2 differs per joint).

Added the same day: `dposer_like_aniso` (`dposer_like` + c, full data, 200k x 5120; c from the z statistics of the
full data), trained twice on workstation1: with the DataLoader and with `train.data_on_device` (section 12; the same
seed, other order of the poses, so in effect two seeds). Hybrid and trajectory as above:

| run | left_leg | legs | arms | trunk | valid loss best / last | left_leg 100 -> 199 | arms 100 -> 199 |
|---|---|---|---|---|---|---|---|
| RSGM dposer_like (full, 200k) | 94.8 / 7.30 / 49.3 | 124.0 / 8.75 / 78.3 | 125.5 / 6.74 / 83.0 | 42.6 / 1.69 / 25.3 | 5.986 / 6.010 | 13.4 -> 13.6 | 24.0 -> 25.5 |
| dposer_like_aniso, DataLoader | 97.3 / 6.45 / 54.4 | 121.0 / 7.48 / 81.6 | 109.6 / 5.22 / 74.5 | 43.9 / 1.59 / 27.3 | 1.613 / 1.616 | 11.9 -> 13.1 | 22.4 -> 22.9 |
| dposer_like_aniso, data on device | 97.4 / 6.28 / 55.5 | 122.9 / 7.57 / 81.7 | 115.0 / 5.07 / 80.5 | 44.8 / 1.57 / 27.9 | 1.610 / 1.612 | 12.0 -> 13.2 | 22.6 -> 23.4 |

The two runs agree within 2 mm except the arms (5.4 mm). Against `dposer_like` the first phase moves the most of all
networks (11.9 deg, 1.1 deg from DPoser) and the arms gain 10-16 mm, but the left leg and the trunk are 1-3 mm worse
and the drift of the second phase is back (+1.2 deg): DPoser's size, data and budget do not beat `abl_norm_aniso`.

D-Flow (Heun EDM 50, `dflow_valid.py` on workstation1) on the 1000 validation poses of section 8, MPVPE mean (mm) /
APD (cm) / best of 10 (mm):

| model | left_leg | legs | arms | trunk | mean / best of 10 | APD |
|---|---|---|---|---|---|---|
| RSGM 512x4 MLP, c10 (isotropic) | 95.2 / 7.65 / 46.2 | 113.0 / 8.76 / 67.1 | 117.8 / 7.12 / 69.4 | 47.4 / 2.06 / 27.0 | 93.33 / 52.43 | 6.40 |
| RSGM dposer_like, full (isotropic) | 94.7 / 7.56 / 46.3 | 116.1 / 9.24 / 67.9 | 119.8 / 7.33 / 69.1 | 46.8 / 2.03 / 26.6 | 94.35 / 52.46 | 6.54 |
| abl_norm_aniso | 96.0 / 6.98 / 51.7 | 110.6 / 7.64 / 69.9 | 112.5 / 6.46 / 68.5 | 49.4 / 2.15 / 27.6 | 92.12 / 54.42 | 5.81 |
| dposer_like_aniso, data on device | 91.7 / 6.53 / 49.4 | 110.2 / 7.55 / 68.6 | 110.8 / 5.95 / 68.3 | 49.3 / 2.02 / 27.9 | 90.49 / 53.54 | 5.51 |

The mean drops by 1-4 mm, but the APD drops too (5.5-5.8 against 6.4-6.5 cm) and the best of 10 is 1-2 mm worse:
less diversity, which D-Flow cannot trade back, so no gain at equal APD can be read from it.

Inpainting with per-joint noise: worse than isotropic, 2-3 mm on c10 (norm), 0.2-1.1 mm on full (dposer_like); see
[the completion report](2026-11-13_pose_completion_rsgm_vs_dposer.md), section 14.

## 11. 2026-11-16: The IGSO(3) noise on the GPU

Not a change of method: where the noise of RSGM is drawn (branch `experimental/igso_gpu`, commits 4e47e47 and 90c159a).
The training loss drew its times, IGSO(3) angles and axes on the CPU in float64 and noised the poses there: every step
copied the batch to the CPU and back, and several runs on one machine fought for its cores (about 50 each, load
average about 160 on workstation1). Now the loss draws them on the device where it has float64 (cuda), still in
float64; on mps (no float64) and for the validation loss (fixed CPU generator, the same number as before) the CPU path
stays, so validation losses compare across the change. Inpainting draws its noise on the model's device in its dtype
(`AngleSampler.sample_at`: the level of s is found on the CPU, so the GPU does not wait). SDS
(`score_completion.py`) still draws on the CPU; D-Flow and the unconditional sampler draw no IGSO(3) noise.
`RSGM.noise_on_cpu = True` restores the CPU path, for `scripts/benchmark_igso_noise.py`.

- Speed of training, workstation1 under load: `abl_aniso` seed 11 ran at about 50 it/s (11 min for 30k steps) on the
  GPU path against 2-3 it/s on the CPU path at the same time; on an idle machine the CPU path took 21 min (seed 10).
- The same training: validation loss of `abl_aniso` seed 11 at 1k-6k steps 2.03 / 1.81 / 1.72 / 1.68 / 1.66 / 1.64
  on the GPU path against 2.03 / 1.83 / 1.74 / 1.69 / 1.67 / 1.64 on the CPU path.
- The same inpainting: 300 validation poses, K = 10, small std, steps 100, resample 2, guidance 0.1, 8 seeds per code
  version on the Mac (mps): mean over the four occlusions 99.55 +- 0.40 against 99.49 +- 0.49 mm (MPVPE mean),
  7.69 +- 0.07 against 7.72 +- 0.06 cm (APD), 52.46 +- 0.40 against 52.59 +- 0.59 mm (best of 10); none of the 15
  differences (4 occlusions and the mean, 3 metrics) is significant (Welch p >= 0.08).
- The float32 sampler (mps, same uniforms as float64 on the CPU, 400k draws): relative error of the angle 5e-8 on
  average, at most 1.1e-4 (only in the far tail); KS distance to IGSO(3) 0.0009 (float32) and 0.0013 (float64),
  1% threshold 0.0026.

`scripts/benchmark_igso_noise.py` (untrained MLP 512 x 4, 19 joints, random poses), on an idle RTX 4090 of bigmama
(torch 2.14.1, 16 CPU threads) and on the Mac (mps; there the loss draws on the CPU either way):

| measurement | cuda: noise on CPU | cuda: on device | speed-up | mps: noise on CPU | mps: on device | speed-up |
|---|---|---|---|---|---|---|
| train step, batch 4096 | 82.2 it/s | 519.8 it/s | 6.32x | 53.1 it/s | 51.9 it/s | 0.98x |
| train step, batch 16384 | 18.8 it/s | 183.2 it/s | 9.74x | 18.9 it/s | 18.7 it/s | 0.99x |
| inpainting, 5000 poses, guidance 0, resample 3 | 7.91 ms/call | 1.29 ms/call | 6.11x | 12.99 ms/call | 5.67 ms/call | 2.29x |
| inpainting, 5000 poses, guidance 0.1, resample 1 | 12.62 ms/call | 9.25 ms/call | 1.36x | 22.63 ms/call | 18.49 ms/call | 1.22x |
| inpainting, 20000 poses, guidance 0, resample 3 | 25.56 ms/call | 3.38 ms/call | 7.56x | 46.97 ms/call | 23.84 ms/call | 1.97x |
| inpainting, 20000 poses, guidance 0.1, resample 1 | 42.64 ms/call | 27.04 ms/call | 1.58x | 86.42 ms/call | 74.71 ms/call | 1.16x |

On cuda the noise was most of the cost: 6-10x for a training step and 6-8x for inpainting without guidance. With
guidance the backward pass through the network dominates (1.4-1.6x). A training step alone runs at 520 it/s, while
the training script reached about 50 it/s: what is left in training is outside the loss (the DataLoader collates
4096 poses one by one on the CPU, and the machine was loaded).

Added the same day: the three aniso runs of section 10 trained with the same configs and seeds on both paths
(workstation1; CPU path in `~/data/experiments/cpu_noise/`, GPU path in `igso_speed/`), evaluated as in section 10
(hybrid on 500 validation poses, MPVPE mean / APD / best of 10; trajectory on 300; `eval_runs.sh` on the Mac). The
same seed is not the same random stream on the two paths (CPU and CUDA generators), so a pair differs by the seed as
well. The rule written before: the paths are equivalent if a pair differs by at most 1-2 mm in the hybrid and about
0.1 deg in the first phase.

| run | noise | left_leg | legs | arms | trunk | valid loss | left_leg 100 -> 199 | arms 100 -> 199 |
|---|---|---|---|---|---|---|---|---|
| abl_aniso s11 | CPU | 99.6 / 7.11 / 52.9 | 120.1 / 8.24 / 75.8 | 115.7 / 6.13 / 76.3 | 47.1 / 2.05 / 26.0 | 1.547 | 12.6 -> 14.5 | 23.1 -> 25.7 |
| abl_aniso s11 | GPU | 93.0 / 6.83 / 49.6 | 114.1 / 7.92 / 71.6 | 121.7 / 5.92 / 82.1 | 47.3 / 1.97 / 26.8 | 1.549 | 12.2 -> 13.5 | 23.1 -> 25.4 |
| abl_residual_aniso | CPU | 89.1 / 5.95 / 50.2 | 109.5 / 6.91 / 72.7 | 113.3 / 4.81 / 81.4 | 46.1 / 1.83 / 27.4 | 1.541 | 12.2 -> 12.7 | 22.6 -> 23.1 |
| abl_residual_aniso | GPU | 89.6 / 6.03 / 50.4 | 107.6 / 6.66 / 71.3 | 115.7 / 4.96 / 82.6 | 46.6 / 1.74 / 28.7 | 1.551 | 12.3 -> 12.8 | 22.7 -> 23.1 |
| abl_norm_aniso | CPU | 86.3 / 5.58 / 50.3 | 103.5 / 6.31 / 69.0 | 107.9 / 4.84 / 75.3 | 43.4 / 1.68 / 25.8 | 1.552 | 12.1 -> 12.3 | 22.1 -> 22.4 |
| abl_norm_aniso | GPU | 87.0 / 5.81 / 49.3 | 105.2 / 6.40 / 70.6 | 108.0 / 4.87 / 75.0 | 44.0 / 1.61 / 26.5 | 1.556 | 12.1 -> 12.3 | 22.4 -> 22.6 |

With residual or LayerNorm the paths agree (0.1-2.4 mm, at most 0.1 deg). The plain MLP `abl_aniso` differs by up to
6.6 mm and 0.5 deg, with signs that change between the occlusions (the legs better on the GPU path, the arms on the
CPU path): the spread of one seed of a network whose second phase drifts, not a shift of the path. So the agreement of
`abl_aniso` seeds 10 and 11 in section 10 (0.4-2 mm) was partly luck; its spread between runs is up to about 7 mm.
The validation losses of the GPU path are 0.002-0.01 higher in all three pairs, too few pairs to tell.

## 12. 2026-11-16: Training throughput (data on the device, TF32, batch size)

Section 11 left the DataLoader as the suspect: it collates a batch pose by pose on the CPU and copies it. Two options
of the trainer (`hp.training.trainer`, branch `experimental/aniso-models`): `train.data_on_device` keeps the training
poses on the device and slices batches of a permutation drawn there every pass (the same sampling: every pose once
per pass); `train.tf32` allows TF32 matrix products on cuda. The EMA update became one foreach call.
`scripts/benchmark_training_throughput.py`: 20 warm-up and 200 timed training steps on the c10 training poses, per
network, batch size, loader and TF32; workstation1, one RTX 4090 shared with other users' jobs, so single rows can be
off (e.g. dposer_like, DataLoader, TF32, batch 20480). Thousands of poses per second (ms per step in brackets):

| network | loader | TF32 | 5120 | 10240 | 20480 | 40960 | 81920 |
|---|---|---|---|---|---|---|---|
| abl_norm_aniso (1.2M) | DataLoader | off | 374 (13.7) | 376 (27.2) | 357 (57.4) | 340 (120.5) | 309 (265.0) |
| abl_norm_aniso | DataLoader | on | 390 (13.1) | 387 (26.5) | 354 (57.8) | 344 (119.2) | 308 (265.8) |
| abl_norm_aniso | device | off | 1740 (2.9) | 2125 (4.8) | 2178 (9.4) | 2176 (18.8) | 2089 (39.2) |
| abl_norm_aniso | device | on | 1681 (3.1) | 2476 (4.1) | **2736** (7.5) | 2533 (16.2) | 2413 (34.0) |
| dposer_like_aniso (7.3M) | DataLoader | off | 331 (15.5) | 318 (32.2) | 298 (68.8) | 290 (141.4) | 268 (306.2) |
| dposer_like_aniso | DataLoader | on | 325 (15.8) | 298 (34.4) | 242 (84.7) | 296 (138.5) | 272 (300.7) |
| dposer_like_aniso | device | off | 588 (8.7) | 613 (16.7) | 613 (33.4) | 628 (65.2) | 632 (129.5) |
| dposer_like_aniso | device | on | 773 (6.6) | 783 (13.1) | 790 (25.9) | **794** (51.6) | 791 (103.5) |

- The DataLoader was the bottleneck: it caps every network at 300-390 thousand poses per second, whatever the batch
  and TF32. With the data on the device a step at batch 5120 is 4.7x faster for the 1.2M network (2.9 against 13.7 ms)
  and 1.8x for the 7.3M one (8.7 against 15.5 ms).
- A larger batch helps only the small network (up to 2.2-2.7 million poses per second from 10240-20480 on); the 7.3M
  network is saturated at 5120 already (588-632 and 773-794 thousand). The batch of the pairs (4096 and 5120) can stay.
- TF32 gives 1.3x for the 7.3M network, little for the 1.2M one. It changes the precision of the network's products,
  so runs that are paired with float32 runs keep it off.
- Memory: the c10 poses on the device take about 1.3 GB, all frames (18.6M poses) about 13 GB in float32.

## 13. 2026-11-17: Plan of a full factorial study of five components (written before any run)

Question: which of the components of sections 6-10 matter, alone and together, for completion and for generation,
and do the noise (first phase) and the stabilization of the network (drift) add up or interact? So far each component
was changed alone on `abl_base`, one seed each; `res_norm` being worse than residual and LayerNorm alone was left
open, and per-joint noise helped the hybrid but not inpainting.

Design: a full 2^5 factorial, one run per cell (32 runs, seed 10). With one run per cell every main effect, every
2-way and every 3-way interaction is estimated without aliasing, each as a difference of two means of 16 runs (its
standard error 2 sigma / sqrt(32), about 1 mm for sigma = 3 mm per run); the six 4- and 5-way interactions estimate
the noise, next to the spread of seeds known from sections 10-11 (about 2 mm with LayerNorm or residual, up to 7 mm
for the plain MLP). A half fraction with two seeds would cost the same and alias the pairs with the triples. Two cells
(the base and the predicted best) get a second seed as a direct check of the noise.

| factor | low (-1) | high (+1) |
|---|---|---|
| R: residual blocks | off | on |
| N: LayerNorm | off | on |
| A: noise | isotropic | per joint (joint_scale, section 10) |
| D: dropout | 0 | 0.1 |
| W: width | 512 | 1024 |

Everything else is `abl_base` (timeres with 2 blocks, the time as a scalar at the input, no zero init, lr 2e-3, 30k x
4096, c10, std weighting), trained with the GPU noise and `train.data_on_device` (sections 11-12). Left out: the time
conditioning and zero init (no effect alone, section 6), the z input (section 9), the size of DPoser and full data
(section 7, later for the chosen cell).

Responses of every run, on validation only:
- completion: inpainting at one fixed point, the rule point of the isotropic `abl_norm` study (200 steps, guidance
  0.095, resample 5, no replacement), 1000 validation poses, K = 10, four occlusions: MPVPE mean, best of 10, APD.
  One point favours networks like `abl_norm`; the effects on the APD at this point show how the trade-off moves.
- generation (the PoseD-Flow protocol on validation instead of test): 20 sets of 500 samples, FID against the joint
  statistics of the validation poses (FID_val) and of the training poses (FID own), APD, dNN; the harness against
  the validation poses: Wasserstein, recall, precision, log-likelihood of 1000 validation poses.
- diagnostics: the hybrid (500 validation poses) and the trajectory (first phase at step 100, drift to step 199).

Rule for the choice, fixed now: among the cells with an inpainting APD of at least 6.40 cm (as the rules of the
completion studies) and an FID_val not worse than that of the cell R-N+A-D-W- (isotropic `abl_norm`) by more than the
half-width of its 95% interval, the cell with the lowest inpainting mean. It and the cell R-N+A-D-W- get a second
seed; the chosen cell is then tuned by an inpainting study (as section 14 of the completion report) and only its
final point goes to the test.

## Conclusions

1. Residual connections or LayerNorm are what a score network on SO(3) needs for score distillation at small noise:
   they stop the drift of the second phase. The time conditioning is not the reason.
2. The slow first phase, and with it 15-20 mm of the gap to DPoser in the hybrid, is not a matter of the network, its
   capacity, the training budget, the amount of data or the representation of its input (section 9: z-scored tangent
   coordinates as DPoser's do not move it). It is the noise: with a Brownian time per joint proportional to its
   squared spread (section 10) the first phase moves halfway to DPoser, and with LayerNorm the hybrid comes within
   2-8 mm of DPoser on the legs and beats it on the arms. The rest of the first phase (about 1.3 deg) may be the
   isotropy within a joint, score against noise prediction, or the hybrid's schedule, which is DPoser's.
3. In D-Flow the network does not matter either (512x4 MLP = dposer_like).
4. Next: completion with the whole reverse process of RSGM (inpainting on SO(3)), where the hidden joints are pulled
   at large noise, which training covers, instead of extrapolating the score at small noise; and per-joint noise in
   every RSGM network: inpainting and D-Flow with `abl_norm_aniso`, and DPoser's size and budget with it.

## What it does not show

- One training seed per network; the noise between seeds is estimated only from near-identical runs (1-2 mm) and
  from `res_norm` (up to 9 mm). A 2^4 factorial study with 3 seeds was planned and dropped as polishing a secondary
  question.
- The hybrid is DPoser's optimization; SDS settings were not tuned for the new networks.
- D-Flow on validation: two rows only. Planned and not finished: `dflow_valid.py 1000 arch --ode heun-edm50` (the
  networks of section 6, the seeds of 512x4 and PoseD-Flow), then the test of the best network by the rule fixed
  before it: the lowest validation mean among the networks not worse than 512x4 in best of 10 by more than 1 mm, and
  only if its gain exceeds the spread of the three 512x4 seeds.
- The likelihood weighting with the new networks.
- The z input (section 9): one seed per run, the z input alone (without the matrices) not tried, and the noise of the
  hybrid stays isotropic in the angle, so the network gets the joints' scale only at the input.
- Per-joint noise (section 10): two seeds for `abl_aniso` only, one for the residual and LayerNorm variants; one scale
  per joint (the noise is isotropic within a joint, IGSO(3) has no anisotropic heat kernel in closed form); only the
  hybrid, not the completion protocols; the mean at equal APD not compared; the noise of the hybrid is matched to the
  model's, so the hybrid of an aniso model also uses DPoser-like noise per joint.

## Where things are

- Runs: `~/data/experiments/rsgm_posedflow_{abl_*,deep,deep_nores,res_norm,res_norm_z,timeres,timeres_small}_s10`,
  `~/data/experiments/full/rsgm_posedflow_dposer_like_z_s10`; the GPU-noise runs of section 10 in
  `~/data/experiments/igso_speed/rsgm_posedflow_{abl_aniso_s11,abl_residual_aniso_s10,abl_norm_aniso_s10}` (their z
  statistics in `~/data/experiments/igso_speed/rsgm_z_statistics.json`),
  `~/data/experiments/full/rsgm_posedflow_dposer_like_s10` (on the Mac and on workstation1).
- z statistics of section 9: `~/data/experiments/rsgm_z_statistics.json` on workstation1 (also in `model.json` of
  each z run).
- `dposer_like_aniso` of section 10: `~/data/experiments/full/rsgm_posedflow_dposer_like_aniso_s10` (DataLoader) and
  `~/data/experiments/full_device/rsgm_posedflow_dposer_like_aniso_s10` (data on device); their D-Flow rows in
  `~/data/experiments/completion/dflow_valid_aniso.jsonl`.
- Throughput rows of section 12: `~/data/experiments/throughput_worstation.jsonl` on workstation1
  (`scripts/benchmark_training_throughput.py`); the CPU-path runs of section 11 in `~/data/experiments/cpu_noise/`.
- The output lines behind the tables of sections 6-10: [2026-11-14_score_network_architectures_eval.txt](2026-11-14_score_network_architectures_eval.txt).
- Scripts (from the root of the repository): `explore/evaluation/completion_diagnostics/eval_runs.sh` (valid loss,
  hybrid, trajectory for a list of runs), `hybrid.py`, `trajectory.py`, `loglik_valid.py`, `dflow_valid.py`; training configs in
  `scripts/train_rsgm_posedflow.py`.
- D-Flow rows: `~/data/experiments/completion/dflow_valid_arch_ws2.jsonl` and the log
  `~/data/experiments/dflow_valid_ws2.log` on workstation2.
