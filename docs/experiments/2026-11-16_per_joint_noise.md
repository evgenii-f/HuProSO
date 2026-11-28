# Per-joint and anisotropic noise of RSGM (validation)

The line of the noise of RSGM's forward process: one IGSO(3) angle for all joints, a Brownian time per joint
(`joint_scale`), and whether a noise per principal axis within a joint is needed. Validation numbers unless a row says
test. The earlier phases are reported in full elsewhere and are summarised here with links: the z input and the
per-joint noise in [the score network report](2026-11-14_score_network_architectures.md) (sections 9 and 10), the
completion fronts in [the completion report](2026-11-13_pose_completion_rsgm_vs_dposer.md) (sections 13 and 15), the
GPU noise in [IGSO(3) noise on the GPU](2026-11-16_igso3_noise_on_gpu.md), motion denoising in
[per-joint noise in motion denoising](2026-11-18_denoising_per_joint_noise.md). The phase of 2026-11-28 (branch
`experimental/fisher-energy`) is reported here in full; it uses the pairwise matrix Fisher energy of
[the ACE line](ace/2026-11-27_fisher_boltzmann.md).

## Summary

- **Why per-joint noise.** In DPoser's SDS optimisation (the hybrid) RSGM had a slow first phase (hidden left leg
  13.2–14.3° at step 100 against DPoser's 10.8°), worth 15–20 mm of the gap to DPoser; the network, its size, the
  budget and the data were ruled out. DPoser adds its noise to z-scored coordinates, so every joint has the same
  signal-to-noise ratio; IGSO(3) with one angle drowns the spine long before the knees.
- **z input: no.** The z-scored tangent coordinates as an input of the network (section 9) did not move the first
  phase and broke the likelihood for a few % of the poses.
- **Brownian time per joint: yes for the hybrid, a trade elsewhere.** With $s_j(t)=\min(c_j s(t),20)$, $c_j$ from
  the squared RMS spread (1–42.8), the first phase reaches 12.1–12.3° and `abl_norm_aniso` comes within 2–8 mm of
  DPoser on the legs and beats it on the arms. But the APD drops with it: in D-Flow the mean is 1–4 mm better and the
  best of 10 1–2 mm worse, and in inpainting the fronts are worse at equal APD (c10 by 0.7–2.9 mm in mean and about
  2 mm in best of 10, full by 0.1–1.1 mm).
- **Motion denoising: no.** Tuned, `abl_residual_aniso` is 0.8 and 1.9 mm worse than isotropic `abl_residual` (40 and
  100 mm noise); the noise of that task is isotropic in positions and hits the short-bone joints, which per-joint
  noise probes least.
- **Anisotropy within a joint (2026-11-28): large, but not the cause.** Principal stds of a joint differ up to 7.9×
  (knees; hips 4.9–5.9×), and with $c_j$ by the RMS spread the narrow axes of the knees and hips are the first of the
  body to be destroyed. In inpainting, however, `abl_norm_aniso` is not worse than `abl_norm` on these axes
  ($D = -0.0332$ rad², rule fixed before the run); its extra error is at the feet and on the wide axis of the hip.
- **Decision.** Per-joint noise stays a feature of completion (the hybrid and D-Flow); an anisotropic kernel on
  SO(3) is not built for the reason that was given for it. What makes per-joint noise lose at equal APD in
  inpainting is open; at the points checked it is less diversity.

## History (2026-11-14 – 2026-11-20)

### Starting point

Score network report, sections 1–8: in the hybrid every isotropic RSGM network has the same slow first phase, the
hidden-joint error at step 100 in a band of 13.2–14.3° (DPoser 10.8°). Residual or LayerNorm remove the drift of the
second phase but not the first phase. Capacity, training budget, the amount of data (c10 against full) and DPoser's
size (`dposer_like`) do not move it.

### z input (2026-11-16, section 9)

`inputs="local+z"`: next to the 9J matrix entries the network gets $z_j=\log(M_j^\top x_j)/\sigma_j$, with $M_j$ the
Karcher mean and $\sigma_j$ the per-coordinate std (`z_statistics`), so the spine is magnified about 20× against the
knees. The noise stays isotropic. The rule (the first phase clearly toward 10.8°, the hybrid on the left leg below
97–103 mm) did not hold, and the likelihood broke for a few % of the poses. Per-coordinate scaling was tried only at
the input; a noise per coordinate was not trained.

### Brownian time per joint (2026-11-16, section 10)

`joint_scale` (`hp/models/rsgm.py`): joint j diffuses with $s_j(t)=\min(c_j s(t),20)$, $\beta_j = ds_j/dt$, so
$x_t\mid x_0$ stays IGSO(3) per joint and the target of the loss stays exact; the network still gets one t.
$c_j=\sigma_j^2/\min_k\sigma_k^2$ with $\sigma_j$ the RMS spread around the Karcher mean (`fit_joint_scale`): 1
(Spine2) to 42.8 (right elbow), about 35 for the knees. In the hybrid the noise of joint j is IGSO(3) of
$c_j s(t_r)$, close to DPoser's own noise of that joint. Pairs that differ by the noise only: `abl_aniso`,
`abl_residual_aniso`, `abl_norm_aniso` (c10, 30k × 4096, seed 10) and `dposer_like_aniso` (full, 200k × 5120).

Hybrid, 500 validation poses, K = 10, MPVPE mean (mm) per occlusion; hidden left leg at steps 100 → 199 (deg):

| run | left_leg | legs | arms | trunk | left_leg APD (cm) | left_leg 100 → 199 |
|---|---|---|---|---|---|---|
| DPoser (control) | **79.4** | **103.3** | 110.6 | 43.5 | – | **10.8 → 11.5** |
| abl_norm | 97.4 | 121.4 | 119.5 | 44.4 | 7.89 | 13.2 → 13.6 |
| abl_norm_aniso | 87.0 | 105.2 | **108.0** | 44.0 | 5.81 | 12.1 → 12.3 |
| dposer_like, full | 94.8 | 124.0 | 125.5 | **42.6** | 7.30 | 13.4 → 13.6 |
| dposer_like_aniso, full (DataLoader run) | 97.3 | 121.0 | 109.6 | 43.9 | 6.45 | 11.9 → 13.1 |

The first phase leaves the band of the isotropic networks for the first time, about half of the way to DPoser; the
APD drops with it.

D-Flow (Heun EDM 50), 1000 validation poses, K = 10; mean over the four occlusions:

| model | mean (mm) | best of 10 (mm) | APD (cm) |
|---|---|---|---|
| RSGM 512×4, c10 (isotropic) | 93.33 | **52.43** | 6.40 |
| RSGM dposer_like, full (isotropic) | 94.35 | 52.46 | **6.54** |
| abl_norm_aniso | 92.12 | 54.42 | 5.81 |
| dposer_like_aniso, full (data on device) | **90.49** | 53.54 | 5.51 |

Lower mean, lower diversity, worse best of 10; D-Flow cannot trade the diversity back, so no gain at equal APD
follows from it.

### Inpainting fronts (Optuna studies, 2026-11-16 – 2026-11-20)

Reverse-process inpainting of RSGM, the studies `rsgm_inpaint_mean_apd_*` (120 trials each, objectives mean and APD,
1000 validation poses, K = 10). The front averaged over bands of APD (mm, lower is better; section 13 of the
completion report for the measure), recomputed for this report with
`explore/evaluation/completion_diagnostics/plot_study_fronts.py --bands`:

| study | mean, APD 6.0–7.0 | mean, APD 6.4–8.5 | mean, APD 7.0–8.5 | best of 10, APD 6.0–7.0 | best of 10, APD 6.4–8.5 |
|---|---|---|---|---|---|
| abl norm (c10) | **90.92** | 97.10 | 99.21 | 50.06 | 51.76 |
| abl norm aniso (c10) | 92.98 | 99.84 | 102.11 | 52.05 | 53.79 |
| dposer-like full | 91.37 | **96.88** | **98.79** | **49.57** | **51.06** |
| dposer-like aniso full | 91.47 | 97.68 | 99.88 | 50.35 | 51.97 |

Per-joint noise is worse in every band: on c10 by 0.7–2.9 mm in mean (0.73 in the band 5.5–6.4, where the
isotropic front has no point of its own) and 2.0 mm in best of 10; on full by 0.1–1.1 mm. Its lowest mean on c10
(89.53 mm at APD 6.10, trial 71) equals that of the isotropic study (89.55 at 6.57, trial 118), at lower diversity.

In D-Flow, where the APD of every model is low, per-joint noise holds the front: on the test of section 15 of the
completion report, 6 of the 8 points of the front "RSGM dflow" (chosen on validation) are `dposer_like_aniso`.

### The IGSO(3) noise on the GPU (2026-11-16)

The aniso models were trained with the noise of the loss drawn on the GPU (11–14 min instead of 21–23 min on
workstation1); the validation losses agree with the CPU path within the gap between two seeds. A technical step,
no effect on the results.

### Motion denoising (2026-11-18)

Tuned on validation like the isotropic model, `abl_residual_aniso` reaches MPJPE 19.47 / 32.68 mm at 40 / 100 mm
noise against 18.63 / 30.83 mm of `abl_residual`; the rule fixed before the studies does not hold. A bug fixed on the
way: the SDS of denoising used one $s(t)$ for all joints, now `joint_s(t)`. The likely reason: the noise of the task
is isotropic in joint positions, a large angular error for short bones (spine, neck, collars), and these joints have
the smallest $c_j$ (1.0–10.0 against 19–43). Denoising keeps the isotropic RSGM.

### Where the line stood

The conclusions of the score network report left "about 1.3°" of the first phase open and named the isotropy within a
joint as one possible reason: one scale per joint, and IGSO(3) has no anisotropic heat kernel in closed form. The
same report listed among what it does not show that per-joint noise was compared only in the hybrid, not at equal APD.
The inpainting fronts above then showed the loss at equal APD.

## 2026-11-28: Anisotropy within a joint

### Question and hypothesis

Per-joint noise made inpainting worse at equal APD. Is the cause that each joint is noised isotropically, while the
rotations of a joint are strongly anisotropic? Hypothesis: with $c_j$ by the RMS spread, the narrow axes of the wide
joints (twist and abduction of the hips and knees) are destroyed first, so a per-joint network loses them. If so, a
noise per principal axis of a joint (an anisotropic kernel on SO(3)) would be the next step.

### Setup

- **Data.** c10 of PoseD-Flow's split (`~/data/AMASS_rotations_posedflow_c10`). Marginal statistics: 200,000 training
  poses (fixed random subset, seed 0); the Karcher means $M_j$ and the per-coordinate stds of
  `hp.models.rsgm.z_statistics`. Spreads around the mean and conditionals: the first 4000 validation poses of
  `load_reference("valid")`. Inpainting: the first 500 of the 1000 validation poses of the Optuna studies
  (`hp.evaluation.completion.valid_poses`).
- **Principal axes** of joint j: eigenvectors of the covariance of $\log(M_j^\top x_j)$ (rotation vectors), widest
  first; the principal stds are the square roots of the eigenvalues.
- **Conditional spread.** The pairwise matrix Fisher energy trained by maximum likelihood (D^1 only,
  `~/data/experiments/ace/pl/energy_bm.pt`). Its conditional $p(x_j \mid x_{-j}) \propto \exp\langle F_j, x_j\rangle$
  is matrix Fisher. RMS: one draw per validation pose, RMS angle to the truth divided by $\sqrt2$. Principal: with
  the proper singular values $s$ of $F_j$, the concentration about axis i is $s_j + s_k$ and the small-angle std
  $1/\sqrt{s_j+s_k}$, averaged over poses. The ML energy is broader than the data, so these spreads are upper bounds.
- **Drowning time** of axis i of joint j: $s^* = \sigma_{j,i}^2 / c_j$, the Brownian time at which the noise of that
  axis reaches its spread. Three scales: isotropic ($c = 1$), $c_j$ by the RMS of the three principal stds (as
  `fit_joint_scale`), $c_j$ by the narrowest axis.
- **Inpainting check** (`explore/ace_energy/axis_errors.py`). Networks `abl_norm` (isotropic,
  `~/data/experiments/rsgm_posedflow_abl_norm_s10/run_0001`) and `abl_norm_aniso` (per-joint noise,
  `~/data/experiments/igso_speed/rsgm_posedflow_abl_norm_aniso_s10/run_0001`): 1.2M parameters, 30k × 4096, seed
  10, c10. Inpainting of RSGM (`run(method="inpaint")`, EDM schedule, no replacement, resample 5), K = 10, occlusions
  left_leg and legs, at one point of each model's study with close APD on its 1000 poses: `abl_norm` trial 118 of
  `rsgm_inpaint_mean_apd_norm` (200 steps, guidance 0.0952; APD 6.57 cm) and `abl_norm_aniso` trial 97 of
  `rsgm_inpaint_mean_apd_norm_aniso` (400 steps, guidance 0.0481; 6.68 cm). Error along axis i:
  $v_i^\top(\log(M_j^\top \hat x) - \log(M_j^\top x))$. Mac (MPS), about 4 minutes.
- **Rule, fixed before the run** (in the script's docstring): on legs, over the hips and knees, $D_i$ = mean squared
  error along axis i of `abl_norm_aniso` minus that of `abl_norm`, $D = \sum_i D_i$. The hypothesis holds if $D > 0$,
  $D_1 + D_2 \ge \tfrac23 D$ and $D_0 \le \tfrac13 D$.
- A cheaper check first proposed, training RSGM with $c_j$ by the narrowest axis, was dropped before it ran: it gives
  the legs only $c$ = 2.8–3.9 (nearly isotropic) and would have tested mainly the arms.

### Anisotropy within a joint

Marginal: principal stds over 200,000 training poses; parent axes: largest / smallest per-coordinate std; conditional:
Fisher, 4000 validation poses. All 19 joints in `pl/joint_anisotropy.log`.

| joint | marginal principal std (rad) | max/min | parent axes max/min | conditional principal std (rad) | cond. max/min |
|---|---|---|---|---|---|
| LeftLeg (knee) | 0.543 / 0.175 / 0.069 | **7.8** | 3.4 | 0.242 / 0.108 / 0.057 | 4.2 |
| RightLeg (knee) | 0.545 / 0.159 / 0.069 | **7.9** | 4.7 | 0.225 / 0.102 / 0.058 | 3.9 |
| LeftUpLeg (hip) | 0.404 / 0.155 / 0.082 | 4.9 | 3.5 | 0.141 / 0.095 / 0.063 | 2.2 |
| RightUpLeg (hip) | 0.415 / 0.140 / 0.070 | 5.9 | 4.3 | 0.146 / 0.096 / 0.058 | 2.5 |
| Spine0 | 0.266 / 0.087 / 0.065 | 4.1 | 3.7 | 0.116 / 0.060 / 0.048 | 2.4 |
| LeftElbow | 0.517 / 0.257 / 0.141 | 3.7 | 2.3 | 0.220 / 0.137 / 0.086 | 2.6 |
| RightElbow | 0.550 / 0.276 / 0.160 | 3.4 | 2.3 | 0.242 / 0.141 / 0.087 | 2.8 |
| LeftShoulder (collar) | 0.247 / 0.156 / 0.073 | 3.4 | 2.4 | 0.081 / 0.075 / 0.058 | 1.4 |
| LeftHand | 0.319 / 0.268 / 0.134 | 2.4 | 2.2 | 0.184 / 0.148 / 0.087 | 2.1 |
| Neck | 0.174 / 0.159 / 0.092 | 1.9 | 1.8 | 0.135 / 0.094 / 0.072 | 1.9 |
| Spine2 | 0.076 / 0.045 / 0.042 | 1.8 | 1.8 | 0.058 / 0.041 / 0.040 | 1.4 |

The legs are the most anisotropic joints, about 60× in variance for the knees against the between-joint range of
$c_j$ of 1–42.6, and their principal axes are not the parent axes: the parent-frame axes, which DPoser's
z-normalisation and the z input of section 9 use, see only about half of the anisotropy (knees 3.4–4.7×).

### Marginal and conditional spread

RMS angle (rad) around the Karcher mean and around the Fisher conditional; $c$ = squared spread relative to Spine2.

| joint | marginal RMS | conditional RMS | c marginal | c conditional | cond / marg |
|---|---|---|---|---|---|
| LeftUpLeg | 0.417 | 0.181 | 12.4 | 4.7 | 0.43 |
| LeftLeg | 0.581 | 0.288 | 23.9 | 11.8 | 0.50 |
| LeftFoot | 0.264 | 0.200 | 4.9 | 5.7 | **0.76** |
| Spine2 | 0.119 | 0.084 | 1.0 | 1.0 | 0.71 |
| LeftShoulder | 0.316 | 0.116 | 7.1 | 1.9 | **0.37** |
| LeftElbow | 0.613 | 0.287 | 26.7 | 11.7 | 0.47 |
| LeftHand | 0.480 | 0.269 | 16.3 | 10.4 | 0.56 |

Given all other joints, a joint's spread is 0.36–0.76 of its marginal one. A conditional $c_j$ would keep the order of
the joints and halve its range (1–13.5 against 1–27.1); it differs in the collars (well predicted by their neighbours,
nearly isotropic, 1.4–1.5×) and in the feet and hands (poorly, 0.56–0.76). Not trained.

### Drowning times of the 57 axes

| noise | c_j range | range of s* | first axes destroyed (s*) | last axis (s*) |
|---|---|---|---|---|
| isotropic | 1 | 175× | Spine2 narrow 0.00173, 0.00204; Spine1 narrow 0.00240 | RightElbow wide 0.3023 |
| c_j by RMS spread | 1.0–42.6 | 63× | **knees narrow 0.00014, 0.00014; hips narrow 0.00024, 0.00033** | RightLeg wide 0.0086 |
| c_j by the narrowest axis | 1.0–14.7 | 62× | the narrowest axes of all joints, 0.00173 | RightLeg wide 0.1071 |

Per-joint scaling by the RMS spread equalises the joints but makes the narrow axes of the knees and hips the first
coordinates of the body to be lost, 12× before the narrowest spine axis under isotropic noise.

### The check: inpainting errors along the principal axes

500 validation poses, K = 10; MPVPE (mm), APD (cm).

| model | left_leg mean / best of 10 / APD | legs mean / best of 10 / APD |
|---|---|---|
| abl_norm (t118) | 86.21 / 43.72 / 7.04 | 108.62 / 62.80 / 9.11 |
| abl_norm_aniso (t97) | 87.14 / 45.99 / 6.75 | 105.02 / 64.14 / 8.15 |

Legs hidden: RMS error along the principal axes (rad, widest first) and $D_i$ (aniso − iso, rad², 95% bootstrap over
poses):

| joint | iso RMS | aniso RMS | D_0 | D_1 | D_2 |
|---|---|---|---|---|---|
| LeftUpLeg | 0.213 / 0.135 / 0.080 | 0.211 / 0.124 / 0.078 | −0.0006 [−0.0039, +0.0027] | **−0.0027 [−0.0044, −0.0010]** | −0.0003 [−0.0008, +0.0003] |
| RightUpLeg | 0.186 / 0.141 / 0.084 | 0.187 / 0.132 / 0.081 | +0.0005 [−0.0023, +0.0033] | **−0.0024 [−0.0044, −0.0007]** | −0.0006 [−0.0011, −0.0002] |
| LeftLeg | 0.431 / 0.188 / 0.077 | 0.420 / 0.184 / 0.076 | −0.0099 [−0.0258, +0.0055] | −0.0014 [−0.0043, +0.0015] | −0.0001 [−0.0006, +0.0006] |
| RightLeg | 0.442 / 0.179 / 0.077 | 0.429 / 0.168 / 0.074 | −0.0115 [−0.0257, +0.0029] | **−0.0038 [−0.0060, −0.0016]** | −0.0004 [−0.0009, +0.0001] |
| LeftFoot | 0.254 / 0.111 / 0.076 | 0.261 / 0.121 / 0.080 | +0.0034 [−0.0017, +0.0084] | +0.0022 [+0.0011, +0.0033] | +0.0006 [+0.0001, +0.0011] |
| RightFoot | 0.245 / 0.116 / 0.076 | 0.259 / 0.119 / 0.075 | +0.0069 [+0.0014, +0.0124] | +0.0007 [−0.0003, +0.0018] | −0.0002 [−0.0006, +0.0002] |

Rule (hips and knees): $D = -0.0332$ rad², $D_0 = -0.0215$, $D_1 = -0.0104$, $D_2 = -0.0014$: there is no extra error
to explain, and the hypothesis does not hold. With the left leg hidden the per-joint model is worse on the wide axis of
the hip ($D_0$ +0.0034 [+0.0014, +0.0053]) and on axis 1 of the knee and the foot (+0.0031 [+0.0003, +0.0059], +0.0028
[+0.0016, +0.0040]). On the narrow axes both networks reach about the spread of the data (knees 0.074–0.077 rad against a
std of 0.069), little better than the marginal alone.

### Conclusions of this phase

- The rotations of a joint are as anisotropic within the joint as the joints are between each other, most in the
  legs; a noise that only scales joints cannot equalise the signal-to-noise ratio of all coordinates.
- But the narrow axes are not where per-joint noise loses in inpainting: its narrow axes of the hips and knees are as
  good as or better than those of the isotropic network. The anisotropic kernel (noise per principal axis) is not
  built for this reason.
- At these two points the per-joint model trades diversity (lower APD, worse best of 10) as in D-Flow, and its extra
  error is at the feet and on the wide axis of the hip.

## Conclusions

1. The isotropic IGSO(3) noise was the main cause of the slow first phase of RSGM in the hybrid: a Brownian time per
   joint, matched to DPoser's z-scored noise, moves it halfway to DPoser; the same scaling at the input does not.
2. Per-joint noise lowers the diversity of every completion procedure. Where the procedure cannot trade diversity
   back (D-Flow), it holds the front; where it can (inpainting), the isotropic network is ahead at equal APD.
3. It does not help motion denoising, where the task's noise is isotropic in positions.
4. Within a joint the anisotropy is large, but the narrow axes of the legs are not where per-joint noise loses.
   Without another reason, an anisotropic kernel (option A: a wrapped anisotropic Gaussian, about 1.5–2 days; option
   B: an anisotropic Brownian heat kernel, a week or more) is not built.

## What it does not show

- Why per-joint noise lowers the APD, and whether a smaller range of $c_j$ (for example the conditional one, 1–13.5)
  would keep the first phase and the diversity. Not trained.
- What the extra error at the feet is; the feet are the joints least determined by their neighbours.
- Seeds: two for `abl_aniso`, one for every other network; the inpainting fronts, one study per model.
- The axis check: one point per model, at different APD on its 500 poses (7.04 against 6.75 cm on left_leg, 9.11
  against 8.15 on legs), enough to locate the extra error by axis, not to rank the models; the 500 poses are among
  the 1000 on which the points were chosen; c10 and 1.2M parameters only, `dposer_like_aniso` not checked.
- Principal axes are a linear summary of a curved distribution (the twist of a knee depends on its bend); the
  conditional spreads come from a model broader than the data.
- A noise per principal axis was not trained, so its effect on the hybrid and on completion is unknown; the check
  only removes the reason given for it.

## Where things are

- Code: `joint_scale`, `fit_joint_scale`, `z_statistics` in `src/hp/models/rsgm.py`; configs `abl_*_aniso` and
  `dposer_like_aniso` in `scripts/train_rsgm_posedflow.py`.
- Runs: `~/data/experiments/rsgm_posedflow_abl_{norm,residual,aniso}_s10`,
  `~/data/experiments/igso_speed/rsgm_posedflow_{abl_aniso_s11,abl_residual_aniso_s10,abl_norm_aniso_s10}`, the full
  runs of `dposer_like_aniso` as in the score network report.
- Studies: `~/data/experiments/completion/optuna/rsgm_{inpaint,dflow}_mean_apd_*aniso*.db` and their isotropic pairs;
  the band table from `explore/evaluation/completion_diagnostics/plot_study_fronts.py --bands`.
- Phase 2026-11-28, scripts (`explore/ace_energy/`): `joint_anisotropy.py` (spreads, principal and conditional stds,
  drowning times), `axis_errors.py` (inpainting of the two networks and the error along the principal axes, with the
  rule). Logs and outputs: `~/data/experiments/ace/pl/joint_anisotropy.log`; `~/data/experiments/ace/axis/`
  (`axis_errors_500.log`, the completions `abl_norm{,_aniso}_{left_leg,legs}_500.pt`, `rule_500.json`). The Fisher
  energy: `~/data/experiments/ace/pl/energy_bm.pt` ([its report](ace/2026-11-27_fisher_boltzmann.md)).
