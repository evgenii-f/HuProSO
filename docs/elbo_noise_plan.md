# Learned per-coordinate noise of DPoser by the ELBO: formulation and plan

Status: plan, 2026-11-21; phase 1 implemented and tested (section 5), not run. Context: the reverse test and the per-axis noise of RSGM
(`docs/experiments/2026-11-14_score_network_architectures.md`, sections 15-16) show that the geometry of the noise is a
part of DPoser's advantage in score distillation, and that the rule from the data (every coordinate at the same
signal-to-noise ratio, alpha = 1) is the best of a one-parameter family. The question here: does a noise geometry
chosen by the likelihood do better than that rule? DPoser is the test bed because it is Euclidean, so the ELBO is
exact and known; RSGM would follow only if the answer is yes.

## 1. Two phases

1. **Phase 1, the ELBO.** The network theta and the noise d (57 numbers) are trained together on the continuous-time
   ELBO. Its only product is d.
2. **Phase 2, the score loss.** d is frozen and DPoser is trained from scratch with its own loss and recipe:
   `train_dposer_posedflow.py` with `model.args.noise_scale = d`, as the `joint` and `iso` runs. It needs no new code.
   It is compared with the control (d = 1) and with `joint` and `iso`.

Why two phases: the ELBO is the objective that makes d identifiable without a trivial solution (section 4), while the
ELBO weighting of the network is known to cost sample quality (likelihood weighting of RSGM, section 13 of the
report). Phase 2 isolates the effect of d.

## 2. The forward process with d

DPoser's sub-VP SDE (`hp.models.dposer`, beta_min 0.1, beta_max 20) in its z-scored axis-angle space R^57:

    log alpha(t) = -t^2 (beta_max - beta_min) / 4 - t beta_min / 2,   beta(t) = beta_min + t (beta_max - beta_min)
    std(t) = 1 - alpha(t)^2                                             (sub-VP: the variance is (1 - alpha^2)^2)
    x_t = alpha(t) x_0 + std(t) d * eps,   eps ~ N(0, I)               (* elementwise)

The noise d is parametrized as

    d_k^2 = 57 softmax(l)_k,   k = 1..57,   l = 0  <=>  d = 1, DPoser as published.

Per coordinate, in the notation of VDM (Kingma et al. 2021, Eq. 2 and 5) and MuLAN (Sahoo et al. 2024, Eq. 2):

    SNR_k(t) = alpha^2 / (std^2 d_k^2) = exp(-gamma_k(t)),   gamma_k(t) = gamma(t) + 2 log d_k,
    gamma(t) = 2 log(1 - alpha^2) - 2 log alpha,   gamma'(t) = beta(t) (1 + alpha^2) / (1 - alpha^2).

d does not depend on t, so **gamma'_k = gamma' is the same for every coordinate**; d shifts each coordinate's log-SNR
curve by a constant. (gamma' is my derivation from DPoser's alpha; it is positive and behaves as 2 / t near t = 0.)
Endpoints: t_0 = loss_eps = 1e-5 gives gamma(t_0) = -27.6 (noise std 1e-6 in z units); t_1 = 1 gives gamma(1) =
10.05 (alpha = 0.0066). With sample_eps = 1e-3 instead, gamma(t_0) = -18.2.

## 3. The ELBO

VDM's decomposition (Eq. 11): -log p(x) <= prior + reconstruction + diffusion. For DPoser with d, per pose, in nats of
the density on z-scored R^57:

**Prior** (VDM Eq. 7 with p(z_1) = N(0, diag d^2), the start of DPoser's sampler):

    L_prior = sum_k 1/2 [ std_1^2 - 1 - log std_1^2 + alpha_1^2 x_0k^2 / d_k^2 ]

**Reconstruction.** VDM defines p(x | z_0) for discrete data (Eq. 8, proportional to q(z_0 | x) and normalized over the
256 values); for continuous data I take the same construction normalized over R, a Gaussian:
p(x | z_0) = N(x; z_0 / alpha_0, (std_0 / alpha_0)^2 diag d^2). Its expectation under q:

    L_rec = sum_k [ 1/2 log(2 pi e (std_0 / alpha_0)^2) + log d_k ]

**Diffusion**, continuous time, eps form (VDM Eq. 17; multivariate as MuLAN Eq. 11, with diag(gamma') = gamma' I here):

    L_diff = 1/2 E_{t ~ U(t_0, 1), eps} [ (1 - t_0) gamma'(t) || eps - eps_hat(x_t, t) ||^2 ]

The sign and the constants check out against VDM's x form (Eq. 15-16): x_0 - x_hat = std d (eps_hat - eps) / alpha
coordinatewise, so -SNR_k' ||x_0k - x_hat_k||^2 = gamma' (eps_k - eps_hat_k)^2. eps_hat is the network's output as now
(`noise_prediction`), the score is -eps_hat / (std d) as now.

**Variance of the estimate.** gamma' grows as 2 / t at small t. Sampling t uniformly in gamma instead of in t
(gamma(t) is monotone and has a closed-form inverse: alpha = (sqrt(e^gamma + 4) - e^(gamma/2)) / 2 from
e^(gamma/2) = (1 - alpha^2) / alpha, then t from the quadratic log alpha(t)) makes the weight constant:

    L_diff = 1/2 (gamma(1) - gamma(t_0)) E_{gamma ~ U, eps} || eps - eps_hat ||^2,

the eps loss averaged uniformly over log-SNR. VDM uses a low-discrepancy sampler of t (App. I.1: t^i = mod(u_0 + i/k,
1) over the batch) and learns the interior of its schedule to minimize the variance (App. I.2); it has no importance
sampling of t. The uniform-in-gamma draw is my choice, plus the low-discrepancy sampler on top of it.

**Phase 1 loss:** E_x [L_prior + L_rec + L_diff], gradients into theta and l. Gradients into l come through x_t (the
reparametrized noise), the prior and the reconstruction term.

## 4. What the ELBO can and cannot learn about d

This corrects what I said in the conversation of 2026-11-21 (that the correlations between coordinates make d
identifiable).

- **With a Bayes-optimal denoiser the ELBO does not depend on d**, up to the boundaries. The gradient of the mutual
  information of a Gaussian channel with respect to the SNR of each coordinate is half the MMSE of that coordinate
  (Palomar and Verdu 2006, the vector I-MMSE relation; to be checked against the paper before it is cited), so the
  MMSE field is a gradient and its line integral does not depend on the path of the SNR vector: it is -log p(x) up to
  the endpoint terms, correlated data or not. The shift of the endpoints by -2 log d_k cancels between L_diff (at high
  SNR the eps error of coordinate k tends to 1, over a stretch of 2 |log d_k| in gamma) and the log d_k of L_rec.
- **So d is chosen by two things only:**
  1. *The network's errors.* MuLAN's argument (Sec. 3.5) is that for a fixed, imperfect denoiser the integrand is not
     a conservative field and the ELBO depends on the path; VDM's invariance (Sec. 5.1, Eq. 18) holds for scalar
     schedules only. Learned d picks the order in which coordinates come out of the noise that this network family
     denoises best. Whether that order is also the best for score distillation is the question of phase 2.
  2. *The boundaries.* The cancellation needs the high-SNR end to be in the regime where every coordinate's eps error
     is 1. If the data are close to a lower-dimensional set at the scale of the noise at t_0 (a coordinate nearly a
     function of others), the eps error of the normal directions tends to 0 instead, the cancellation fails, and the
     ELBO gains without bound by sending d_k of those directions to 0. With std(t_0) = 1e-6 this is a real risk.
- **The softmax removes one direction the ELBO cannot fix:** a common scale of all d_k changes the ELBO only through
  the boundaries. With the sum of d_k^2 fixed only the shape is learned. (MuLAN fixes the endpoints gamma_min, gamma_max
  for the same kind of reason; App. E.2.)

Safeguards: log the three terms and their gradients with respect to l separately; repeat phase 1 with t_0 = 1e-3; if
some d_k run away to 0, bound the logits (or a floor on d) and say so.

## 5. Checks of the code

Implemented (2026-11-21) as a separate model, `dposer_elbo` (`hp/models/dposer_elbo.py`, a subclass of DPoser with
`learn_noise`, `noise_logit_scale`, `elbo_t0`), so that `hp/models/dposer.py` is unchanged;
`tests/test_dposer_golden.py` pins DPoser as published (golden values of commit 7847844) and with noise_scale. The
configs `elbo`, `elbo_t0` (model `dposer_elbo`) and `learned` (`--noise-from`, model `dposer`) of
`scripts/train_dposer_posedflow.py`; `explore/evaluation/learned_noise.py` for the convergence of d, d per joint,
rule 1 and the ELBO terms. Tests in `tests/test_dposer_elbo.py` (3-D Gaussian with correlations 0.8 and 0.4, 200,000
samples, float64):

1. With the exact denoiser E[eps | x_t] the -ELBO of the code is the entropy of the Gaussian within 0.4 nats for d^2 =
   (1, 1, 1), (0.1, 0.4, 2.5) and (2.8, 0.1, 0.1): 3.35 against 3.45 for all of them. This checks the formulas and the
   invariance of section 4 at once.
2. The denoiser that ignores the correlations is invariant too: it is the exact denoiser of N(0, diag Sigma), so its
   field is also conservative; its -ELBO is that model's cross-entropy (4.16 for all d). So "imperfect" in section 4
   must mean *not the posterior mean of any distribution*, not just "the posterior mean of the wrong one".
3. A denoiser of limited form (one shrinkage factor for all coordinates, the best of its form) gives a -ELBO that
   depends on d: 4.16 at d = 1, 5.48 at d^2 = (0.5, 0.5, 2), 8.55 at (0.1, 0.1, 2.8). This is the mechanism by which
   phase 1 learns d: through what the network can represent.
4. The inverse of gamma is exact to 1e-6 relative in float32 from t = 1e-5 to 1; gamma' matches a finite difference.
5. A 300-step smoke run on MPS (14 it/s at batch 4096, so about 36 min for phase 1 on the Mac): the loss falls, the
   logits move, checkpoints and EMA carry them, `learned_noise.py` reads them; its -ELBO (1658 nats) is above the
   exact -log p of the ODE (1322), as it must be.

Still to check on a trained phase-1 model: that the ELBO stays below the exact log-likelihood.

## 6. Runs and rules (to be fixed before the runs)

- **Phase 1:** c10 and the RSGM budget (about 10 minutes on workstation2) if d settles there, else the full data; two
  seeds; separate learning rate for l (to be tuned on the first run only for stability). Progress, soft stop and resume
  as all training.
- **Rule 1 (is d determined?):** the correlation of log d over the 57 coordinates between the two seeds is > 0.8, and
  d is the same with t_0 = 1e-3. Otherwise phase 2 is not run and the result is "the ELBO does not determine d here".
- **Phase 2:** full data, the paper's budget, seed 10, `noise_scale` = the mean of the two seeds' d.
- **Rule 2 (does it help?):** against the control (d = 1) on validation, `trajectory.py` left_leg at step 100 and the
  mean of four occlusions of `hybrid.py ... dposer`: an effect if lower by more than 0.2 deg and 2 mm (the spread of
  seeds seen so far); worse by as much, a negative result; in between, a second seed of phase 2.
- **Also reported:** d per joint and axis against the data rule (d = 1), and the ELBO and exact likelihood of phase 2
  against the control.

## 6a. Outcome (2026-11-24)

Run and closed as a negative result; the numbers are in section 20 of
[the score report](experiments/2026-11-14_score_network_architectures.md). Phase 1: with logit scale 10 d collapses
onto one coordinate (an optimization artifact, the ELBO 3 nats worse than d = 1); with scale 1 rule 1 passes
(correlation of log d 0.98-1.00) and the ELBO is 0.6-1.0 nats better than d = 1. Phase 2: by rule 2 negative (left leg
at step 100 +0.40 deg, hybrid +5.5 mm); on the test with the PoseD-Flow protocol not better than the published DPoser
(FID 0.0126 ± 0.0010 against 0.0114 ± 0.0006). The learned d is kept as data
(`~/data/experiments/learned_noise_elbo_slow_rule1.txt`).

## 7. What it does not answer

- Pose-dependent noise d(x_0): the score is then no longer -eps_hat / (std d) of a Gaussian convolution, and SDS would
  need a rule for which pose to take d from.
- The SO(3) version: the wrapped normal of RSGM's per-axis noise is not a diffusion, so its ELBO is not this one.
- Novelty: this phase is MuLAN without input conditioning, applied to poses; a diagnostic, not a contribution.

## References (to check before citing)

- Kingma, Salimans, Poole, Ho, Variational Diffusion Models, NeurIPS 2021 (arXiv 2107.00630): Eq. 2, 5, 7, 8, 11, 15-18;
  App. I.1-I.2. Read from the HTML version, 2026-11-21.
- Sahoo, Gokaslan, De Sa, Kuleshov, Diffusion Models With Learned Adaptive Noise (MuLAN), NeurIPS 2024 (arXiv
  2312.13236): Eq. 2, 11, 12; Sec. 3.5, 4.4; App. E.2. Read from the HTML version; App. C and E.5 not read.
- Palomar, Verdu, Gradient of mutual information in linear vector Gaussian channels, IEEE Trans. Inf. Theory 2006: not
  read, from memory.
- Song, Durkan, Murray, Ermon, Maximum likelihood training of score-based diffusion models, NeurIPS 2021: not read.
