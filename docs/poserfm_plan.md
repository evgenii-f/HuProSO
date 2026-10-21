# PoseRFM in the model zoo: plan

Branch `experimental/poserfm` (from `dev`).

## Goal of this task

Implement PoseRFM, the Riemannian flow matching prior of PoseD-Flow (Nadar, Foti, Birdal, CVPR 2026), and add it to the model zoo as a `PoseModel`: model, config, tests, and a short training run that shows it learns. Training it at the full budget and comparing it with RSGM and the HuProSO3 flow are follow-ups (below).

With PoseRFM in the zoo, three paradigms on SO(3)^19 can be compared on the same data, in the same density space and with the same metrics: a discrete normalizing flow (HuProSO3), score-based diffusion (RSGM) and flow matching (PoseRFM).

Why we implement it ourselves: the code that the paper announces (github.com/circle-group/PoseDFlow) is not public (404, checked 2026-10-15). Even with their checkpoint, the log-likelihood would not be comparable: they model 21 joints, we model 19, and a CNF cannot be marginalized over 2 joints.

## What the paper does (the parts we need)

- **Path.** Uniform source on SO(3)^K (not stated explicitly; implied by Fig. 1 and Chen & Lipman 2024). Geodesic interpolation per joint, x_t = Exp_{x0}(t Log_{x0}(x1)), conditional target u_t(x_t | x1) = Log_{x_t}(x1) / (1 - t), t ~ U(0, 1). Convention: t = 0 is noise, t = 1 is data (the opposite of our RSGM).
- **Network.** MLP with 4 hidden layers of 512, Swish (SiLU). Input: the K rotation matrices flattened (9K) and t. Output: tangent vectors (3K).
- **Training.** 50k steps, AdamW lr 1e-3, weight decay 1e-4, EMA 0.99, batch 4096, ReduceLROnPlateau (factor 0.5, patience 5 evaluations). 8 h on an A30.
- **Sampling.** Midpoint solver, 100 steps. In their ablation, retraction onto SO(3) only at the last step works as well as an Exp map at every step.
- **Data.** The usual AMASS split (= our `ORIGINAL_AMASS_SPLITS`), sequences trimmed by 10% at both ends, 30 Hz, about 18M poses, **K = 21** joints. Release not stated.
- **Metrics.** FID on joint positions, APD, d_NN (the NRDF protocol, 20 sets of 500 poses). No likelihood, and no comparison with HuProSO3.

## Steps of this task

### 1. The model `src/hp/models/poserfm.py`

- `@register_model("poserfm")`, `has_log_prob = True`, `density_space = "SO(3)^J"`, representation "matrix".
- **Network:** the time-conditioned MLP on flattened rotation matrices, output: tangent coordinates [B, J, 3] in the frame of x (as in RSGM). Defaults of the paper: hidden [512] × 4, silu.
- **`loss`:** x0 uniform (`roma.random_rotmat`), t ~ U(0, 1 - δ), x_t = x0 · exp(t · log(x0ᵀ x1)) per joint, target log(x_tᵀ x1) / (1 - t) in the frame of x_t, squared norm summed over the joints (the angle metric, as in RSGM). In eval mode times and source points from a fixed seed, so that a validation loss is the same number every time.
- **`sample`:** from uniform x0, geodesic midpoint steps from t = 0 to 1 (`sample_steps = 100`), every step on SO(3) (`exp_frame`), so samples are valid rotations without a final projection.
- **`log_prob`:** the ODE backwards from the data (t = 1) to t = 0 with the exact divergence (`torch.func.jacfwd`; the frame fields are divergence free, as in RSGM): log p₁(x) = log p₀(x₀) - ∫ div v dt, with log p₀ = 0 (uniform). Comparable with RSGM and the flow.
- **Reuse, not refactor:** the small helpers of RSGM (`exp_frame`, `move`, the activations) are imported from `hp/models/rsgm.py`; RSGM itself is not changed in this task. A shared module for both models can come later, once both exist.
- **Numerical points to settle (and test):**
  - the target 1 / (1 - t) blows up as t → 1: δ for training, and the likelihood integrates over [0, 1 - δ] (the density of the slightly smoothed data, as RSGM's eps);
  - Log is ambiguous at angle π (the cut locus), which occurs for uniform x0: such pairs are rare; make the loss robust (no NaN) and measure how often it happens;
  - float32 on mps: the log map near angle 0 and π.

### 2. Config `config/models/poserfm.yaml`

The paper's model and training settings where the trainer has them: hidden [512] × 4, silu, AdamW lr 1e-3 with weight decay 1e-4, EMA 0.99, batch 4096, 50k steps, 100 sampling steps; the number of likelihood steps chosen by convergence (as for RSGM: 100 / 200 / 400).

**Deviation from the paper:** ReduceLROnPlateau is not in the trainer (a separate task). Until then the step schedule (or none); the config says so.

### 3. Tests `tests/test_poserfm.py`

- The path: x_t at t = 0 is x0 and tends to x1 as t → 1; the target equals the finite-difference velocity of the path.
- A zero vector field gives log_prob = 0 (uniform) and leaves samples uniform.
- The likelihood ODE is right for a field with a known flow (e.g. a constant rotation field: a volume-preserving flow, log_prob unchanged), and for one joint exp(log_prob) integrates to 1 over uniform rotations (Monte Carlo).
- Samples are valid rotations; the validation loss is reproducible in eval mode; the loss is finite on mps and at the cut locus.
- The model contract test (`tests/test_model_contract.py`) picks the model up from the registry.

### 4. A short training run

Smoke run on every 30th frame (a few thousand steps) with the training script: the loss goes down, samples are valid, the log-likelihood of validation poses is finite and rising. Also the cost: steps per second, sampling and likelihood time per 1000 poses on mps.

## Follow-ups (separate tasks)

- **ReduceLROnPlateau** in the trainer (`scheduler: plateau`), to train with the paper's recipe.
- **Full training:** PoseRFM on the data of our best RSGM baseline (trial 34 on every 10th frame: 139.1-139.2 nats) and/or all frames, **with at least two seeds**: the noise between seeds of RSGM is ~0.2 nats (`docs/rsgm_mirror_augmentation.md`), so single runs cannot tell differences of a few tenths of a nat.
- **Evaluation:** the HuProSO3 protocol (`hp/evaluation/huproso3_protocol.py`), the harness of the model zoo with the contrast, robust likelihood statistics and the rare poses of `docs/rsgm_rare_poses.json`, next to RSGM and the flow.
- **The metrics of PoseD-Flow / NRDF** (FID on joint positions, APD, d_NN, 20 × 500 poses) in the protocol, for all models: the only bridge to their Tab. 1 (with the caveat of other data and 19 vs. 21 joints).
- A shared manifold-ODE module for RSGM and PoseRFM (with a golden test that RSGM's numbers do not change).
- Optional: a Euclidean flow matching baseline (PoseFM of the paper), for the effect of the geometry; Riemannian D-Flow (guidance for completion, denoising, HMR) as a common downstream test of the priors.

## Not to do

- Mirror augmentation: tried for RSGM, it makes every metric worse (`docs/rsgm_mirror_augmentation.md`).
- 21 joints: the whole pipeline is built on 19.
