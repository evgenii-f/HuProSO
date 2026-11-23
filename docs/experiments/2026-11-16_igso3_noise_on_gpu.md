# Drawing the IGSO(3) noise of RSGM on the GPU: 6-10x faster training steps, the same model

RSGM spent most of a training step on the CPU, not on the GPU. Where was the time going, and can it be moved to
the GPU without changing what is learned or sampled?

## Summary

- **The bottleneck was the noise, not the network.** Every training step drew its diffusion times, IGSO(3) angles and
  axes on the CPU in float64. It noised the batch there and copied it to the GPU. Inpainting did the same at every
  reverse step. The network itself (an MLP of about 1M parameters on a batch of 4096) takes well under a millisecond
  on an RTX 4090. With several runs on one machine, the CPU side made the processes compete for the cores: about 50
  cores per run, load average about 160 on workstation1. The GPUs stood at 0% SM.
- **The change.** Training draws the noise on the device when the device has float64 (cuda), still in float64.
  Inpainting draws its noise on the model's device in the model's dtype. The angle sampler interpolates a float64
  quantile table, kept on the CPU and copied to each device and dtype. For one diffusion time the level of the table
  is found on the CPU, so the GPU never waits for the CPU.
- **Speed (idle RTX 4090, benchmark).** A training step is 6.3x faster at batch 4096 (82 to 520 it/s) and 9.7x at
  16384. Inpainting without guidance is 6.1-7.6x faster per network call; with guidance it is 1.4-1.6x faster, as
  the backward pass then dominates.
- **Speed (loaded workstation1, real training).** A 30k-step run took 11-14 min on the new path. Runs on the old path
  that were started at the same time ran at 2-3 it/s, so a 30k-step run would take hours.
- **The same model.** On cuda the training loss has the same distribution as on the CPU path (a test over the 4
  weighting x noise variants), and the validation loss is the same number. Training curves agree within 0.02 nats
  at 1k-6k steps. After 30k steps three runs end within 0.002-0.011 nats of their CPU-path twins, against a gap of
  0.008 nats between two seeds of the same run. Inpainting metrics agree over 8 seeds (no significant difference in
  15 comparisons). The float32 sampler matches float64 to about 6e-8 on average.
- **Decision.** Merged into dev (2aaf2e8). All new trainings and inpainting studies run on the new path.
  `RSGM.noise_on_cpu = True` restores the old path for comparisons.

## Setup

**What RSGM draws.** The training loss (`RSGM.loss`, `src/hp/models/rsgm.py`) draws, per pose, a time t
(`loss_times`), per joint a Brownian time s_j(t), an IGSO(3) angle omega ~ f(omega; s_j) and a uniform axis u. It
then sets x_t = x_0 exp(omega u) and uses the target std_j(t) d/d omega log f(omega; s_j) u. The angle comes from the
inverse CDF: `igso3.AngleSampler` holds a quantile table [1024 levels of log s x 4097 probabilities] and interpolates
it in both directions. The target needs float64: near s = 0 the score is about 1/s, and the image sum of the density
underflows in float32. Inpainting (`rsgm_inpaint`, `src/hp/models/inpainting.py`) draws IGSO(3) noise of one Brownian
time for the visible joints (replacement) and for all joints (resampling) at every reverse step.

**Before** (dev before 2aaf2e8, e.g. b661709). All of it ran on the CPU in float64. The loss also moved the poses to
the CPU and the noised poses back. Inpainting drew on the CPU and copied every step (`.to(device)`), which also
synchronised the GPU every step.

**After** (`experimental/igso_gpu`: 4e47e47, 90c159a, 66d0006; merged into dev 2aaf2e8):

- `AngleSampler.table_on(device, dtype)`: copies of the float64 table, cached per device and dtype. The dtype is
  changed on the CPU side of the copy, since mps has no float64.
- `AngleSampler.sample(s, generator, dtype)`: draws on the device of s, in float64 by default.
- `AngleSampler.sample_at(s, shape, device, dtype)`: one Brownian time for all entries. The level is found from a
  Python float, so there is no range check on the device and no synchronisation.
- `igso3.random_axes(shape, generator, device, dtype)`.
- `RSGM.loss`: draws on `self.device` in float64 when the device has float64 (cuda, cpu). On mps (no float64) and
  for the validation loss (a fixed CPU generator, so that it stays the same number as before) it keeps the CPU path.
- `rsgm_inpaint`: draws on the model's device in the model's dtype (float32).
- `RSGM.noise_on_cpu` (a class attribute, not saved with the model): True restores the CPU path in both places. The
  benchmark uses it to time both ways in one process.

**Not changed.** SDS (`score_completion.py`) still draws its noise on the CPU. Sequence sampling draws on the CPU once
at the start. D-Flow and the unconditional sampler draw no IGSO(3) noise (Gaussian steps in the tangent space,
already on the device).

**Measurements.**

1. Benchmark, `scripts/benchmark_igso_noise.py`: an untrained MLP 512 x 4, 19 joints, uniform random poses (the time
   does not depend on the data). It times a training step (loss, backward, Adam; 100 steps after 10 warm-up) and a
   network call of inpainting (20 steps; resample 3 without guidance, resample 1 with guidance 0.1; 3 calls after 1
   warm-up). Both ways run in one process. Machines: bigmama (idle RTX 4090, torch 2.14.1, 16 CPU threads) and the
   Mac (mps, torch 2.13.0).
2. Real training on workstation1 (RTX 4090, loaded by other runs and Optuna studies): `abl_aniso` (timeres 512 x 2,
   per-joint noise, 30k steps x 4096, c10), seed 11, on both paths at the same time. Runs `abl_residual_aniso` and
   `abl_norm_aniso` (seed 10) on the new path. Configs in `scripts/train_rsgm_posedflow.py`. Report
   [2026-11-14_score_network_architectures.md](2026-11-14_score_network_architectures.md), section 10.
3. Equivalence of training: `tests/test_rsgm.py::test_the_training_loss_on_cuda_draws_its_noise_there_with_the_cpu_distribution`
   (std and likelihood weighting, with and without `joint_scale`). It compares the mean of 300 training losses on cuda
   against the CPU (within 4 standard errors) and requires the same validation loss (rel. 1e-5).
4. Equivalence of inpainting: `explore/evaluation/igso_noise/inpainting_equivalence.py`, on the Mac (mps). 300
   validation poses, K = 10, the four occlusions, RSGM small std (`full/rsgm_posedflow_small_s10`). Steps 100,
   resample 2, guidance 0.1, replace, EDM grid. 8 seeds per way. Welch's t-test per metric and occlusion.
5. Precision of the float32 sampler: `explore/evaluation/igso_noise/sampler_precision.py` (mps). The same uniforms go
   through the float64 table on the CPU and the float32 table on the device. KS distance of 400k draws each to the
   IGSO(3) angle CDF.

## Results

### Speed

Benchmark, network call or training step (`2026-11-16_igso3_noise_on_gpu_benchmark.txt`):

| measurement | cuda: noise on CPU | cuda: on device | speed-up | mps: noise on CPU | mps: on device | speed-up |
|---|---|---|---|---|---|---|
| train step, batch 4096 | 82.2 it/s | **519.8 it/s** | **6.32x** | 53.1 it/s | 51.9 it/s | 0.98x |
| train step, batch 16384 | 18.8 it/s | **183.2 it/s** | **9.74x** | 18.9 it/s | 18.7 it/s | 0.99x |
| inpainting, 5000 poses, guidance 0, resample 3 | 7.91 ms | **1.29 ms** | **6.11x** | 12.99 ms | 5.67 ms | 2.29x |
| inpainting, 5000 poses, guidance 0.1, resample 1 | 12.62 ms | 9.25 ms | 1.36x | 22.63 ms | 18.49 ms | 1.22x |
| inpainting, 20000 poses, guidance 0, resample 3 | 25.56 ms | **3.38 ms** | **7.56x** | 46.97 ms | 23.84 ms | 1.97x |
| inpainting, 20000 poses, guidance 0.1, resample 1 | 42.64 ms | 27.04 ms | 1.58x | 86.42 ms | 74.71 ms | 1.16x |

On cuda the noise was most of the cost. On mps the loss draws on the CPU either way (no float64), so only
inpainting gains.

Real training, 30k steps x 4096 (`train_time.json` of the runs):

| run | path | machine state | time |
|---|---|---|---|
| `abl_aniso` s11 | GPU | workstation1, loaded | 643 s (10.7 min), about 50 it/s |
| `abl_residual_aniso` s10 | GPU | workstation1, loaded | 813 s (13.6 min) |
| `abl_norm_aniso` s10 | GPU | workstation1, loaded | 679 s (11.3 min) |
| `abl_aniso` s11 | CPU, started alongside | workstation1, loaded | 2-3 it/s (6283 steps in 33.6 min) |
| `abl_aniso` s10 | CPU | workstation1, earlier | 1372 s (22.9 min) |
| `abl_base`, `abl_residual`, `abl_norm` s10 | CPU | workstation1, earlier | 861-991 s (14-17 min) |

The gain of a whole run depends on the load of the machine. The old path is limited by the CPU and the new one is
not, so the gap is largest on a busy server. The training script reaches about 50 it/s against 520 it/s for the bare
step: what remains is outside the loss. The DataLoader collates 4096 poses one by one on the CPU, and the machine
was loaded.

### The same model

Validation loss of `abl_aniso` seed 11 during training (nats; the old path from its log on workstation1):

| step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| noise on CPU | 2.03 | 1.83 | 1.74 | 1.69 | 1.67 | 1.64 |
| noise on GPU | 2.03 | 1.81 | 1.72 | 1.68 | 1.66 | 1.64 |

Final validation loss after 30k steps, the same configs and seeds on the two paths (nats; `history.json` of each
run, workstation1, cuda; the configs differ only in `run_root`):

| run | noise on CPU | noise on GPU | difference |
|---|---|---|---|
| `abl_aniso`, seed 11 | 1.5470 | 1.5486 | +0.0016 |
| `abl_residual_aniso`, seed 10 | 1.5406 | 1.5514 | +0.0108 |
| `abl_norm_aniso`, seed 10 | 1.5516 | 1.5562 | +0.0046 |

For scale: `abl_aniso` seeds 10 and 11, both on the CPU path, end at 1.5389 and 1.5470 (0.0081 apart). The GPU
runs end higher in all three rows (mean +0.006 nats). Three of three with the same sign is not evidence of a shift
(p = 0.25, two-sided sign test). Two of the differences are within the seed gap; `abl_residual_aniso` is slightly above
it. A systematic shift of the order of 0.005 nats cannot be excluded with one seed per path.

The CUDA tests pass on bigmama (8 passed): the loss in all four variants and the float32 sampler on cuda.

Inpainting, 300 validation poses, K = 10, mean +- std over 8 seeds (`2026-11-16_igso3_noise_on_gpu_inpainting.json`):

| metric | occlusion | noise on CPU | noise on device | difference | p (Welch) |
|---|---|---|---|---|---|
| MPVPE mean (mm) | left_leg | 96.02 +- 0.78 | 96.56 +- 0.69 | +0.54 | 0.16 |
| MPVPE mean (mm) | legs | 120.81 +- 0.38 | 120.27 +- 1.34 | -0.54 | 0.30 |
| MPVPE mean (mm) | arms | 134.76 +- 1.40 | 134.73 +- 1.12 | -0.03 | 0.97 |
| MPVPE mean (mm) | trunk | 46.63 +- 0.26 | 46.40 +- 0.23 | -0.23 | 0.08 |
| MPVPE mean (mm) | mean | 99.55 +- 0.40 | 99.49 +- 0.49 | -0.06 | 0.78 |
| APD (cm) | left_leg | 8.36 +- 0.12 | 8.43 +- 0.15 | +0.07 | 0.33 |
| APD (cm) | legs | 10.70 +- 0.11 | 10.67 +- 0.12 | -0.03 | 0.62 |
| APD (cm) | arms | 9.43 +- 0.15 | 9.53 +- 0.14 | +0.11 | 0.17 |
| APD (cm) | trunk | 2.29 +- 0.03 | 2.27 +- 0.04 | -0.02 | 0.29 |
| APD (cm) | mean | 7.69 +- 0.07 | 7.72 +- 0.06 | +0.03 | 0.33 |
| best of 10 (mm) | left_leg | 44.95 +- 0.51 | 45.45 +- 0.72 | +0.50 | 0.13 |
| best of 10 (mm) | legs | 68.45 +- 1.27 | 68.79 +- 1.56 | +0.33 | 0.65 |
| best of 10 (mm) | arms | 71.41 +- 1.12 | 71.29 +- 1.16 | -0.12 | 0.84 |
| best of 10 (mm) | trunk | 25.04 +- 0.25 | 24.84 +- 0.36 | -0.20 | 0.22 |
| best of 10 (mm) | mean | 52.46 +- 0.40 | 52.59 +- 0.59 | +0.13 | 0.62 |

No difference is significant (Welch p >= 0.08 over 15 comparisons; the largest, trunk mean, is 0.23 mm). With the
same seeds, `noise_on_cpu = True` gives exactly the numbers of the old code (two separate checkouts, run before the
switch existed), so the switch reproduces the old path.

Precision of the sampler (400k draws per s, `2026-11-16_igso3_noise_on_gpu_benchmark.txt`):

- The mean relative error of the angle in float32 is 4e-8 to 6e-8, the rounding of float32.
- The largest error is 1.1e-4 relative. It occurs only in the far tail, where the table is steep and float32 rounds
  the uniform itself (24 bits over 4096 cells). For a median angle of 28 deg that is 0.009 deg.
- KS distance to IGSO(3) is 0.0008 (float64) and 0.0017 (float32), against a 1% threshold of 0.0026.
- The noise is used in float32 by the network and by `exp_frame` anyway.

## Conclusions

- RSGM's training and inpainting on a GPU were limited by drawing the noise on the CPU. Drawing it on the device makes
  a training step 6-10x faster and inpainting without guidance 6-8x faster on an RTX 4090.
- The method is unchanged: the same loss distribution, the same validation loss, the same inpainting metrics within
  the spread of seeds. Results before and after compare directly; random draws differ, so a run or study started on
  one path is not resumed on the other.
- Next, measured but not done:
  - The training script now spends about 90% of a step outside the loss (50 against 520 it/s). Keeping the training
    poses on the GPU and slicing a random permutation (0.3 ms instead of 3.3 ms per batch on the Mac) is the next step.
  - SDS still draws on the CPU; the same change applies there.
  - With guidance, inpainting is dominated by the backward pass through the network.
- For a paper this is an implementation detail, not a contribution. A sentence fits in the implementation section:
  "IGSO(3) noise is sampled on the GPU by inverse-CDF interpolation of a precomputed float64 quantile table". So does
  the speed (520 training steps/s at batch 4096 on an RTX 4090 for a 1M-parameter score network).

## What it does not show

- The final validation loss is compared on one seed per path for three configs. A shift smaller than about
  0.01 nats would need several seeds per path to detect. The finished models were not compared in completion or
  denoising.
- Inpainting metrics on cuda were not compared (only on mps). The code path is the same (float32 on the device),
  and the sampler on cuda passes its tests.
- One network size in the benchmark (MLP 512 x 4) and one GPU type (RTX 4090). Larger networks gain less in relative
  terms, since the network then takes more of the step.
- Whole runs on the old and new path were timed on a loaded machine, not on an idle one.

## Where things are

- Code: `src/hp/models/igso3.py` (`AngleSampler`, `random_axes`), `src/hp/models/rsgm.py` (`RSGM.loss`,
  `noise_on_cpu`), `src/hp/models/inpainting.py` (`_igso3_noise`). Tests: `tests/test_igso3.py`, `tests/test_rsgm.py`.
- Benchmark: `scripts/benchmark_igso_noise.py`. Equivalence and precision: `explore/evaluation/igso_noise/`
  (`inpainting_equivalence.py`, `sampler_precision.py`).
- Outputs: [2026-11-16_igso3_noise_on_gpu_benchmark.txt](2026-11-16_igso3_noise_on_gpu_benchmark.txt),
  [2026-11-16_igso3_noise_on_gpu_inpainting.json](2026-11-16_igso3_noise_on_gpu_inpainting.json).
- Runs of the new path: `~/data/experiments/igso_speed/rsgm_posedflow_{abl_aniso_s11,abl_residual_aniso_s10,abl_norm_aniso_s10}`
  (on the Mac and on workstation1). Runs of the old path: the same three in `~/data/experiments/cpu_noise/` (on the
  Mac; on workstation1 in `~/data/experiments/`), and `~/data/experiments/rsgm_posedflow_abl_aniso_s10`.
- Branch: `experimental/igso_gpu` (merged into dev, 2aaf2e8).
