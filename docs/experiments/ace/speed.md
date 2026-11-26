# ACE energy, 2026-11-26: the score made 2-4 times faster without changing the model

Branch `experimental/ace-descriptors`. No model was changed and nothing was evaluated; the numbers are timings and
errors against the code before. Part of the ACE line: [overview](README.md).

## Summary

- **Why.** Before an Optuna study of the MLP embedding and the basis ([MLP embedding](2026-11-26_mlp_embedding.md)),
  the code was made faster. Only exact changes: the same energy and the same score to float32 precision. Changes of the
  model (fewer density centres, dropping duplicate or odd columns, without the descriptors; notes in
  `explore/ace_energy/OPTIMIZATION.md`) were left for later on purpose.
- **Five steps**, each measured on its own with `explore/ace_energy/bench.py` against a reference saved with the code
  before (e9cd138); every error below 1.1e-5 of the largest entry (the rule: 1e-4).
- **GPU (workstation2), 500 poses, before → after:** the score of the MLP energy 30.4 → 6.8 ms at one noise level and
  30.3 → 7.0 ms at training noise levels (4.3-4.5×); a training step 52.5 → 25.7 ms (2.0×); DPS 58.7 → 31.8 ms
  (1.8×). The linear energy: 20.3 → 7.1, 19.3 → 6.0, DPS 42.5 → 31.4 ms.
- **Mac CPU (4 threads):** the MLP score 619 → 108 ms, DPS 1241 → 261 ms, a training step 1093 → 323 ms.
- **What is left is DPS** (about 31 ms on the GPU, the same for both models): the double backward through the
  descriptors and the pair harmonics. A closed form of those (step 3b) was deferred.

## The steps

| step | change | where | commit |
|---|---|---|---|
| 0 | the code before | | e9cd138 |
| 1 | the coefficients of each bin as one dense matrix [K, P, 19], cached, instead of interpolating the coefficients per pose | `Embedded.dense` | 981ff47 |
| 2 | the products of the 171 pairs of local rotations as a quadratic form in the 171 entries m of x: m^T A_k m with A [K, 19·9, 171] per bin, instead of 13,851 feature columns | `Embedded._matrices`, `WithHarmonics.forward(products=False)` | 981ff47 |
| 3a | the score of the joint density in closed form (responsibilities of the centres), its graph only when DPS or training need it | `JointDensity.scaled_score(graph=)` | 981ff47 |
| 4 | `torch.compile` of `feature_energy`, for the scores without a graph only | `Embedded.compile()`, `bench.py --compile` | 981ff47 |
| 5 | on a GPU, all bins used by a batch in one product, weighted by the interpolation; the CPU keeps the groups by bin | `Embedded.densities_split`, `Embedded.all_bins` (None: by device) | 3e122b1 |

- The energy is interpolated linearly between the centres of the bins of log s; a pose at t uses two neighbouring
  bins. Step 1 makes the matrices once per bin; step 5 avoids one loop over the bins of a batch, which on a GPU cost
  more than the extra work (the bins are few: a batch at training times touches all 16).
- Step 4: aot_autograd has no double backward, so compile cannot serve DPS or training (`create_graph`). `torch.func`
  inside compile fell back to eager. The compiled function recompiles for each count of bins (the warning of the
  recompile limit is harmless).

## Results: Mac, CPU

ms per call, median of 5 after one warm-up, 500 validation poses, 4 threads, mean of two interleaved rounds (steps 0-3a;
the Mac was shared with other jobs, so one round alone varies by 10-20%). Step 4 is one round; it does not apply to DPS
and training (–). `bench_mac.log`.

| model | case | 0 | 1 | 2 | 3a | 4 |
|---|---|---|---|---|---|---|
| linear P+L | score, one t | 278 | 231 | 204 | 159 | 117 |
| linear P+L | score, many t | 309 | 309 | 247 | 152 | 145 |
| linear P+L | DPS | 660 | 541 | 486 | 253 | – |
| MLP K8 | score, one t | 619 | 265 | 237 | 179 | 108 |
| MLP K8 | score, many t | 632 | 362 | 262 | 171 | 145 |
| MLP K8 | DPS | 1241 | 609 | 515 | 261 | – |
| MLP K8 | train step | 1093 | 648 | 490 | 323 | – |

- Step 1 gives most for the MLP (its K = 8 densities were interpolated per pose); step 3a halves DPS, where the graph of
  the joint density dominated.
- Step 5 is off on the CPU by default (the groups by bin are cheap there); on the Mac it was run only as a check of
  exactness (errors below 6.2e-6, `bench_mac.log`), not timed against the other steps.

## Results: GPU, workstation2

ms per call, the same cases, `cuda:0`. A repeat of step 0 differs by up to 2% (`bench_gpu.log`, `bench_gpu_step5.log`).

| model | case | 0 | 1-3a | 1-3a + compile | 1-3a + 5 | 1-3a + 5 + compile |
|---|---|---|---|---|---|---|
| linear P+L | score, one t | 20.3 | 14.8 | 6.6 | 14.5 | 7.1 |
| linear P+L | score, many t | 19.3 | 24.0 | 17.7 | 14.1 | **6.0** |
| linear P+L | DPS | 42.5 | 32.2 | 32.7 | **31.4** | 31.3 |
| MLP K8 | score, one t | 30.4 | 16.1 | 6.9 | 14.7 | **6.8** |
| MLP K8 | score, many t | 30.3 | 23.9 | 16.7 | 15.0 | **7.0** |
| MLP K8 | DPS | 58.7 | 32.1 | 33.1 | **31.8** | 32.8 |
| MLP K8 | train step | 52.5 | 44.8 | 46.6 | **25.7** | 26.4 |

- Without step 5 the scores at training times got slower on the GPU (19.3 → 24.0 ms): the groups by bin are many small
  kernels. Step 5 removes that and halves the training step.
- Compile adds a factor of about 2 to the scores without a graph and nothing to DPS and training.
- Largest error to the reference over all cases: 5.8e-6 (GPU), 1.1e-5 (CPU, step 1).

## Conclusions

1. **Training of the MLP is twice as fast on the GPU** (25.7 against 52.5 ms per step of 500 poses), the scores for
   sampling 4 times with compile: an Optuna study of the embedding costs half of what it did.
2. **Inpainting is bounded by DPS** (31 ms per call, 1.8× faster for the MLP than before, 1.35× for the linear energy).
   Further gains there need the closed form of the descriptors and the pair harmonics (step 3b), or changes of the model.
3. **Use:** `energy.compile()` for sampling and the likelihood ODE (scores without a graph); nothing to set otherwise:
   step 5 is on by itself on a GPU.

## What it does not show

- The time of whole runs (training, the harness, inpainting) was not measured; only calls of the score at 500 poses.
- Two machines; other GPUs or batch sizes may order the variants differently.
- Not tried: step 3b, half precision, CUDA graphs, and every change of the model (fewer density centres, duplicate and
  odd columns, without the descriptors).

## Where things are

- **Benchmark:** `explore/ace_energy/bench.py` (cases, models and the check in its docstring):

  ```bash
  python explore/ace_energy/bench.py --save-reference --label "before"   # with the old code
  python explore/ace_energy/bench.py --device cuda:0 --label "after"
  python explore/ace_energy/bench.py --device cuda:0 --compile
  ```

  Flags `--no-quadratic` (step 2 off) and `--all-bins` (step 5 on the CPU) time the variants without checking out old
  code; step 0 was timed from `git archive e9cd138`.
- **Logs and reference:** `~/data/experiments/ace/bench_mac.log`, `bench_gpu.log`, `bench_gpu_step5.log`,
  `bench_reference.pt` (on the Mac; workstation2 has its own reference made there).
- **Code:** `explore/ace_energy/embedding.py`, `fit.py` (`WithHarmonics.product_slice`), `joint_density.py`.
