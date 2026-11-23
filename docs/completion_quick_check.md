# Quick check of a completion model

A new model (or new sampling settings) through pose completion in minutes instead of hours: the first 200 poses of
the fixed validation draw, the four occlusions, K = 10. The numbers below are the yardstick to put it next to.

## How

```bash
python scripts/evaluate.py completion --run ~/data/experiments/new_s10/run_0001 --quick --together
```

- `--quick` is `--split valid --poses 200`: the first 200 of the 1000 validation poses of the Optuna studies (a
  random subset fixed by its seed), results in `<out>/valid_first200`. `--poses n` changes the number. Never the test.
- The method and its settings: D-Flow with their protocol's defaults unless told otherwise. `--like LABEL` takes the
  method and the tuned settings of a row of the registry (`MODELS` in `scripts/evaluate_completion.py`) or of a point
  of `config/completion_test_points.yaml`, e.g. `--like "DPoser: their full, inpaint t90"`; `--method`, `--settings
  JSON` and `--set key=value` set or override them. The defaults of `inpaint` and `dposer` are those of the functions,
  not tuned: without `--like` they score worse than the rows below.
- Add `--models "RSGM 512x4: their c10, inpaint t52"` (or any row) to compute a reference on the same machine in the
  same table. A cached row is reused only on the machine that made it (the cache knows the weights by size and
  modification time), so elsewhere compare with the tables below rather than recompute 64 models (about 3 h on one GPU).
- Finished results are skipped; `--recompute` replaces results made with other settings, `--force` recomputes all.
  `--dry-run` shows what is cached.

## How far to trust it

On random subsets of the test caches (71 models, 5000 poses), the paired difference of two models' mean MPVPE has an
sd of 1.4 mm at 200 poses (1.9 at 100, 0.6 at 1000), and 0.3% of the pairs that differ by 2 mm or more swap order. On
the 43 points of the fronts, these 200 poses give a mean MPVPE 0.8 mm lower than the recorded 1000 (sd 1.35, max
3.0), best of 10 0.6 lower (sd 0.9), APD the same (sd 0.19); Spearman 0.98 / 0.96 / 0.99. So: a change of a few mm
shows; settings closer than about 2 mm are chosen on the 1000 poses (`--split valid`).

## Reference numbers (2026-11-23)

MPVPE in mm (mean over K per occlusion; mean and best of 10 over the four occlusions), APD in cm; 200 validation poses,
`--together`, batch 500, on workstation2 (dev 2b71383). Sorted by best of 10. Per row and occlusion in
`~/data/experiments/completion/cache/valid_first200/results.jsonl` (workstation2 and the Mac).

### Rows of the registry

| model | left_leg | legs | arms | trunk | mean | best of 10 | APD |
|---|---|---|---|---|---|---|---|
| DPoser: their full, inpaint t90 | 82.6 | 103.9 | 118.1 | 42.6 | 86.8 | 47.6 | 6.49 |
| RSGM dposer-like: their full, inpaint t108 | 86.6 | 108.6 | 120.4 | 40.0 | 88.9 | 47.7 | 6.61 |
| RSGM abl norm: their c10, inpaint t118 | 87.1 | 107.9 | 115.5 | 41.2 | 87.9 | 48.2 | 6.50 |
| RSGM abl residual: their c10, inpaint t81 | 86.5 | 106.2 | 114.4 | 41.9 | 87.3 | 48.3 | 6.41 |
| DPoser: their full, DPoser method t28 | 91.7 | 110.8 | 118.4 | 46.1 | 91.7 | 48.4 | 7.43 |
| DPoser: their c10, RSGM budget, inpaint t68 | 86.5 | 104.2 | 114.2 | 43.6 | 87.1 | 48.5 | 6.40 |
| RSGM deep: their c10, inpaint t75 | 89.1 | 109.8 | 116.1 | 42.0 | 89.3 | 48.7 | 6.80 |
| DPoser: their c10, RSGM budget, DPoser method t3 | 86.2 | 102.7 | 107.5 | 43.9 | 85.0 | 48.9 | 6.06 |
| RSGM 512x4: their full, inpaint t16 | 90.2 | 113.8 | 120.1 | 44.6 | 92.2 | 49.3 | 7.16 |
| RSGM 512x4: their c10, inpaint t52 | 91.3 | 111.3 | 113.7 | 45.0 | 90.3 | 49.5 | 6.93 |
| DPoser: their full, DPoser method | 79.6 | 101.3 | 112.2 | 42.6 | 83.9 | 50.4 | 5.41 |
| PoseD-Flow: their checkpoint, D-Flow t17 | 84.5 | 100.5 | 111.3 | 39.2 | 83.9 | 50.9 | 5.07 |
| DPoser: their c10, RSGM budget, DPoser method t28 | 94.2 | 109.5 | 113.5 | 47.1 | 91.1 | 51.1 | 7.11 |
| DPoser: their c10, RSGM budget, DPoser method | 82.5 | 101.4 | 110.6 | 44.2 | 84.7 | 51.3 | 5.61 |
| RSGM 512x4: their c10, Heun EDM 50 NFE | 91.4 | 108.8 | 118.0 | 43.8 | 90.5 | 51.7 | 6.21 |
| PoseD-Flow: their checkpoint | 87.1 | 102.0 | 111.0 | 38.6 | 84.7 | 52.8 | 4.68 |
| RSGM 512x4: their c10, inverted start | 90.3 | 104.8 | 118.6 | 43.1 | 89.2 | 53.0 | 5.57 |
| RSGM 512x4: their c10 | 89.8 | 104.1 | 115.9 | 42.5 | 88.1 | 54.1 | 5.12 |
| RSGM 512x4: their full | 88.0 | 103.2 | 117.9 | 42.8 | 88.0 | 54.4 | 5.15 |
| RSGM 512x4: their c10, seed 12 | 88.4 | 101.4 | 119.8 | 41.8 | 87.8 | 54.8 | 5.08 |
| RSGM 512x4: their c10, ODE Euler 50 | 91.5 | 104.3 | 118.4 | 42.6 | 89.2 | 54.9 | 5.17 |
| RSGM 512x4: their c10, seed 11 | 87.4 | 102.2 | 118.5 | 42.7 | 87.7 | 54.9 | 4.99 |
| RSGM 1536x6: their c10 | 92.0 | 106.7 | 123.4 | 43.1 | 91.3 | 55.0 | 5.55 |
| DPoser: their full, inverted start | 101.8 | 116.5 | 156.5 | 55.4 | 107.6 | 62.2 | 7.16 |
| DPoser: their full | 130.0 | 150.3 | 173.3 | 76.7 | 132.6 | 73.1 | 9.32 |
| DPoser: their c10, RSGM budget | 170.6 | 181.9 | 171.9 | 97.0 | 155.3 | 83.7 | 11.81 |

### Points of the fronts (`config/completion_test_points.yaml`)

The points that are also rows of the registry are in the table above. The last column: the validation numbers on 1000
poses recorded with the point.

#### DPoser dposer

| model | left_leg | legs | arms | trunk | mean | best of 10 | APD | valid 1000: mean / best / APD |
|---|---|---|---|---|---|---|---|---|
| DPoser full/dposer_posedflow_s10, dposer_full_method_mean_apd t3 | 82.4 | 102.6 | 108.3 | 42.1 | 83.8 | 48.0 | 5.79 | 83.7 / 48.7 / 5.66 |
| DPoser full/dposer_posedflow_s10, dposer_full_method_mean_apd t47 | 96.5 | 115.1 | 122.0 | 47.9 | 95.4 | 50.3 | 7.96 | 94.7 / 50.0 / 7.82 |
| DPoser full/dposer_posedflow_s10, dposer_full_method_mean_apd t112 | 103.2 | 119.2 | 122.2 | 49.3 | 98.5 | 51.2 | 8.53 | 97.4 / 50.8 / 8.36 |
| DPoser dposer_posedflow_rsgm_budget_s10, dposer_c10_method_mean_apd t68 | 104.6 | 120.2 | 119.7 | 50.5 | 98.7 | 52.2 | 8.58 | 99.1 / 52.7 / 8.52 |
| DPoser dposer_posedflow_rsgm_budget_s10, dposer_c10_method_mean_apd t29 | 77.5 | 95.0 | 106.2 | 40.6 | 79.9 | 52.9 | 4.19 | 80.6 / 53.2 / 4.22 |

#### DPoser inpaint

| model | left_leg | legs | arms | trunk | mean | best of 10 | APD | valid 1000: mean / best / APD |
|---|---|---|---|---|---|---|---|---|
| DPoser full/dposer_posedflow_s10, dposer_full_inpaint_mean_apd t56 | 82.5 | 102.7 | 116.3 | 41.9 | 85.9 | 47.3 | 6.31 | 86.3 / 47.7 / 6.32 |
| DPoser full/dposer_posedflow_s10, dposer_full_inpaint_mean_apd t89 | 83.8 | 106.8 | 124.8 | 43.0 | 89.6 | 47.9 | 7.00 | 90.5 / 48.3 / 7.04 |
| DPoser full/dposer_posedflow_s10, dposer_full_inpaint_mean_apd t49 | 86.4 | 111.4 | 131.7 | 45.1 | 93.6 | 49.2 | 7.45 | 94.6 / 49.9 / 7.51 |
| DPoser dposer_posedflow_rsgm_budget_s10, dposer_c10_inpaint_mean_apd t103 | 93.6 | 115.4 | 136.3 | 48.6 | 98.5 | 51.3 | 7.90 | 100.0 / 53.1 / 8.03 |
| DPoser full/dposer_posedflow_s10, dposer_full_inpaint_mean_apd t38 | 98.5 | 121.1 | 142.7 | 49.9 | 103.1 | 53.8 | 8.45 | 103.8 / 53.8 / 8.52 |

#### PoseD-Flow dflow

| model | left_leg | legs | arms | trunk | mean | best of 10 | APD | valid 1000: mean / best / APD |
|---|---|---|---|---|---|---|---|---|
| PoseD-Flow their checkpoint, posedflow_dflow_guess_mean_apd t19 | 94.2 | 113.4 | 124.1 | 41.6 | 93.3 | 50.0 | 6.96 | 94.8 / 51.5 / 7.01 |
| PoseD-Flow their checkpoint, posedflow_dflow_guess_mean_apd t4 | 90.9 | 108.5 | 121.9 | 40.0 | 90.3 | 50.2 | 6.40 | 92.0 / 51.2 / 6.52 |
| PoseD-Flow their checkpoint, posedflow_dflow_guess_mean_apd t12 | 88.0 | 104.1 | 115.3 | 40.0 | 86.9 | 50.4 | 5.50 | 88.4 / 52.6 / 5.53 |
| PoseD-Flow their checkpoint, posedflow_dflow_guess_mean_apd t8 | 84.1 | 101.0 | 112.7 | 38.7 | 84.1 | 52.3 | 4.69 | 85.9 / 54.4 / 4.76 |
| PoseD-Flow their checkpoint, posedflow_dflow_guess_mean_apd t14 | 81.9 | 97.6 | 110.4 | 37.9 | 82.0 | 54.9 | 3.95 | 84.2 / 56.8 / 4.01 |
| PoseD-Flow their checkpoint, posedflow_dflow_mean_apd t11 | 105.5 | 128.5 | 135.7 | 45.2 | 103.7 | 55.4 | 7.96 | 104.7 / 56.6 / 8.00 |

#### RSGM dflow

| model | left_leg | legs | arms | trunk | mean | best of 10 | APD | valid 1000: mean / best / APD |
|---|---|---|---|---|---|---|---|---|
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_dflow_mean_apd_full_dposer_like t51 | 93.5 | 112.5 | 123.6 | 44.8 | 93.6 | 49.7 | 6.94 | 96.3 / 51.2 / 7.26 |
| RSGM full/rsgm_posedflow_dposer_like_aniso_s10, rsgm_dflow_mean_apd_full_dposer_like_aniso t19 | 96.0 | 116.1 | 124.3 | 47.3 | 95.9 | 51.0 | 7.49 | 98.4 / 51.1 / 7.74 |
| RSGM full/rsgm_posedflow_dposer_like_aniso_s10, rsgm_dflow_mean_apd_full_dposer_like_aniso t25 | 92.1 | 106.4 | 119.6 | 45.4 | 90.9 | 51.0 | 6.37 | 93.5 / 51.5 / 6.61 |
| RSGM full/rsgm_posedflow_dposer_like_aniso_s10, rsgm_dflow_mean_apd_full_dposer_like_aniso t36 | 101.4 | 116.7 | 124.8 | 48.1 | 97.8 | 51.1 | 7.81 | 100.5 / 51.8 / 8.06 |
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_dflow_mean_apd_full_dposer_like t45 | 103.4 | 124.4 | 131.3 | 48.8 | 102.0 | 51.8 | 8.25 | 103.8 / 53.0 / 8.50 |
| RSGM full/rsgm_posedflow_dposer_like_aniso_s10, rsgm_dflow_mean_apd_full_dposer_like_aniso t58 | 89.5 | 103.9 | 117.2 | 45.2 | 88.9 | 51.8 | 5.96 | 91.1 / 51.3 / 6.18 |
| RSGM full/rsgm_posedflow_dposer_like_aniso_s10, rsgm_dflow_mean_apd_full_dposer_like_aniso t51 | 86.5 | 101.5 | 114.1 | 44.8 | 86.7 | 53.1 | 5.26 | 89.7 / 53.0 / 5.57 |
| RSGM full/rsgm_posedflow_dposer_like_aniso_s10, rsgm_dflow_mean_apd_full_dposer_like_aniso t29 | 86.3 | 98.5 | 111.3 | 44.5 | 85.2 | 56.3 | 4.30 | 87.7 / 56.3 / 4.56 |

#### RSGM dposer

| model | left_leg | legs | arms | trunk | mean | best of 10 | APD | valid 1000: mean / best / APD |
|---|---|---|---|---|---|---|---|---|
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_sds_mean_apd_full_dposer_like t82 | 104.6 | 126.5 | 131.7 | 43.2 | 101.5 | 56.5 | 7.58 | 101.0 / 55.8 / 7.51 |
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_sds_mean_apd_full_dposer_like t106 | 103.0 | 124.0 | 129.1 | 43.1 | 99.8 | 56.8 | 7.21 | 99.0 / 55.9 / 7.16 |
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_sds_mean_apd_full_dposer_like t63 | 100.1 | 121.9 | 126.8 | 42.1 | 97.7 | 56.9 | 6.66 | 97.1 / 56.7 / 6.55 |
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_sds_mean_apd_full_dposer_like t22 | 108.7 | 131.8 | 137.4 | 44.1 | 105.5 | 57.1 | 8.27 | 104.3 / 56.2 / 8.18 |
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_sds_mean_apd_full_dposer_like t56 | 111.3 | 135.9 | 141.2 | 44.7 | 108.3 | 57.5 | 8.73 | 106.4 / 56.7 / 8.51 |
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_sds_mean_apd_full_dposer_like t64 | 98.2 | 118.9 | 125.1 | 41.8 | 96.0 | 57.6 | 6.16 | 95.5 / 57.2 / 6.10 |
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_sds_mean_apd_full_dposer_like t99 | 97.0 | 118.0 | 124.1 | 41.4 | 95.1 | 57.6 | 5.83 | 94.8 / 57.9 / 5.79 |
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_sds_mean_apd_full_dposer_like t15 | 95.2 | 118.8 | 123.5 | 40.4 | 94.5 | 59.3 | 5.51 | 92.0 / 61.7 / 4.62 |
| RSGM full/rsgm_posedflow_dposer_like_s10, rsgm_sds_mean_apd_full_dposer_like t24 | 95.3 | 118.0 | 122.2 | 40.7 | 94.1 | 59.5 | 5.22 | 93.7 / 60.0 / 5.20 |

#### RSGM inpaint

| model | left_leg | legs | arms | trunk | mean | best of 10 | APD | valid 1000: mean / best / APD |
|---|---|---|---|---|---|---|---|---|
| RSGM rsgm_posedflow_abl_residual_s10, rsgm_inpaint_mean_apd_residual t59 | 86.7 | 106.0 | 113.8 | 41.9 | 87.1 | 48.0 | 6.39 | 88.2 / 49.3 / 6.39 |
| RSGM full/rsgm_posedflow_small_s10, rsgm_inpaint_mean_apd_full_small t63 | 90.1 | 114.5 | 126.2 | 45.1 | 94.0 | 49.4 | 7.35 | 95.6 / 50.5 / 7.52 |
| RSGM rsgm_posedflow_small_s10, rsgm_inpaint_pareto t56 | 91.2 | 111.6 | 114.0 | 45.0 | 90.5 | 49.5 | 6.95 | 92.5 / 50.8 / 7.00 |
| RSGM full/rsgm_posedflow_small_s10, rsgm_inpaint_mean_apd_full_small t26 | 94.5 | 118.3 | 133.2 | 47.7 | 98.4 | 50.7 | 8.02 | 99.5 / 52.0 / 8.03 |
| RSGM full/rsgm_posedflow_small_s10, rsgm_inpaint_mean_apd_full_small t105 | 101.2 | 124.6 | 141.6 | 46.3 | 103.4 | 52.9 | 8.44 | 103.7 / 53.1 / 8.52 |
