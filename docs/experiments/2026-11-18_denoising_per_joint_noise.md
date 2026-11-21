# Per-joint noise does not help motion denoising (validation)

Per-joint noise in training (`joint_scale`) improves pose completion through the score at small noise. Section 10
of [2026-11-14_score_network_architectures.md](2026-11-14_score_network_architectures.md) covers it. Does it also
improve motion denoising, where the prior enters through score distillation (DPoser's optimization, SDS)?

## Summary

- **No.** Tuned on validation by the same Optuna procedure as the isotropic model, `abl_residual_aniso` reaches
  MPJPE 19.47 mm at 40 mm noise and 32.68 mm at 100 mm noise. Its isotropic pair `abl_residual` reaches 18.63 /
  30.83 mm at the isotropic settings. The rule fixed before the studies (aniso better if below 18.63 / 30.83) does not
  hold at either noise level: aniso is 0.8 and 1.9 mm worse.
- **A bug fixed on the way.** The SDS of motion denoising drew the IGSO(3) noise with one Brownian time s(t) for all
  joints, so an aniso model got noise it was never trained on. With the isotropic settings unchanged it gave 29.2 /
  41.1 mm. `dposer_denoise` now uses `joint_s(t)`, which is s(t) for an isotropic model, so the isotropic rows are
  unchanged. It also draws the noise on cuda.
- **Why, most likely.** The noise of this task is isotropic in joint positions (±40 or ±100 mm on every joint). For
  joints with short bones (spine, neck, collars) that is a large error in angle. The aniso prior probes exactly these
  joints with the smallest noise: c_j is 1.0-10.0 for the spine, neck, head, collars and ankles, against 19-43 for
  the hips, knees, shoulders, elbows and wrists. Moving the global noise
  window, which is all the tuning can do, does not change how the noise is spread over the joints.
- **For comparison, on validation:** the isotropic RSGM rows (18.6-18.7 / 30.8-31.0 mm) are ahead of DPoser full
  (19.19 / 32.23) and PoseD-Flow (19.83 / 31.73). Their settings were tuned on the same validation sequences, so this
  is optimistic until the test.
- **Decision.** Motion denoising keeps the isotropic RSGM. By the rule of `rsgm_sds_40mm.yaml`, the RSGM row with the
  lowest validation MPJPE at the tuned settings is ours: `abl_residual`. Per-joint noise stays a completion feature.

## Setup

- Protocol: motion denoising of PoseD-Flow (Tab. 3; `hp/evaluation/denoising.py`): 60-frame chunks, Gaussian noise of
  40 or 100 mm on the 22 body joints of SMPL-X neutral, MPJPE of the 22 joints (and MPVPE). Validation: 100 chunks of
  HDM05/SFU/MoSh (`sequences("valid")`); the test (190 HumanEva chunks) was not used.
- Procedure: DPoser's optimization with RSGM as the prior on SO(3)^J (`hp.models.motion_denoising.dposer_denoise`):
  3 iterations of `steps_per_iter` Adam steps on the axis-angle poses. At each step the current pose is noised by
  IGSO(3) of Brownian time s_j(t), with t going down from t_max to t_min, denoised by one Tweedie step of RSGM, and
  pulled toward it with weight `sds_weight`. A data term and a temporal term complete the objective.
- Models (c10, 30k x 4096, timeres 512 x 2, seed 10). Isotropic: `abl_norm`, `abl_residual`
  (`~/data/experiments/rsgm_posedflow_abl_{norm,residual}_s10`). Per-joint noise: `abl_norm_aniso`,
  `abl_residual_aniso` (`~/data/experiments/igso_speed/rsgm_posedflow_abl_{norm,residual}_aniso_s10`), with
  c_j from 1.0 (a spine joint) to 42.8 (right elbow) and a geometric mean of 12.69 (`model.json`, `joint_scale`).
  Reference: RSGM 512x4 c10, whose settings were tuned (`rsgm_sds_40mm.yaml`, `rsgm_sds_100mm.yaml`, 40 trials each).
- Isotropic settings (best trials of those studies): 40 mm t_max 0.0595, t_min 0.387 t_max, weight 7.75, lr 0.0413,
  3 x 67 steps; 100 mm t_max 0.0515, t_min 0.881 t_max, weight 4.38, lr 0.0422, 3 x 133 steps.
- Mapped t for aniso models: s(t') = s(t) / k. With "t geo" (k = 12.69), a joint with the geometric-mean c_j gets the
  noise the isotropic model got. With "t max" (k = 42.79), no joint gets more noise than before.
- Studies `rsgm_sds_aniso_{40,100}mm` (`config/denoising_tuning/`): `abl_residual_aniso`, the space of the isotropic
  studies with the t_max range moved down (0.003-0.1 at 40 mm, 0.005-0.15 at 100 mm), 30 trials (TPE), first trial
  the isotropic best with t geo. Run on workstation2 (cuda, GPUs shared with other users): about 6 and 11 min per
  trial. The rule is written in the files before the runs.

## Results

Validation, 100 chunks, MPJPE / MPVPE (mm). The input MPJPE is 63.7 mm at 40 mm noise and 159.3 mm at 100 mm noise.

| row | 40 mm | 100 mm |
|---|---|---|
| RSGM 512x4 c10, isotropic settings (tuned on it) | 18.71 / 38.19 | 30.87 / 51.43 |
| abl_norm, isotropic settings | 18.71 / 38.05 | 30.99 / 51.27 |
| abl_residual, isotropic settings | **18.63** / 38.00 | **30.83** / 51.31 |
| abl_norm_aniso, isotropic settings | 29.36 / 55.23 | 41.31 / 69.80 |
| abl_residual_aniso, isotropic settings | 29.18 / 54.93 | 41.11 / 69.55 |
| abl_norm_aniso, t geo | 19.39 / 39.05 | 32.96 / 53.75 |
| abl_norm_aniso, t max | 19.80 / 41.30 | 37.32 / 62.00 |
| abl_residual_aniso, t geo | 19.44 / 39.12 | 32.81 / 53.58 |
| abl_residual_aniso, tuned (30 trials) | 19.47 / 39.09 (trial 0, i.e. t geo) | 32.68 / 53.16 (trial 22) |
| DPoser full, its own method | 19.19 / 38.15 | 32.23 / 54.38 |
| PoseD-Flow, their checkpoint | 19.83 / 37.19 | 31.73 / 56.89 |

The rows with "isotropic settings" and the per-joint rows ran on the Mac (mps). The tuned rows ran on workstation2
(cuda), where trial 0, t geo, gave 19.47 at 40 mm against 19.44 on the Mac.

- At 40 mm the tuning found nothing better than its first point: the 5 best trials are 19.47, 19.50, 19.62, 19.70 and
  19.81 mm. At 100 mm it gained 0.1 mm over t geo (32.68 against 32.82, trial 22: t_max 0.0170, t_min 0.394 t_max,
  weight 7.53, lr 0.0378, 3 x 133 steps).
- t geo is the right scale: t max (no joint noised more than in the isotropic model) is worse, especially at 100 mm.
  The aniso prior needs the larger noise on the wide joints. It is the small noise on the narrow joints that fails
  here.
- Among the isotropic networks the differences are at most 0.16 mm, and residual is marginally best.

## Conclusions

- Per-joint noise is a feature for completion, where hidden joints are generated, and not for denoising, where every
  joint is corrupted by the same noise in position. Motion denoising keeps the isotropic RSGM. The candidate row for
  the test is `abl_residual` at the tuned isotropic settings: 18.63 / 30.83 mm on validation, against DPoser 19.19 /
  32.23 and PoseD-Flow 19.83 / 31.73.
- The SDS of motion denoising now follows the model's per-joint noise. Without that fix any comparison with aniso
  models is meaningless (29 against 19 mm).

## What it does not show

- Validation only. The isotropic settings were tuned on the same 100 chunks, so all RSGM numbers here are optimistic.
- One seed per model. The differences among the isotropic networks (0.1-0.2 mm) are within the spread one would
  expect between seeds.
- Only a global rescaling of the noise was tried. A prior whose per-joint scale matches the noise of the task (e.g.
  c_j by bone length, or a mix of isotropic and per-joint noise) was not.
- The aniso networks are c10 and 1.2M parameters. The full-data `dposer_like_aniso` was not tried in denoising.

## Where things are

- Code: `src/hp/models/motion_denoising.py` (`dposer_denoise`, per-joint noise since 69c5550),
  `scripts/evaluate_denoising.py`, `scripts/tune_denoising.py`.
- Rows of the table: `explore/evaluation/denoising_aniso/rows.py`, cache
  `~/data/experiments/denoising/cache_aniso/` (Mac). DPoser and PoseD-Flow from `~/data/experiments/denoising/cache/`.
- Studies: `~/data/experiments/denoising/optuna/rsgm_sds_aniso_{40,100}mm.{db,json}` (Mac and workstation2), configs
  `config/denoising_tuning/rsgm_sds_aniso_{40,100}mm.yaml`; the isotropic ones `rsgm_sds_{40,100}mm`.
- Models: see Setup; the aniso runs are on the Mac, workstation1 and workstation2 (`igso_speed`).
