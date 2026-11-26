# ACE energy, 2026-11-26: tuning inpainting does not help; the energy knows good legs, the sampler misses them

Branch `experimental/ace-descriptors`. Validation numbers only; no test was run. Part of the ACE line: [overview](README.md).

## Summary

- **Question 1.** Is the completion gap of the linear energy P+L (97.49 against 85.23 mm, [completion and
  plateau](2026-11-26_completion_and_plateau.md)) partly due to settings tuned for the network? **No.** An Optuna study of
  inpainting with P+L (20 trials, 500 validation poses) finds 98.43 mm at its best point against 98.73 mm at t81, the
  network's point: −0.3 mm, far below the 2 mm of the rule. The check on the other 500 poses was not run (below).
- **Question 2.** On hard poses (pose 189, a crouch in a boxing guard with the legs hidden) the energies give the legs
  splits and side kicks; the MLP embedding does not fix it. Is the energy wrong there, or does the sampler fail to reach
  what the energy prefers? Measured: the energy of the true pose against the energies of the 30 hypotheses (network,
  P+L, MLP) on the 200 poses of the completion runs.
- **At s ≥ 0.02 the energies prefer the network's hypotheses to their own.** For the MLP energy, legs hidden, s = 0.02:
  0.50 of its own hypotheses lie below the truth, 0.65 of the network's; on the worst 20% of poses 0.54 against 0.74.
  Pose 189: the truth and the network's legs 2–20 nats below the MLP's splits at s = 0.02 … 0.2. The energy tells the
  good legs from the bad; its own sampler does not reach the good ones. This is the ranking result of the plateau
  phase seen from the other side (the network's hypotheses chosen by P+L: 79.32 mm).
- **At s = 0.002 (nearly clean poses) it does not.** On the worst 20% of poses 0.72–0.92 of the hypotheses of both rows
  lie below the truth, the energy's own as many as the network's (MLP, legs: 0.79 and 0.77; P+L, legs: 0.84 and 0.70).
  Pose 189 is the extreme: the MLP's splits are 10–40 nats below the truth and below the network's legs.
- **Next step proposed, not run:** the energy and the network in two noise bands (`complete.py --band`) to find at which
  noise levels the wrong legs are made; then the model or the method accordingly.

## Setup

- **Prior:** P+L, the linear fit `energy_b_rich_products_jd_r0.001.pt` (products of local rotations + joint densities,
  ridge 1e-3), in place of the score of `fact_RDW` s10 (`~/data/experiments/factorial/rsgm_posedflow_fact_RDW_s10/run_0001`),
  whose noising and sampler it uses; the MLP embedding K = 8 on P+L (`energy_mlp_pl_k8.pt`,
  [MLP embedding](2026-11-26_mlp_embedding.md)) in part 2.
- **Study** (`config/completion_tuning/ace_pl_inpaint_mean_apd.yaml`, `scripts/tune_completion.py`, commit 40970e1):
  inpainting, objectives mean MPVPE (min) and APD (max), K = 10; steps {100, 200, 400}, guidance 0.01–0.5 (log),
  resample {1, 2, 3, 5}, replace {true, false}, schedule edm; 20 trials, 8 random before TPE, the first three enqueued
  (t81 and two guidance probes of P+L); the first 500 poses of the 1000-pose validation draw (`valid_poses`, nested: a
  later study of another model on 500 poses uses the same poses). workstation2, one process on one GPU, 3 h 10 min (t81:
  27 min; the speed-ups of [speed](speed.md)).
- **Rule, written before the study:** among the points of the front with APD ≥ 6.40 cm the lowest mean; then that point
  on validation poses 500–999 next to P+L at t81 and the network at t81; a gain counts if the mean is lower than P+L at
  t81 by at least 2 mm with a paired 95% interval excluding 0.
- **Part 2** (`explore/ace_energy/gt_energy.py`, on the Mac): the hypotheses of three completion runs of the plateau and
  MLP phases (first 200 validation poses, K = 10, inpainting at t81, four occlusions): the network
  (`diag2/inpaint_network_rank.pt`), P+L (`diag2/inpaint_products.pt`), MLP (`diag5/inpaint_mlp_pl_k8.pt`). Each energy
  (P+L, MLP) gives E_t in nats of the truth and of the 30 hypotheses at s = 0.002, 0.02, 0.05, 0.2, 0.5, 1, evaluated at
  the clean poses. The visible joints of every hypothesis are exactly the truth's, so the comparison is about the
  hidden joints only.

## Results: the study

Validation poses 0–499, K = 10, MPVPE in mm, APD in cm (`tune_completion.py --study ace_pl_inpaint_mean_apd --summary`).
The six best of 20 by the mean:

| trial | steps | guidance | resample | replace | mean | best of K | APD |
|---|---|---|---|---|---|---|---|
| 7 | 400 | 0.026 | 5 | true | **98.43** | 55.07 | 7.24 |
| 13 | 200 | 0.087 | 3 | true | 98.59 | 55.76 | 7.10 |
| 0 (t81) | 400 | 0.049 | 5 | false | 98.73 | 54.53 | 7.43 |
| 1 | 400 | 0.02 | 5 | false | 100.07 | 54.82 | 7.63 |
| 14 | 100 | 0.287 | 3 | true | 101.04 | 60.18 | 6.63 |
| 2 | 200 | 0.1 | 2 | false | 101.48 | 54.81 | 7.93 |

- The rule picks trial 7: 0.3 mm below t81. The rest of the space is worse: resample 1 or a guidance below 0.02 gives
  106–133 mm.
- **The check on poses 500–999 was not run**: a gain of 2 mm there is not plausible when the
  gain is 0.3 mm on the poses that chose the point, since a choice on the same poses overstates its gain.
- The 500 poses include the 200 of the earlier runs; t81 here is 98.73 mm against 97.49 mm on the first 200.

## Results: the energy of the truth

The share of hypotheses whose energy lies below the truth's, mean over poses; "worst 20%": the 40 poses where the
energy's own row has the highest mean error (`gt_energy.py --shares`, `gt_energy_shares.log`). With good samples of
a calibrated model it is near 0.5; the network's samples, the better ones, are the reference. Legs hidden:

| energy | s | own below truth | worst 20% | network's below truth | worst 20% |
|---|---|---|---|---|---|
| P+L | 0.002 | 0.70 | 0.84 | 0.67 | 0.70 |
| P+L | 0.02 | 0.59 | 0.60 | 0.65 | 0.74 |
| P+L | 0.2 | 0.55 | 0.55 | 0.60 | 0.63 |
| MLP | 0.002 | 0.60 | 0.79 | 0.67 | 0.77 |
| MLP | 0.02 | 0.50 | 0.54 | 0.65 | 0.74 |
| MLP | 0.2 | 0.48 | 0.47 | 0.61 | 0.67 |
| MLP | 1 | 0.47 | 0.47 | 0.53 | 0.58 |

- At s ≥ 0.02 fewer of the energy's own hypotheses than of the network's lie below the truth, and the difference is
  larger on the worst poses (MLP: 0.54 against 0.74 at s = 0.02). The other occlusions agree (left leg, arms, trunk:
  own 0.45–0.60, network 0.50–0.73, all poses).
- At s = 0.002 both rows lie mostly below the truth on the worst poses (0.72–0.92 over the occlusions). The worst poses
  are far from the training data, so the truth itself is improbable there; the energy does not separate the good
  hypotheses from the bad at this level. P+L on legs prefers its own (0.84 against 0.70), the MLP does not (0.79 against
  0.77).
- In the 30 hypotheses, the truth is lowest in 0–3% of the poses, and lower than all of the energy's own in 3–12%.

Pose 189, legs hidden, E_t in nats (MLP energy; `gt_energy.log`, `gt_energy_high.log`); the MLP's own hypotheses have
errors of 366–567 mm, the network's 108–209 mm:

| s | truth | network's hypotheses | MLP's hypotheses |
|---|---|---|---|
| 0.002 | −542.3 | −540.8 … −556.4 | −551.4 … −581.2 |
| 0.02 | −149.7 | −144.3 … −153.5 | −130.2 … −147.6 |
| 0.05 | −118.6 | −113.2 … −120.0 | −101.2 … −112.8 |
| 0.2 | −156.9 | −153.7 … −157.9 | −148.5 … −153.2 |

P+L gives the same picture (s = 0.002: truth −450.8, its own −458.6 … −502.2; s = 0.02: truth −120.9, its own −103.5 …
−120.5).

## Conclusions

1. **Tuning inpainting for the energy is closed.** t81 is as good as anything in the space; the gap to the network is in
   the energy, not in the settings.
2. **The energy is a better judge than a generator, also on hard poses.** At s ≥ 0.02 it places the network's legs
   below its own wrong ones. The splits are not what the energy prefers at these levels; the sampler ends there anyway.
3. **At the clean end (s = 0.002) the energy does not discriminate on hard poses**, and on some (pose 189) it prefers
   splits strongly. Whether the sampler is drawn there at the end, or the wrong legs are made earlier by the score at
   noised points (not measured here), is open. The user's reading of the figures, that the observed arms move the legs
   too little, agrees with the weak couplings of distant joints of the plateau phase (ratio 0.55–0.60 of the network's
   at tree distance 10), but this test does not isolate it.
4. **Next, to be discussed:** two runs of `complete.py --band` with the MLP energy, legs only, 200 poses: the energy at
   s ≥ 0.02 with the network below, and the reverse. If the splits vanish when the network takes the low levels, the
   clean end is the problem (training at small s, or stopping the energy near s = 0.02); if not, the score at high noise
   (inputs of the embedding that see the whole body, for the joint study). A network to propose and an energy to choose
   stays the strongest combination seen.

## What it does not show

- Energies of clean poses only, not of the noised points the sampler visits, and not the volume of a region: a broad
  region of slightly higher energy can hold more mass than a narrow low one.
- One run per row, 200 poses in part 2; the worst 20% are chosen by the error of the same rows.
- Part 1: one prior (P+L), one objective, 20 trials; the MLP energy was not tuned.
- The hypotheses of the three rows come from runs with different batch sizes (network 200, energies 100); the
  sampler's noise differs.

## Where things are

- Study: `~/data/experiments/completion/optuna/ace_pl_inpaint_mean_apd.{db,json}`, log
  `ace_pl_inpaint_mean_apd_gpu1.log` (copied from workstation2); config
  `config/completion_tuning/ace_pl_inpaint_mean_apd.yaml`.
- Part 2: `explore/ace_energy/gt_energy.py`; `~/data/experiments/ace/gt_energy.{pt,log}` (s = 0.002, 0.02),
  `gt_energy_high.{pt,log}` (s = 0.05 … 1), `gt_energy_shares.log`.
- The figure of pose 189: `explore/evaluation/pose_completion.ipynb`, section 6.
