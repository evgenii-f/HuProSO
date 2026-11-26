# ACE energy: an interpretable energy of a pose as a prior on SO(3)^J

The line of work on branch `experimental/ace-descriptors` (worktree `HuProSO-ace`). This file is its one report in the
sense of `docs/experiments/README.md`: the question, where the line stands, and one paragraph per phase. Each phase has
its own dated file here with the setup, the tables and what it does not show. Validation numbers only; no test was
run in this line.

## Question

Can an energy built from a fixed basis of pose features (ACE-like: functions of joints, of pairs and of the whole
body), with the score $s_t = -\nabla E_t$, replace or help the score network of RSGM? Compared with the network
`fact_RDW` s10 on the same validation poses: the loss of RSGM, generation (the harness of `generate.py`: Wasserstein,
recall, precision in degrees per joint) and pose completion (inpainting, SDS, no-DPS; MPVPE in mm, K = 10).

## Where the line stands (2026-11-26)

| model | file (`~/data/experiments/ace/`) | val. loss | Wasserstein (°) | inpainting mean (mm) |
|---|---|---|---|---|
| network `fact_RDW` s10 | – | 5.951 | 14.86 | 85.23 |
| **MLP embedding K = 8 on P+L** | `energy_mlp_pl_k8.pt` | **6.090** | **15.31** | **95.71** |
| P+L: products + joint limits, ridge 1e-3 (linear, closed form) | `energy_b_rich_products_jd_r0.001.pt` | 6.336 | 15.93 | 97.49 |
| first linear energy (descriptors + one-body terms) | – | 7.23 | 18.24 | 128.16 |

Rows 1-3 are from one harness run (1000 samples) and one completion run (200 poses), 2026-11-26; row 4 from
2026-11-23/24.

- **Current model:** the MLP embedding on P+L. It is a success in generation; in completion it is better than P+L by
  1.8 mm, below the 2 mm rule of its phase.
- **As a judge of hypotheses** the energy is strong: the network's 10 hypotheses ranked by P+L give 79.32 mm, the MLP
  energy's own ranked by itself 82.44 mm.
- **Next:** a joint Optuna study of the embedding and the basis (to be announced before it starts), chosen by
  Wasserstein on validation; then completion of the winner once. The code was made faster for it first.
- **Deferred:** mixing the scores of the energy and the network (end of the line); changes of the model that make it
  smaller or faster (`explore/ace_energy/OPTIMIZATION.md`).

## Phases

| date | report | in one line |
|---|---|---|
| 2026-11-23 | [Descriptors and the linear energy](2026-11-23_descriptors_and_linear_energy.md) | Negative as an input of the score network (`ace_RDW` inpainting 89.36 against 86.64 mm); as a linear energy fitted in closed form 94% of the network's loss gain; 2026-11-24: harmonics of SO(3) add little, first MLP embedding 6.54, completion 128.16 mm, a rare bent knee dominates the gap |
| 2026-11-25 | [Products of local rotations and joint limits](2026-11-25_products_and_joint_limits.md) | Products of all pairs of local rotations plus every joint's noised density: Wasserstein 16.04° against 14.93°; ridge, bins, triples and descriptors change it by less than the noise: a plateau |
| 2026-11-26 | [Completion and the plateau](2026-11-26_completion_and_plateau.md) | Inpainting 97.49 against 85.23 mm, the products carry it; the gap is in poses far from train and in bent knees; many-body terms and 5× data do not close it |
| 2026-11-26 | [Ideas from ACE for the network](2026-11-26_network_ideas.md) | Mixing the scores lowers the loss like a second network, the harness is unchanged (deferred); a cluster block on `fact_RDW` fails its gate (closed) |
| 2026-11-26 | [MLP embedding on P+L](2026-11-26_mlp_embedding.md) | Loss 6.090, Wasserstein 15.31° against 15.93° (network 14.86°); a linear control trained alike does not move; completion −1.8 mm, below the rule |
| 2026-11-26 | [Speed of the score](speed.md) | Exact speed-ups: on the GPU the MLP score 30 → 7 ms, a training step 52 → 26 ms, DPS 59 → 32 ms (500 poses); DPS is what is left |

## Decisions

- 2026-11-23: descriptors as an input of the network closed; the energy line continues.
- 2026-11-25: the products of local rotations and the joint densities are the basis (P+L); more linear features stop.
- 2026-11-26: the linear family is on a plateau; a nonlinear embedding is the next lever. The cluster block is closed;
  mixing with the network is deferred.
- 2026-11-26: the MLP embedding on P+L is the model of the line (user's decision: generation counts as the success).
- 2026-11-26: exact speed-ups first, then the joint study; changes of the model (descriptors out, fewer centres) not
  yet.

## Where things are

- **Code:** `explore/ace_energy/` — `fit.py` (the linear fit, the basis `WithHarmonics`), `joint_density.py`,
  `embedding.py` (the embeddings and their training), `inspect_energy.py` (`load_energy`), `generate.py` (the harness),
  `complete.py` (completion), `diagnose.py` (paired comparisons and the gap), `bench.py` (speed),
  `OPTIMIZATION.md` (what the basis could lose); features in `hp.models.ace_features`, harmonics in
  `hp.models.so3_harmonics`.
- **Runs:** `~/data/experiments/ace/` (fits `energy_*.pt` with their `.json`, logs, completion caches `diag*/`,
  samples); each phase file lists its own.
- **New phases:** a new dated file in this folder, a row in the table above, and an update of "Where the line stands".
  The index `docs/experiments/README.md` links only this file.
