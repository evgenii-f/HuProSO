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
  energy's own ranked by itself 82.44 mm. On hard poses too: at s ≥ 0.02 the energies place the network's legs below
  their own wrong ones; at s = 0.002 they do not separate them.
- **Completion settings** are not the gap: tuning inpainting for P+L gains 0.3 mm over t81 (20 trials, 500 poses).
- **Small noise:** far from the data the noise levels of the energies disagree by hundreds of nats; a Fokker-Planck
  term makes them agree but leaves the holes and makes completion 7 mm worse (closed in that form).
- **2026-11-27, where the gap is made:** with the energy only at s < 0.04 (the network above) inpainting keeps 92% of the gap, at s ≥ 0.04 the linear energy is as good as the network (+0.2 mm); a compact pairwise matrix Fisher basis keeps generation, not completion ([compact basis](2026-11-27_compact_fisher_basis.md)).
- **2026-11-27, the Fisher energy without noise** (branch `experimental/fisher-energy`): the pairwise matrix Fisher
  energy trained by maximum likelihood has no holes (its Gibbs completions sit at the energy of the truth) and is the
  best judge of the network's hypotheses (76.43 mm against 81.10 for P+L), but as a completer it is at 112.87 mm and
  as a generator at 21.52°; one-body D^{2,3} terms help generation only ([Fisher, ML and Gibbs](2026-11-27_fisher_boltzmann.md)).
  Next there, to be discussed: pair terms of higher degree.
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
| 2026-11-26 | [Tuning inpainting; the energy of the truth](2026-11-26_completion_tuning_and_truth_energy.md) | Optuna on inpainting with P+L: 98.43 against 98.73 mm at t81, closed; the energies prefer the network's hypotheses to their own at s ≥ 0.02, not at s = 0.002; splits on hard poses come from the sampler or the clean end, to be told apart by noise bands |
| 2026-11-27 | [A Fokker-Planck term](2026-11-27_fokker_planck.md) | The levels of s disagree far from the data by hundreds of nats; the heat equation as a training term cuts that 3–5 times at the same loss, but the holes stay and inpainting is +7.1 mm worse than without it: negative |
| 2026-11-27 | [A compact pairwise matrix Fisher basis](2026-11-27_compact_fisher_basis.md) | Products + D^{1,2,3}(x_j) + limits without descriptors (15,428 features) keep generation (16.14° against 16.04°) at half the score time, but inpainting is +1.1 mm (best of 10 +1.7 mm, fails); D^1 alone fails (18.81°); noise bands: 92% of the completion gap is made at s < 0.04; hard joint limits, estimated from the completed poses (not built), would be worth ≤ 1.5 mm |
| 2026-11-27 | [The Fisher energy by likelihood, sampled by Gibbs](2026-11-27_fisher_boltzmann.md) | Pseudo-likelihood ranks the network's hypotheses (77.83 mm) but has a hole in the arms (Gibbs inpainting 166.75 mm); maximum likelihood (PCD) closes it: inpainting 114.10 mm, Wasserstein 23.13°; one-body D^{2,3} with converged chains (3000 sweeps): 21.52°, inpainting 112.87 mm (−1.23, n.s.), ranking 76.43 mm, the best of the line; refinement of the network's hypotheses and a mixture of linear energies fail their rules |
| 2026-11-26 | [Speed of the score](speed.md) | Exact speed-ups: on the GPU the MLP score 30 → 7 ms, a training step 52 → 26 ms, DPS 59 → 32 ms (500 poses); DPS is what is left |

## Decisions

- 2026-11-23: descriptors as an input of the network closed; the energy line continues.
- 2026-11-25: the products of local rotations and the joint densities are the basis (P+L); more linear features stop.
- 2026-11-26: the linear family is on a plateau; a nonlinear embedding is the next lever. The cluster block is closed;
  mixing with the network is deferred.
- 2026-11-26: the MLP embedding on P+L is the model of the line (user's decision: generation counts as the success).
- 2026-11-26: exact speed-ups first, then the joint study; changes of the model (descriptors out, fewer centres) not
  yet.
- 2026-11-26: tuning completion settings for the energy is closed (t81 kept); the check on poses 500–999 not run.
- 2026-11-27: the Fokker-Planck term at isotropic points is closed (completion +7.1 mm); ideas for the linear model
  next.
- 2026-11-27: no mixture of linear energies (the probe fails its rule); mixtures, ensembles and products with the
  network deferred until the Fisher model is exhausted (user's decision). The Fisher energy is trained by maximum
  likelihood; refinement of the network's hypotheses by its conditionals is closed.

## Where things are

- **Code:** `explore/ace_energy/` — `fit.py` (the linear fit, the basis `WithHarmonics`), `joint_density.py`,
  `embedding.py` (the embeddings and their training), `inspect_energy.py` (`load_energy`), `generate.py` (the harness),
  `complete.py` (completion), `diagnose.py` (paired comparisons and the gap), `gt_energy.py` (the energy of the truth against
  the hypotheses), `fp_residual.py` and `fokker_planck.py` (the heat equation between the levels), `bench.py` (speed),
  `OPTIMIZATION.md` (what the basis could lose); features in `hp.models.ace_features`, harmonics in
  `hp.models.so3_harmonics`.
- **Runs:** `~/data/experiments/ace/` (fits `energy_*.pt` with their `.json`, logs, completion caches `diag*/`,
  samples); each phase file lists its own.
- **New phases:** a new dated file in this folder, a row in the table above, and an update of "Where the line stands".
  The index `docs/experiments/README.md` links only this file.
