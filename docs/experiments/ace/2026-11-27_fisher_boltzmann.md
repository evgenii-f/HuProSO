# ACE energy, 2026-11-27: the pairwise matrix Fisher energy trained by likelihood, sampled by Gibbs

Branch `experimental/fisher-energy` (worktree `HuProSO-fisher`), cut from `experimental/ace-descriptors`. Validation
numbers only; no test was run. Part of the ACE line: [overview](README.md). It follows
[the compact basis](2026-11-27_compact_fisher_basis.md), which found that 92% of the completion gap of the DSM-fitted
energy is made at small noise (s < 0.04).

## Summary

- **Question.** A pairwise matrix Fisher energy on SO(3)^19 has exact conditionals: every joint given the others is a
  matrix Fisher distribution. Can it be trained without noise (pseudo-likelihood, then maximum likelihood), so that the
  holes of the DSM energies at small noise go away, and then complete poses by exact Gibbs sampling?
- **Pseudo-likelihood ranks well but has holes.** Chosen by validation (ridge 1e-4), it picks among the network's 10
  hypotheses at 77.83 mm against 81.10 mm for P+L (−3.27 [−5.38, −1.24]). Its Gibbs samples are bad (Wasserstein
  71.4°) and its inpainting is 166.75 mm, carried by the arms: 340 mm, the completions 31.8 nats below the truth.
- **Maximum likelihood (persistent contrastive divergence) closes the holes.** With D^1 only: completions within ±1.2
  nats of the truth, inpainting 114.10 mm (−50.2 against pseudo-likelihood, all of it in the arms), generation
  Wasserstein 23.13°, ranking 77.37 mm (−3.72 [−5.72, −1.78] against P+L's choice). It is mass-covering: APD 22.8 cm
  against 15.5 cm for training poses.
- **One-body terms D^{2,3} help generation, not completion.** Once the chains have converged (3000 sweeps; at 200
  they sit 16 nats above the data), Wasserstein 21.52° and FID 0.149 against 23.13° and 0.295 for D^1; inpainting
  112.87 mm, −1.23 [−2.69, +0.32] against D^1 (legs −4.4, arms +5.2); ranking 76.43 mm, the best of the line.
- **Refining the network's hypotheses with the energy fails.** No setting of the declared grid keeps best of 10 within
  1.5 mm of the network's; only iterated conditional modes lowers the mean (79.45 against 85.71 mm) and it does so by
  collapsing the hypotheses (APD 1.7 against 6.4 cm).
- **A mixture of linear energies is not built.** A component fitted on one k-means cluster places its own wrong legs
  below the truth (0.70 of them at s = 0.002), the rule fixed before the run failed. Mixtures and ensembles are deferred
  by the user until the Fisher model is exhausted.
- **Where it stands:** all Fisher models are far from P+L (Wasserstein 15.93°, inpainting 97.49 mm) and the network
  (14.86°, 85.23 mm) as generators and completers; they are the best judges of hypotheses in the line. Next, to be
  discussed: pair terms of higher degree (D^1⊗D^2, D^2⊗D^2) between neighbours in the tree, since completion is a
  conditional and the conditionals are set by the pair terms.

## Setup

- **Energy** (`explore/ace_energy/pseudo_lik.py`, class `PairwiseFisher`). With v the 171 entries of the 19 local
  rotations, $E(x) = -\langle a, v\rangle - \tfrac12 v^\top W v$, W symmetric with zero 9×9 diagonal blocks (pairs
  $B_{jk}$): 14,022 parameters, the D^1 part of the compact basis. The conditional of joint j is
  $p(x_j \mid x_{-j}) \propto \exp\langle F_j, x_j\rangle$, $F_j = a_j + \sum_k B_{jk} x_k$. Its normalizer
  $\log c(F)$ by 64-node Gauss–Legendre quadrature over the proper singular values (error 2e-6 against 128 nodes).
- **With one-body terms** (`boltzmann.py`, class `Boltzmann`, `--degrees 2 3`): $E - \sum_j \langle C_{j}, (D^2, D^3)(x_j)\rangle$,
  74 more coefficients per joint (real Wigner D-matrices of `hp.models.so3_harmonics`). The conditional becomes
  $\exp(\langle F_j, x_j\rangle + g_j(x_j))$, without a closed normalizer.
- **Pseudo-likelihood:** the mean over 100,000 training poses of the 19 conditional NLLs, convex, L-BFGS (strong
  Wolfe, 300 iterations) per ridge on the pair blocks; ridges 1e-2, 1e-3, 1e-4, chosen by the conditional NLL of the
  8,601 validation poses. workstation2 GPU, 12 min per fit.
- **Maximum likelihood** (`boltzmann.py`): gradient $\mathbb E_\text{data}[\nabla E] - \mathbb E_\text{model}[\nabla E]$, the
  model's expectation over persistent Gibbs chains (one sweep per step), started from the pseudo-likelihood fit, chains
  from uniform rotations with 20 burn-in sweeps; Adam, lr 0.02, cosine schedule, 5000 steps, 200,000 training poses,
  ridge 1e-4. D^1: 2000 chains and 2000 poses per step, Mac CPU, 0.10 s per step. D^{2,3}: 10,000 chains and 10,000
  poses per step, 3 Metropolis steps per joint update with the matrix Fisher as proposal, workstation1 GPU, 0.61 s per
  step. One run each, seed 0.
- **Sampling** (`gibbs.py`): a matrix Fisher distribution is a Bingham distribution of quaternions,
  $\langle F, R(q)\rangle = q^\top K q$, drawn by rejection from an angular central Gaussian envelope (Kent, Ganeiber
  and Mardia 2013); every sweep updates all free joints once in a random order.
- **Generation:** the harness of `generate.py` (`hp.evaluation.evaluate`): 1000 samples from uniform starts, batches
  of 500; Wasserstein, recall, precision in degrees per joint, FID_val, APD (cm), dNN.
- **Inpainting:** the first 200 validation poses of `complete.py`, K = 10 chains per pose, the visible joints at the
  truth, the hidden ones started uniform, hidden toes at rest; MPVPE (mm) mean, best of 10, APD. The comparison rows
  (P+L, network) are inpainting at t81 from the overview; they are not Gibbs.
- **Ranking:** the 10 hypotheses of the network per pose (`diag2/inpaint_network_rank.pt`), the one of lowest energy;
  against the choice of P+L's energy at s = 0.02, paired bootstrap over poses. The network's own mean on this file is
  85.84 mm.
- **Holes and convergence** (`completion_energy.py`): the mean energy of a model's own completions minus that of the
  truth, and the share below the truth. With a right model and converged chains the gap is near 0 and the share near
  0.5; a large negative gap is a hole, a positive one chains that have not converged.

## Results

### Pseudo-likelihood

| ridge | train NLL | valid NLL (nats per pose, 19 conditionals) | norm of B |
|---|---|---|---|
| 1e-2 | −124.53 | −117.50 | 29 |
| 1e-3 | −137.09 | −129.55 | 104 |
| **1e-4** | −148.69 | **−138.50** | 317 |

The smallest ridge wins on validation and is kept (uniform rotations: 0).

### Generation (harness, 1000 samples)

| model | sweeps | mean energy of the chains (nats) | Wasserstein (°) | recall (°) | precision (°) | FID_val | APD (cm) | dNN |
|---|---|---|---|---|---|---|---|---|
| training poses | – | – | – | – | – | 0.035 | 15.48 | 0.0001 |
| pseudo-likelihood | 200 | −3647.0 | 71.41 | 18.74 | 57.85 | 2.044 | 15.99 | 0.407 |
| pseudo-likelihood | 1000 | −3648.3 | 72.18 | 18.96 | 58.33 | 2.084 | 15.20 | 0.414 |
| ML, D^1 | 200 | −3481.5 (valid −3479.9) | 23.13 | 14.99 | 19.76 | 0.295 | 22.80 | 0.134 |
| ML, D^{2,3} | 200 | −3456.3 (valid −3472.7) | 32.56 | 15.07 | 26.90 | 0.897 | 28.83 | 0.175 |
| ML, D^{2,3} | 1000 | −3471.5 | 24.22 | 14.56 | 21.59 | 0.314 | 23.67 | 0.137 |
| **ML, D^{2,3}** | 3000 | −3474.9 | **21.52** | 14.61 | **19.75** | **0.149** | **20.34** | 0.122 |
| P+L (overview) | – | – | 15.93 | | | | | |
| network fact_RDW s10 (overview) | – | – | 14.86 | | | | | |

Pseudo-likelihood samples fall far below the data in energy and away from it: its holes. ML puts the chains at the
energy of the data; D^{2,3} needs thousands of sweeps to get there (its chain energy still falls between 1500 and 3000
sweeps: −3473.1, −3474.9) and then beats D^1 on every column but recall.

### Inpainting (200 poses, K = 10)

| model | sweeps | left leg | legs | arms | trunk | **mean** | best of 10 | APD (cm) | against ML D^1 |
|---|---|---|---|---|---|---|---|---|---|
| pseudo-likelihood | 200 | 124.30 | 140.31 | 345.33 | 57.04 | 166.75 | 84.25 | 11.18 | |
| pseudo-likelihood | 1000 | 124.45 | 136.27 | 339.94 | 56.70 | 164.34 | 85.36 | 10.79 | +50.24 [+47.12, +53.36] |
| ML, D^1 | 200 | 125.66 | 134.05 | 140.22 | 56.46 | 114.10 | 60.10 | 9.84 | – |
| ML, D^{2,3} | 200 | 120.88 | 143.09 | 179.17 | 56.98 | 125.03 | 59.76 | 11.21 | +10.93 [+8.83, +13.12] |
| ML, D^{2,3} | 1000 | 120.12 | 129.90 | 152.82 | 55.71 | 114.64 | 59.58 | 9.76 | +0.54 [−1.09, +2.17] |
| ML, D^{2,3} | 3000 | 120.97 | 129.67 | 145.41 | 55.42 | **112.87** | 60.08 | 9.53 | −1.23 [−2.69, +0.32] |
| P+L, t81 (overview) | | | | | | 97.49 | | | |
| network, t81 (overview) | | | | | | 85.23 | | | |

Per occlusion, D^{2,3} at 3000 sweeps against D^1: left leg −4.68 [−7.92, −1.26], legs −4.39 [−7.02, −1.61], arms
+5.19 [+1.05, +9.24], trunk −1.04 [−2.47, +0.32] mm.

### Holes and convergence of the completions

| model | sweeps | left leg | legs | arms | trunk |
|---|---|---|---|---|---|
| pseudo-likelihood | 1000 | +0.4 / 0.38 | +1.0 / 0.38 | **−31.8 / 0.92** | +0.1 / 0.37 |
| ML, D^1 | 200 | +0.4 / 0.40 | +0.4 / 0.39 | −1.2 / 0.45 | −0.1 / 0.39 |
| ML, D^{2,3} | 200 | −0.2 / 0.46 | +2.8 / 0.44 | +23.7 / 0.43 | +7.9 / 0.39 |
| ML, D^{2,3} | 1000 | −0.4 / 0.47 | −0.5 / 0.47 | +6.8 / 0.48 | +1.8 / 0.41 |
| ML, D^{2,3} | 3000 | −0.3 / 0.47 | −0.7 / 0.47 | +1.2 / 0.49 | +0.1 / 0.42 |

Each cell: mean energy of the completions minus that of the truth (nats) / share of completions below the truth. The
only hole of the pseudo-likelihood is in the arms, and ML removes it; outside the arms the two fits complete alike
(within ±2.2 mm). D^{2,3} at 200 sweeps has not converged (positive gaps), at 3000 it has.

### Ranking the network's hypotheses (200 poses, K = 10)

| energy | left leg | legs | arms | trunk | **mean** | against P+L's choice (81.10) |
|---|---|---|---|---|---|---|
| pseudo-likelihood | 74.14 | 92.78 | 104.85 | 39.53 | 77.83 | −3.27 [−5.38, −1.24] |
| ML, D^1 | 72.54 | 92.85 | 104.52 | 39.59 | 77.37 | −3.72 [−5.72, −1.78] |
| **ML, D^{2,3}** | 71.55 | 89.97 | 103.88 | 40.32 | **76.43** | **−4.66 [−6.75, −2.74]** |

Against the network's own mean (85.84 mm) the D^{2,3} choice is −9.41 [−11.36, −7.55] mm. In the share of the
network's hypotheses below the truth (0.61–0.81 for D^{2,3}, 0.62–0.81 for D^1) the Fisher energies prefer plausible
hypotheses to the truth; that is not a hole of completion, since the completions above are near the truth's energy.

### Refining the network's hypotheses (ML, D^1)

`refine.py`: the 10 hypotheses of the network, a few sweeps over the hidden joints of the conditionals sharpened by
β ($\exp(\beta\langle F_j, x_j\rangle)$; β = ∞: the mode, iterated conditional modes). Grid β ∈ {1, 2, 4, ∞} ×
sweeps ∈ {1, 3, 10}, announced before the run. Rule: on poses 0–99, the lowest mean among the settings whose best of 10
is at most 1.5 mm above the network's (48.78 mm); reported on poses 100–199.

| refinement | poses 0–99: mean | best of 10 | APD (cm) | poses 100–199: mean | best of 10 | APD (cm) |
|---|---|---|---|---|---|---|
| network | 85.97 | 47.28 | 6.21 | 85.71 | 46.45 | 6.41 |
| β 1, 1 sweep | 108.18 | 58.19 | 9.39 | 108.01 | 56.87 | 9.46 |
| β 2, 1 sweep | 96.40 | 54.31 | 7.20 | 95.16 | 52.64 | 7.15 |
| β 4, 1 sweep | 89.93 | 53.57 | 5.74 | 88.97 | 53.62 | 5.73 |
| β 4, 10 sweeps | 90.64 | 57.62 | 5.12 | 89.15 | 57.73 | 5.11 |
| β ∞, 1 sweep | 82.50 | 60.92 | 3.20 | 81.43 | 60.08 | 3.23 |
| β ∞, 3 sweeps | 81.74 | 69.07 | 1.73 | 79.45 | 67.90 | 1.71 |
| β ∞, 10 sweeps | 83.08 | 76.50 | 0.88 | 80.18 | 73.97 | 0.79 |

(the full grid in `pl/refine_bm.log`). No setting passes the rule. Finite β makes every hypothesis worse; only the mode
lowers the mean, by pulling the 10 hypotheses together. It was not run for D^{2,3}.

### Probe of a mixture of linear energies

`mixture_clusters.py` (k-means on the 171 rotation entries), `fit_cluster.py` (`fit.py` on cluster 4 of K = 8:
166,314 of 1,861,105 training poses, elbows bent), `cluster_tests.py`. Rule before the run: the component must place
the truth below its own wrong completions at s = 0.002.

| test | component | P+L | network |
|---|---|---|---|
| DSM loss, validation poses of the cluster (952) | **6.806** | 7.324 | 6.949 |
| DSM loss, other clusters (5000 poses) | 8.423 | 6.147 | 5.764 |
| inpainting t81, 22 poses with the truth in the cluster, mean (mm) | 90.98 | 102.86 | 85.63 |
| the same, legs hidden | 131.02 | 132.76 | 106.27 |
| own completions below the truth, legs hidden, s = 0.002 | 0.70 | 0.79 | |

The component gains at large noise only (s ≥ 0.05) and is as bad as P+L at s ≈ 1e-3–1e-2. Its −11.89 [−16.76, −6.94]
mm on the cluster is inflated: the cluster is taken from the truth, hidden joints included (−27.8 mm on the arms of an
elbows-bent cluster); with both legs hidden it gives −1.74 [−10.81, +9.62] mm. It has holes of its own (0.70), so the
rule fails and no full mixture is built.

## Conclusions

- Training without noise works as intended: maximum likelihood removes the hole that pseudo-likelihood leaves (the
  arms, 32 nats), and the energy of the completions then agrees with that of the truth. The holes of small noise are
  a matter of how the energy is trained.
- The model is too simple, not the sampler: with converged chains the Fisher energies complete at 113–114 mm, 15 mm
  behind P+L and 28 mm behind the network, and their samples are 30–50% broader than the data (APD 20–23 against 15.5 cm).
  ML is mass-covering: a limited family spreads its mass.
- The one-body terms D^{2,3} shape the marginals (generation better) but barely the conditionals of hidden joints given
  visible ones (completion −1.2 mm, not significant). The pair terms are what completion depends on.
- The Fisher energies are the best judges of the network's hypotheses in the line (76.43 mm against 81.10 mm for P+L
  and 85.84 mm for the network's mean), but correcting the hypotheses by their conditionals fails.
- Next (to be discussed before it is carried out): pair terms of higher degree between neighbours in the tree, trained
  by ML on a GPU with long chains. Mixtures, ensembles and products with the network wait (user's decision).

## What it does not show

- One ML run per model (seed 0, one learning rate, one ridge, 5000 steps); the training gap of the chains still moves
  by a few nats between checkpoints. A longer or tuned training may change the numbers.
- 200 validation poses, K = 10; 1000 samples for the harness. No test.
- D^1 ML was sampled with 200 sweeps only; the energies of its completions say it had converged (gaps within ±1.2
  nats), but its generation at 3000 sweeps was not run.
- The D^{2,3} chain energy in generation was still falling slowly at 3000 sweeps.
- Gibbs inpainting is compared with t81 inpainting of the other rows, not with the same sampler.
- Refinement and the holes test of the network's hypotheses at s = 0 are for the D^1 fits; refinement was not run for
  D^{2,3}.
- The mixture probe used one cluster of one k-means (K = 8) and took the cluster from the truth.

## Where things are

- **Code** (`explore/ace_energy/`): `pseudo_lik.py` (the energy, its normalizer, the pseudo-likelihood fit, the holes
  and ranking tests), `gibbs.py` (the sampler, `--generate`, `--inpaint`, `--device`), `boltzmann.py` (ML by persistent
  contrastive divergence, one-body degrees), `refine.py` (refinement of the network's hypotheses),
  `completion_energy.py` (holes and convergence of completions, paired differences), `mixture_clusters.py`,
  `fit_cluster.py`, `cluster_tests.py` (the mixture probe).
- **Fits** (`~/data/experiments/ace/pl/`): `energy_pl.pt` (pseudo-likelihood, log `pl.log`), `energy_bm.pt` (ML, D^1,
  `bm.log`), `energy_bm_w123.pt` (ML, D^{2,3}, `bm_w123.log`).
- **Samples and completions** (same folder): `generate_gibbs_{200,1000}.pt`, `generate_bm_200.pt`,
  `generate_bm_w123_200.pt`, `generate_bm_w123_long_{1000,3000}.pt`; `inpaint_gibbs_{50,200,1000}.pt`,
  `inpaint_bm_200.pt`, `inpaint_bm_w123_200.pt`, `inpaint_bm_w123_long_{1000,3000}.pt`; their logs `generate_*.log`,
  `inpaint_*.log` (`inpaint_bm_w123_long3000.log` for 3000 sweeps).
- **Tests:** `tests.log` (pseudo-likelihood), `tests_bm.log`, `tests_bm_w123.log` (ranking, holes),
  `refine_bm.log`, `completion_energy.log` (the holes and convergence table).
- **Mixture probe** (`~/data/experiments/ace/`): `mixture_clusters.pt`, `energy_cluster8_4.pt`, `cluster8_4.log`,
  `cluster/` (`tests.log`, `tests_holes_mac.log`, `inpaint.log`, `inpaint_cluster8_4.pt`).

Commands, from the root:

    python explore/ace_energy/pseudo_lik.py --device cuda:0 --no-tests
    python explore/ace_energy/boltzmann.py --out ~/data/experiments/ace/pl/energy_bm.pt
    python explore/ace_energy/boltzmann.py --degrees 2 3 --chains 10000 --batch 10000 --device cuda:0 \
        --out ~/data/experiments/ace/pl/energy_bm_w123.pt
    python explore/ace_energy/gibbs.py --fit ~/data/experiments/ace/pl/energy_bm_w123.pt --generate --sweeps 1000 3000 \
        --device cuda:0 --out ~/data/experiments/ace/pl/generate_bm_w123_long.pt
    python explore/ace_energy/gibbs.py --fit ~/data/experiments/ace/pl/energy_bm_w123.pt --inpaint --sweeps 3000 \
        --device cuda:0 --out ~/data/experiments/ace/pl/inpaint_bm_w123_long.pt
    python explore/ace_energy/refine.py --fit ~/data/experiments/ace/pl/energy_bm.pt
    python explore/ace_energy/completion_energy.py pl/energy_bm.pt:pl/inpaint_bm_200.pt \
        pl/energy_pl.pt:pl/inpaint_gibbs_1000.pt pl/energy_bm_w123.pt:pl/inpaint_bm_w123_long_3000.pt
