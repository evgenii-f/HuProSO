# ACE energy, 2026-11-26: ideas from ACE for the network of RSGM (neutral and negative)

Branch `experimental/ace-descriptors`. Validation numbers only; no test was run. Part of the ACE line: [overview](README.md).

## Summary

- **2026-11-26: ideas from ACE for the network of RSGM**.
  - Mixing the scores of the network and the energy below s = 0.04 lowers the validation loss by 0.43, as much as a
    second network (0.38); the harness does not change (14.99 against 14.93°) and is saturated. Deferred to the end.
  - A block that aggregates the joints by body parts, added to `fact_RDW` (branch `experimental/cluster-block`), does
    not pass its one-point gate: inpainting t99 mean −0.76 mm with APD −0.35 cm and best of 10 +1.04 mm. Closed.

**Question.** The linear energy generalizes better than the network: its gap between fresh train and validation loss
below s = 0.04 is 3.5 (41.41 / 44.90), the network's 8.2 (35.60 / 43.75). Can this help
the network, (a) by mixing the two scores without training, or (b) by a block that aggregates the joints by body
parts, added to the network?

## Setup

- **(a) Mixing.** score = (1 − λ) network + λ partner below s = 0.04, the network `fact_RDW` s10, the partner the linear
  energy products + limits (ridge 1e-3, "P+L") or triples 5, or another isotropic network of the factorial as a control
  (`RD`, `RW`, `DW`, seed 10). `explore/ace_energy/mix_scores.py`: the loss of the 10 bins below s = 0.04, 2500
  validation draws per bin and seed; the loss is quadratic in λ, so λ is fitted on the draws of seed 1 and the change is
  measured on those of seed 2 (the same validation pool, other noise).
- **(b) Cluster block** (branch `experimental/cluster-block`, 2a72db8; `score_nets.ClusterBlock`, RSGM `net_clusters`,
  config `rdw_clusters`). A per-joint encoder of x_j, a joint embedding and t; the mean over six clusters (each leg,
  the spine, neck and head, each arm); an MLP exchange over the six tokens; a zero-initialized readout of 3 numbers per
  joint added to the output of the `fact_RDW` network, so that the model starts as the network (0.42M parameters next
  to 4.43M). Trained as `fact_RDW` (c10, 30k × 4096, lr 2e-3, seed 10) on workstation2, 4.8 minutes against 3.4.
- **(b) Gate**, written before the runs (`~/data/experiments/clusters_gate/PLAN.md`; changed before any run from the 8
  points of the pairs gate to one point, at the user's request): inpainting t99 of the validation front of `fact_RDW` at
  beta 0, the lowest mean of all RSGM points (86.16 mm, best of 10 47.64 mm, APD 6.25 cm; DPoser's own optimization at
  a similar APD, t3: 86.17 / 49.99 / 6.02), 1000 validation poses, K = 10, `scripts/tune_completion.py` at the fixed
  point (`explore/ace_energy/cluster_gate.py`). Step 1, seed 10: pass if d mean ≤ −1.0 mm, d APD ≥ −0.2 cm and d best
  of 10 ≤ +0.3 mm against `fact_RDW` s10; otherwise the line is closed. Step 2, seed 11, only after a pass. A
  reproduction of `fact_RDW` s10 at the point with the new checkout first. Prediction: d mean within ±1 mm, no pass.

## Results: mixing the scores

Loss change on the draws of seed 2 at the λ fitted on seed 1 (network alone 43.78; s.e. 0.002–0.013):

| partner | partner alone | λ | loss change |
|---|---|---|---|
| ACE P+L | 44.97 | 0.35 | **−0.43** |
| ACE triples 5 | 45.26 | 0.32 | −0.39 |
| network `RD` | 43.60 | 0.59 | −0.38 |
| network `RW` | 45.41 | 0.11 | −0.02 |
| network `DW` | 42.78 | 0.90 | −1.01 |

- The energy errs differently from the network: a network of similar loss (`RW`) adds nothing, the energy −0.43; the
  correlation of the residuals with the network's is 0.94–0.98 per bin for the energy, 0.98–0.99 for `RD`. The gain of
  the energy is largest at the smallest noise (−0.86 in the first bin, −0.20 in the last); a λ per bin gives −0.48.
- But a second network of the same family (`RD`) gives nearly as much: the gain is that of an ensemble, not one only the
  energy has. (`DW`, without the residual connections, has the lowest loss of all; the score report shows that the loss
  does not choose a model.)
- Harness (1000 samples, batch 500, validation; Wasserstein / precision, deg): network 14.93 / 12.86, the mix with P+L
  at λ = 0.35 14.99 / 12.94, the mix with `RD` at λ = 0.585 14.98 / 12.92, the training poses in the place of the
  samples 15.15 / 13.08. No difference, and the harness is saturated for the network: its samples score as well as
  the training poses. Completion of the mixes was not run; the user deferred the ensemble to the end of the line.

## Results: the cluster block

Inpainting t99, 1000 validation poses, K = 10 (mm, APD in cm; the four occlusions are the mean MPVPE):

| model | mean | best of 10 | APD | left leg | legs | arms | trunk |
|---|---|---|---|---|---|---|---|
| `fact_RDW` s10 (front) | 86.16 | **47.64** | 6.25 | 82.85 | 105.73 | 111.38 | 44.68 |
| `fact_RDW` s10, reproduced with the new checkout | 86.16 | 47.64 | 6.25 | 82.85 | 105.73 | 111.38 | 44.68 |
| `fact_RDW` s11 (pairs gate) | 86.19 | 47.80 | 6.28 | 82.96 | 106.08 | 110.79 | 44.91 |
| `fact_RDW` + cluster block s10 | **85.40** | 48.69 | 5.90 | 82.00 | 104.93 | 109.96 | 44.70 |
| block − `fact_RDW` s10 | −0.76 | +1.04 | −0.35 | −0.85 | −0.80 | −1.42 | +0.02 |

- By the rule the block does not pass step 1: none of the three conditions holds (−0.76 > −1.0 mm, APD −0.35 < −0.2 cm,
  best of 10 +1.04 > +0.3 mm). Seed 11 was not run; the line is closed. The reproduction is exact, and the two control
  seeds differ by 0.03 mm in the mean at this point.
- The lower mean comes with narrower hypotheses. With the slope of inpainting along the front (6.9–8.3 mm per cm of
  APD, the pairs gate) the 0.35 cm explain about −2.4 to −2.9 mm, so at equal APD the block is likely 1.6–2.1 mm worse
  (post hoc, an extrapolation: the front of `fact_RDW` has no point below APD 6.25 cm).
- Its validation loss is worse too: 5.938 against 5.848 (`history.json`, the last six values stable).

## Conclusions (ideas from ACE for the network)

1. **The energy is a usable partner of the network but not a unique one.** Mixing lowers the loss by 0.43, about as
   much as a second network; the harness cannot show it. Deferred by the user to the end of the line; the open test is
   completion of the mix with the `RD` mix as control.
2. **Aggregating the joints by body parts does not help the network** at the best inpainting point: narrower hypotheses,
   a worse best of 10 and a worse loss. Closed.
3. Neither idea improves what we measure for the network (the harness, completion); the ACE line goes on without them.

## What it does not show (ideas from ACE for the network)

- Mixing: one network, two partners of each kind, one draw of the validation poses; only the loss and the harness.
- The cluster block: one seed, one point chosen for the network without the block, one design of the block (6 clusters,
  128 channels, mean pooling); the comparison at equal APD is an extrapolation.

## Where things are

- **2026-11-26, ideas from ACE for the network:**
  - mixing: `explore/ace_energy/mix_scores.py` (loss), `explore/ace_energy/mix_harness.py` (harness); notes and the
    first scripts in `~/data/experiments/ace/mix_check/`;
  - cluster block: code on `experimental/cluster-block` (2a72db8), worktree `~/repos/HuProSO-cluster` on workstation2;
    run `~/data/experiments/clusters/rsgm_posedflow_rdw_clusters_s10` (workstation2 and the Mac); gate plan, configs,
    studies and log in `~/data/experiments/clusters_gate/` (`PLAN.md`, `optuna/`, `chain_s10.log`); the comparison
    `explore/ace_energy/cluster_gate.py compare`.
