# MACE-lite: a graph score network for RSGM (plan, not implemented)

Status: draft for discussion, 2026-11-23. Branch `experimental/wigner-inputs`.

## Why

Section 18 of [the score report](experiments/2026-11-14_score_network_architectures.md) found:

- the relations of the joints help the score at small noise when given as plain rotations (l = 1): `local+pairs`
  lowers the first phase of SDS from 11.24 to 11.00 deg and keeps the spread and the best of 10;
- the gain comes from the 11 left-right and sibling pairs, not from the chains to the root (`local+global`: 11.27);
- a larger input hurts: the higher harmonics (D^2, D^3) and their pair products make the score more confident, not
  more accurate, and most of the parameters of those networks sit in the first layer (2.4-6.2M).

A dense MLP sees every relation as its own input number with its own weights. A graph network sees them as edges
of one kind with shared weights, as MACE does for atoms. Expected: the same or better first phase with an order of
magnitude fewer parameters, and less overfitting. That is the claim to test.

## Model

- **Nodes:** the 19 joints. Node input: a learned embedding e_j of the joint (16-32 numbers), its local rotation
  x_j (9 numbers, D^1) and the time.
- **Edges:** the 18 parent-child edges of the tree and the 11 pairs of `WIGNER_PAIRS`, in both directions (58
  directed edges). Edge input: the relative rotation R_ik = G_i^T G_k (9 numbers; for a parent and a child it is
  x_k), an embedding of the edge type (tree or pair, and which), and the time.
- **Layer** (repeated L = 2-3 times; nothing is equivariant, as there is no symmetry to keep):
  - message A_i = sum over k in N(i) of phi(R_ik, type_ik, t) * W h_k, channel-wise. This is the analog of the
    atomic base of ACE: a learned radial-and-angular function of the edge times the neighbour's state;
  - products B_i = (U A_i) * (V A_i), channel-wise: correlations of two neighbours, body order 3, as the
    product step of MACE;
  - update h_i <- h_i + MLP([h_i, A_i, B_i, t]), with residual connections (which the factorial found helpful).
- **Readout:** the score of joint i from its own node, s_i = Q_i h_i (3 numbers), then the existing whitening of
  the per-axis noise (`_unscale`). Q_i per joint, or shared plus the joint embedding.
- **Size:** C = 256 channels, L = 3: about 1-2M parameters against 14.5M of `wide_RAD_x` (to be computed when
  written). The cost is that of 19 nodes and 58 edges: small.
- **In RSGM:** a new `net="mace"` next to `"mlp"` and `"timeres"`. The network takes x [B, 19, 3, 3] and t directly
  (not the flattened features), with the same per-joint, per-axis noise of `wide_RAD_x`, so the likelihood stays as
  it is (none with `joint_cov`) and the exact divergence of `log_prob` still works for the isotropic noise, since the
  network is a function of x only.

## Tests (before any training)

- shapes and the number of parameters;
- the score of a joint depends only on joints within L edges of it (a locality check by finite differences);
- the exact divergence equals finite differences;
- a short training step: finite loss and gradients;
- the edges: tree and pairs as listed, both directions.

## Experiment

`mace_x`: the graph network with the noise, data and budget of `wide_RAD_x` (full data, 30k x 4096, seed 10), on
workstation2. Evaluated by the staged rule below, against `local+pairs` (the best dense network) and `local`.

**Prediction:** non-inferior to `local+pairs` at a fifth or less of its parameters, with a validation loss not worse.

## Rule: a staged evaluation on validation (fixed 2026-11-23, for this and the next models of this line)

The model is for completion and generation, so the decision rests on both, on all occlusions, and the cheap stages
run first: a stage stops only a clearly bad model, the decision is taken on stages 1 and 2 together, and the costly
diagnostics run for the models that pass. `explore/evaluation/candidate_panel.py` runs stages 0-2 (about 2 minutes
per model on the Mac) and appends one line per run to `~/data/experiments/candidate_panel.jsonl`.

| stage | what | gate |
|---|---|---|
| 0. training | last validation loss of the run | stop if worse than the reference by > 0.03, or diverged |
| 1. completion | `hybrid.py 200` (DPoser's procedure with the RSGM denoiser), 4 occlusions, K = 10 | stop if the mean or the best of 10 is worse by > 3 mm, or the APD lower by > 0.8 cm |
| 2. generation | the harness on validation (`hp.evaluation.evaluate`, own sampler, 20k samples, 2000 queries, 3 repeats, no likelihood): Wasserstein, recall, precision | the decision |
| 3. details | phase diagnostics (`trajectory.py`), quick protocol | only after stage 2 passes; for the write-up |

**Decision** (against the reference on the same panel):

- **improvement:** at least one of: completion mean or best of 10 better by >= 1.5 mm (with the APD not lower by
  > 0.3 cm); Wasserstein, recall or precision better by >= 0.25 deg; and nothing worse than the tolerances;
- **a strong result:** a generation metric better by >= 1.5 deg (more than a third of the distance to real poses,
  3.6-4.9 deg), with nothing worse than the tolerances;
- **tolerances:** completion mean and best of 10 not worse by > 1.0 mm, each generation metric not worse by > 0.15 deg;
- otherwise no effect.

**Where the thresholds come from** (the references on the panel, full data, 2026-11-23):

| model | valid loss | completion mean / APD / best of 10 | Wasserstein | recall | precision |
|---|---|---|---|---|---|
| local (`wide_RAD_x`) s10 | 1.639 | 81.7 / 4.27 / 53.2 | 18.64 | 14.03 | 17.03 |
| local s11 | 1.645 | 81.4 / 4.25 / 53.7 | 18.56 | 14.05 | 16.98 |
| local+pairs (l = 1) s10 | 1.646 | 80.4 / 4.14 / 52.4 | 18.34 | 13.92 | 16.73 |
| real training poses (contrast) | | | 14.75 | 11.78 | 12.15 |

Between the two seeds of `local`: 0.006 in the validation loss, 0.3-0.5 mm in completion, 0.02-0.08 deg in
generation; the repeats of the harness within a run are 0.01-0.08 deg. The thresholds are about three times that
spread (one pair of seeds only, so a rough estimate). By this rule `local+pairs` against `local` is an improvement:
Wasserstein -0.31 and precision -0.30 deg, completion -1.3 mm mean and -0.8 best of 10, nothing worse.

Ablations if it works: the tree alone against the tree and the pairs; L = 1, 2, 3; the products B_i off (body order
2 only); channels 128 and 512.

## Open questions

- Literature, to check before writing: graph networks for human pose priors and on the kinematic tree, and how they
  relate to MACE (Batatia et al. 2022) and ACE (Drautz 2019). What is new here must be stated against them.
- Whether the time goes into every layer (as `timeres` can) or into the node input only.
- Whether the score of a joint should also see the edge to its parent explicitly in the readout (the score is a
  tangent vector in the frame of x_j).
