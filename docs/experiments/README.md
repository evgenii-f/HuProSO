# Experiment reports

One file per experiment: why we ran it, how, what came out and what we conclude. Named
`YYYY-MM-DD_<topic>.md` by the day the experiment started; data and figures of a report sit next to it (`figures/`,
`<date>_<topic>.json`). References, guides and plans stay in `docs/` (`comparison_protocols.md`,
`resuming_training.md`, `poserfm_plan.md`).

A report: the question, a summary with the answer, the setup (data, models, settings, scripts), the results with
numbers, the conclusions, what it does not show, and where the runs and caches are. Settings chosen on validation are
reported with the rule that chose them; a test is run once. When to write one and the checklist are in `CLAUDE.md`
and the skill `.claude/skills/experiment-report/SKILL.md` (agents use it; anyone can follow it). A line of work keeps
one report: a later phase adds a dated section and updates the summary.

| date | report | in one line |
|---|---|---|
| 2026-10-14 | [RSGM hyperparameter tuning](2026-10-14_rsgm_hyperparameter_tuning.md) | Two Optuna rounds: 131.7 → 138.3 nats on validation, capacity and a higher lr are the levers; a plateau |
| 2026-10-14 | [RSGM r2t12: unstable training on rare poses](2026-10-14_rsgm_rare_pose_instability.md) | Loss spikes and broken densities on a few rare poses, not overfitting; data in `2026-10-14_rsgm_rare_poses.json` |
| 2026-10-20 | [Mirror augmentation of RSGM](2026-10-20_rsgm_mirror_augmentation.md) | Negative: every metric worse, AMASS is not mirror symmetric |
| 2026-10-22 | [Models with the HuProSO3 paper's protocol](2026-10-22_model_comparison_paper_protocol.md) | RSGM and PoseRFM beat the HuProSO3 flow by 2-6 nats; RSGM full t23 beats real data on recall and precision |
| 2026-10-22 | [Conditional RSGM](2026-10-22_rsgm_conditioning.md) | p(pose \| keypoints) by an extra input of the score network; the paper's condition MLP and occlusion masks |
| 2026-11-11 | [PoseD-Flow on CUDA: the QR projection](2026-11-11_posedflow_cuda_qr_projection.md) | Their final `torch.linalg.qr` costs ~400 µs per pose on CUDA, not the flow; fix by Gram-Schmidt |
| 2026-11-13 | [Pose completion: RSGM against PoseD-Flow and DPoser](2026-11-13_pose_completion_rsgm_vs_dposer.md) | D-Flow with Heun EDM 50 NFE matches PoseD-Flow in best of 10, 8.7 mm behind in the mean; the likelihood weighting helps SDS and hurts D-Flow |
| 2026-11-14 | [Score networks and a DPoser-like network for RSGM](2026-11-14_score_network_architectures.md) | The time-conditioned residual network removes the drift of SDS at small noise; ablation on `experimental/score-net-ablation` |
