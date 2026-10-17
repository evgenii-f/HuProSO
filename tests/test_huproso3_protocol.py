"""The evaluation protocol of the HuProSO3 paper (hp/evaluation/huproso3_protocol.py)."""
import numpy as np
import pytest
import roma
import torch
from test_evaluation_harness import Around, cloud

from hp.evaluation.huproso3_protocol import (PaperProtocolConfig, evaluate_paper_protocol, joint_positions,
                                             nearest_mpjpe)
from hp.evaluation.neighbours import nearest_distances

CONFIG = PaperProtocolConfig(n_queries=50, n_pool=400, n_likelihood=60, repeats=2, sample_batch=200,
                             likelihood_batch=25, chunk=128)


class AroundWithDensity(Around):
    """Around with a log_prob that returns fixed values, to check the likelihood statistics."""
    has_log_prob = True
    density_space = "SO(3)^J"

    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        self.values = values

    def log_prob(self, pose, cond=None):
        return self.values(pose)


def test_the_geodesic_distance_is_the_sum_over_the_joints():
    a, b = cloud(7, seed=1), cloud(30, seed=2)
    angles = torch.stack([roma.rotmat_geodesic_distance(a[:, None, j], b[None, :, j]) for j in range(a.shape[1])]).sum(0)
    assert torch.allclose(nearest_distances(a, b, "sum", chunk=8), angles.min(1).values, atol=1e-5)


def test_nearest_mpjpe_is_the_brute_force_minimum():
    g = torch.Generator().manual_seed(0)
    q, c = torch.randn(9, 22, 3, generator=g), torch.randn(50, 22, 3, generator=g)
    brute = (q[:, None] - c[None]).norm(dim=-1).mean(-1).min(1).values
    assert torch.allclose(nearest_mpjpe(q, c, chunk=7), brute, atol=1e-5)


def test_joint_positions_are_in_mm_and_rooted_at_the_origin():
    rest = joint_positions(torch.eye(3).expand(2, 19, 3, 3).contiguous())
    assert rest.shape == (2, 22, 3)
    assert torch.equal(rest[:, 0], torch.zeros(2, 3))
    assert 1500 < rest[0, :, 1].max() - rest[0, :, 1].min() < 1700  # head to foot of the rest pose, about 1.6 m


def test_a_model_like_the_data_scores_better_than_a_shifted_one():
    splits = {"test": cloud(500, seed=0), "train": cloud(500, seed=1)}
    good = evaluate_paper_protocol(Around(), splits, CONFIG)
    bad = evaluate_paper_protocol(Around(shift_deg=30), splits, CONFIG)
    for split in splits:
        for metric in ("recall_geodesic", "precision_geodesic"):
            assert good[split][metric]["mean"]["mean"] * 3 < bad[split][metric]["mean"]["mean"]
    assert "recall_mpjpe" not in good["test"]  # only for the 19 joints of the body


def test_the_dataset_poses_are_the_same_for_every_model_and_the_report_is_reproducible():
    splits = {"test": cloud(500, seed=0)}
    a = evaluate_paper_protocol(Around(), splits, CONFIG)
    b = evaluate_paper_protocol(Around(), splits, CONFIG)
    assert a["test"]["recall_geodesic"] == b["test"]["recall_geodesic"]
    # real poses as the "generator": the same dataset queries, so recall of the data against itself is 0
    given = evaluate_paper_protocol(None, splits, CONFIG, generated=splits["test"])
    assert given["test"]["recall_geodesic"]["mean"]["mean"] == pytest.approx(0, abs=1e-2)
    assert given["test"]["likelihood"] is None


def test_the_likelihood_statistics_see_broken_values():
    def values(pose):  # 130 everywhere, but one pose in fifty is broken
        v = torch.full((len(pose),), 130.0)
        v[::50] = 7000.0
        return v

    cfg = PaperProtocolConfig(**{**CONFIG.__dict__, "likelihood_batch": 100})  # all 60 poses in one call: 2 broken
    r = evaluate_paper_protocol(AroundWithDensity(values), {"test": cloud(500, seed=0)}, cfg)["test"]["likelihood"]
    assert r["median"]["mean"] == 130.0
    assert r["mean"]["mean"] > 200  # the mean alone hides the problem
    assert r["implausible_share"]["mean"] == pytest.approx(2 / 60)
    assert len(r["values"]) == CONFIG.n_likelihood


def test_model_and_generated_poses_are_exclusive():
    with pytest.raises(ValueError, match="either a model or generated"):
        evaluate_paper_protocol(Around(), {"test": cloud(10)}, CONFIG, generated=cloud(10))
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_paper_protocol(Around(), {"test": cloud(10)[:, :2]}, CONFIG)
