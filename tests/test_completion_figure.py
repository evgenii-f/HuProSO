"""Pictures of pose completions (hp/visualization/completion_figure.py), with a small fake body instead of SMPL-X."""
import numpy as np
import pytest
import roma
import torch

from hp.evaluation.completion import cache_path
from hp.visualization import completion_figure as cf


class FakeBody:
    """A tetrahedron moved by the rotation of joint 0, and 22 "joints": the rotated axes of the 21 joints."""

    def __init__(self):
        self.model = type("Model", (), {})()
        self.model.init_v_template = torch.zeros(1, 4, 3)
        self.model.f = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        self.vertices = {"legs": [2, 3], "arms": [0]}
        self.tetra = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)

    def __call__(self, poses):
        poses = poses.double()
        verts = torch.einsum("bij,vj->bvi", poses[:, 0], self.tetra)
        joints = torch.cat([torch.zeros(len(poses), 1, 3, dtype=torch.float64), poses[..., 0]], dim=1)
        return verts, joints


def write_cache(root, label, occlusion, pred, mpvpe):
    path = cache_path(root, label, occlusion)
    path.parent.mkdir(parents=True)
    torch.save({"pred": pred.half(), "settings": "{}"}, path)
    torch.save({"settings": "{}", "plain": {"mpvpe": torch.from_numpy(mpvpe)}, "rest": {}},
               path.with_name(f"{occlusion}_metrics.pt"))


def test_hidden_vertices_are_those_of_the_metric():
    body = FakeBody()
    mask = cf.hidden_vertices(body, "legs")
    assert mask.tolist() == [False, False, True, True]
    colors = cf.vertex_colors(mask)
    assert np.allclose(colors[2], cf.HIDDEN) and np.allclose(colors[0], cf.VISIBLE)


def test_choose_the_hypotheses():
    torch.manual_seed(0)
    poses = roma.random_rotmat(10 * 21).view(10, 21, 3, 3)
    errors = np.array([5.0, 1, 9, 3, 7, 2, 8, 4, 6, 0])
    assert cf.choose(poses, 3, "first") == [0, 1, 2]
    best = cf.choose(poses, 3, "best to worst", errors)
    assert best[0] == 9 and best[-1] == 2  # the best and the worst at the ends
    diverse = cf.choose(poses, 4, "diverse", body=FakeBody())
    assert len(set(diverse)) == 4
    with pytest.raises(ValueError):
        cf.choose(poses, 3, "best to worst")


def test_pick_pose_from_the_errors():
    rng = np.random.default_rng(0)
    a = np.arange(100, dtype=float)[:, None].repeat(10, 1)   # method 1: pose i has error i
    b = np.full((100, 10), 50.0)
    assert cf.pick_pose("hard for 1", [a, b], rng, top=5) >= 95
    assert cf.pick_pose("easy for 1", [a, b], rng, top=5) < 5
    assert cf.pick_pose("1 better than 2", [a, b], rng, top=5) < 5
    assert cf.pick_pose("2 better than 1", [a, b], rng, top=5) >= 95
    assert 0 <= cf.pick_pose("random", [a, None], rng) < 100
    with pytest.raises(ValueError):
        cf.pick_pose("1 better than 2", [a, None], rng)


def test_rows_from_the_cache(tmp_path):
    torch.manual_seed(0)
    pred = roma.random_rotmat(6 * 10 * 21).view(6, 10, 21, 3, 3)
    mpvpe = np.random.default_rng(0).uniform(10, 100, (6, 10)).astype(np.float32)
    write_cache(tmp_path, "A: method, one", "legs", pred, mpvpe)
    assert cf.cached_labels(tmp_path) == ["A: method, one"]
    gt = roma.random_rotmat(6 * 21).view(6, 21, 3, 3)
    row = cf.completion_rows(tmp_path, ["A: method, one", "B: missing"], "legs", pose=4, k=3, body=FakeBody(), gt=gt,
                             how="best to worst")
    a, b = row["methods"]
    assert torch.allclose(a["poses"], pred[4, a["indices"]].half().float())
    assert np.allclose(a["errors"], mpvpe[4, a["indices"]]) and a["errors"][0] == mpvpe[4].min()
    assert b["poses"] is None and torch.equal(row["gt"], gt[4])


def test_figure_renders(tmp_path):
    pytest.importorskip("pyvista")
    torch.manual_seed(0)
    pred = roma.random_rotmat(2 * 10 * 21).view(2, 10, 21, 3, 3)
    write_cache(tmp_path, "A", "legs", pred, np.ones((2, 10), dtype=np.float32))
    body, gt = FakeBody(), roma.random_rotmat(2 * 21).view(2, 21, 3, 3)
    row = cf.completion_rows(tmp_path, ["A"], "legs", pose=1, k=2, body=body, gt=gt)
    images = cf.render_row(row, body, azimuths=(0, 90), size=(60, 80), ghost=True)
    assert images["methods"]["A"].shape == (2, 2, 80, 60, 3) and images["gt"].shape == (2, 80, 60, 3)
    pink = np.array(cf.HIDDEN) * 255
    assert (np.abs(images["gt"].astype(float) - pink).sum(-1) < 120).any()  # the hidden part is drawn pink
    fig = cf.completion_figure([row], [images], names={"A": "method A"})
    fig.savefig(tmp_path / "figure.png")
    assert (tmp_path / "figure.png").stat().st_size > 0
