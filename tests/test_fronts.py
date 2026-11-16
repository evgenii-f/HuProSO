"""Pareto fronts (hp/evaluation/fronts.py) and their drawing (hp/visualization/fronts.py)."""
import math

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hp.evaluation.fronts import front  # noqa: E402
from hp.visualization.fronts import draw_front  # noqa: E402


def names(rows):
    return [r["name"] for r in rows]


def test_two_objectives_min_and_max():
    rows = [dict(name="a", mean=90, apd=5), dict(name="b", mean=100, apd=8), dict(name="c", mean=95, apd=4),
            dict(name="d", mean=120, apd=8), dict(name="e", mean=130, apd=12)]
    # c is worse than a on both, d than b (equal APD, higher mean)
    assert names(front(rows, ("mean", "min"), ("apd", "max"))) == ["a", "b", "e"]
    # the order is that of the first objective from its best end
    assert names(front(rows, ("apd", "max"), ("mean", "min"))) == ["e", "b", "a"]


def test_three_objectives_keep_a_point_good_only_on_the_third():
    rows = [dict(name="a", x=1, y=1, z=5), dict(name="b", x=2, y=2, z=1), dict(name="c", x=3, y=3, z=6)]
    assert names(front(rows, ("x", "min"), ("y", "min"))) == ["a"]
    assert names(front(rows, ("x", "min"), ("y", "min"), ("z", "min"))) == ["a", "b"]


def test_equal_rows_both_stay_and_a_function_as_value():
    rows = [dict(name="a", v=(1, 2)), dict(name="b", v=(1, 2)), dict(name="c", v=(2, 2))]
    got = front(rows, (lambda r: r["v"][0], "min"), (lambda r: r["v"][1], "min"))
    assert names(got) == ["a", "b"]


def test_tolerance_makes_a_near_tie_dominated():
    # b is faster than a by 5 %, a has the better metric: both on the front without a tolerance; with 10 % the times
    # are a tie and a, better on the metric, dominates b
    rows = [dict(name="a", time=1.05, fid=0.1), dict(name="b", time=1.0, fid=0.2)]
    assert names(front(rows, ("time", "min"), ("fid", "min"))) == ["b", "a"]
    assert names(front(rows, ("time", "min", 0.1), ("fid", "min"))) == ["a"]


def test_tolerance_of_a_maximized_objective():
    rows = [dict(name="a", mean=90, apd=4.9), dict(name="b", mean=95, apd=5.0)]
    assert names(front(rows, ("mean", "min"), ("apd", "max"))) == ["a", "b"]
    assert names(front(rows, ("mean", "min"), ("apd", "max", 0.05))) == ["a"]


def test_rows_without_a_value_are_left_out():
    rows = [dict(name="a", x=1, y=None), dict(name="b", x=2, y=math.nan), dict(name="c", x=3, y=1)]
    assert names(front(rows, ("x", "min"), ("y", "min"))) == ["c"]


def test_bad_objectives():
    with pytest.raises(ValueError):
        front([dict(x=1)])
    with pytest.raises(ValueError):
        front([dict(x=1)], ("x", "lowest"))
    with pytest.raises(ValueError):
        front([dict(x=1)], ("x", "min", -0.1))


@pytest.mark.parametrize("corner, expected", [
    ("y", ([1, 1, 2, 2, 3], [10, 20, 20, 30, 30])),
    ("x", ([1, 2, 2, 3, 3], [10, 10, 20, 20, 30])),
])
def test_staircase(corner, expected):
    fig, ax = plt.subplots()
    draw_front(ax, [1, 2, 3], [10, 20, 30], "steps", corner=corner)
    x, y = ax.get_lines()[-1].get_data()
    assert (list(x), list(y)) == expected
    plt.close(fig)


def test_lines_and_points():
    fig, ax = plt.subplots()
    draw_front(ax, [1, 2], [3, 4], "lines")
    draw_front(ax, [1, 2], [3, 4], "points")
    lines, points = ax.get_lines()
    assert list(lines.get_xdata()) == [1, 2] and lines.get_linestyle() == "--"
    assert points.get_linestyle() == "None" and points.get_marker() == "o"
    with pytest.raises(ValueError):
        draw_front(ax, [1], [1], "dots")
    plt.close(fig)
