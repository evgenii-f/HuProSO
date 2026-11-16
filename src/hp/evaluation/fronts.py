"""Pareto fronts of rows (dicts) on any number of objectives, shared by the notebooks of the comparisons: the sample
quality against the cost of sampling (pareto_posedflow_vs_rsgm.ipynb) and the pose completion on the validation poses
(completion_fronts.ipynb).

An objective is (value, direction) or (value, direction, tolerance): value a key of the rows or a function of a row,
direction "min" or "max", tolerance a relative tie (default 0). A row dominates another if it is at least as good on
every objective, a value within the tolerance of the other's counting as good (worse by at most tolerance * |other|),
and strictly better on one. The front is the rows no other row dominates.

    front(rows, ("mean", "min"), ("apd", "max"))
    front(timed, (lambda r: r["times"][n], "min", 0.1), ("FID", "min"))   # times within 10 % are a tie
"""
import math
from typing import Any, Callable, List, Sequence, Tuple, Union

Value = Union[str, Callable[[dict], float]]
Objective = Union[Tuple[Value, str], Tuple[Value, str, float]]
DIRECTIONS = ("min", "max")


def _parse(objective: Objective) -> Tuple[Callable[[dict], float], float, float]:
    """(get, sign, tolerance): sign 1 for "min" and -1 for "max", so that lower sign * value is better."""
    if len(objective) not in (2, 3):
        raise ValueError(f"an objective is (value, direction) or (value, direction, tolerance), got {objective!r}")
    value, direction = objective[0], objective[1]
    tolerance = float(objective[2]) if len(objective) == 3 else 0.0
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    get = value if callable(value) else (lambda row, key=value: row[key])
    return get, (1.0 if direction == "min" else -1.0), tolerance


def _usable(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def front(rows: Sequence[dict], *objectives: Objective) -> List[dict]:
    """The rows on the Pareto front of `objectives` (see the module doc), in the order of the first objective from
    its best end (ties in the order of `rows`). Rows without a value (None or NaN) of an objective are left out."""
    if not objectives:
        raise ValueError("give at least one objective")
    parsed = [_parse(o) for o in objectives]
    points = []  # (row, [sign * value of each objective])
    for row in rows:
        values = [get(row) for get, _, _ in parsed]
        if all(_usable(v) for v in values):
            points.append((row, [sign * float(v) for v, (_, sign, _) in zip(values, parsed)]))
    tolerances = [t for _, _, t in parsed]

    def dominates(a: List[float], b: List[float]) -> bool:
        no_worse = all(x <= y + t * abs(y) for x, y, t in zip(a, b, tolerances))
        return no_worse and any(x < y for x, y in zip(a, b))

    kept = [(row, v) for row, v in points if not any(other is not row and dominates(w, v) for other, w in points)]
    return [row for row, _ in sorted(kept, key=lambda item: item[1][0])]

