"""Drawing Pareto fronts (hp.evaluation.fronts) with matplotlib, shared by the notebooks of the comparisons
(pareto_posedflow_vs_rsgm.ipynb, completion_fronts.ipynb): a front as a staircase, lines or rings; labels of the points
placed so as not to overlap; the points of one model highlighted.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

STYLES = {"steps": "steps: the best within a budget", "lines": "lines between the front points",
          "points": "points only: a ring on each front point"}


def draw_front(ax, x: Sequence[float], y: Sequence[float], style: str = "steps", color="0.25", ls="--", lw: float = 1.4,
               ring: float = 15, corner: str = "y"):
    """One front through the points (x[i], y[i]) in the given order (that of hp.evaluation.fronts.front): "steps" a
    staircase, from each point along `corner` ("y": vertically, "x": horizontally) to the level of the next, then
    across to it; "lines" straight lines; "points" a ring of size `ring` on each point."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {list(STYLES)}, got {style!r}")
    x, y = list(x), list(y)
    if style == "points":
        ax.plot(x, y, marker="o", ms=ring, ls="", mfc="none", mec=color, mew=1, zorder=1)
        return
    if style == "steps" and x:
        if corner == "y":
            x, y = [v for a in x for v in (a, a)][:-1], [y[0]] + [v for b in y[1:] for v in (b, b)]
        else:
            x, y = [x[0]] + [v for a in x[1:] for v in (a, a)], [v for b in y for v in (b, b)][:-1]
    ax.plot(x, y, ls=ls, lw=lw, color=color, zorder=1)


def overlap(a, b) -> float:
    """The area shared by two boxes (x0, y0, x1, y1)."""
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def place_labels(fig, labels: Dict[object, List[Tuple[float, float, str]]], fontsize: float = 7.5):
    """Put the labels {ax: [(x, y, text)]} near their points, greedily: the most crowded points first, each label at the
    nearest place free of the other labels and of the markers, joined to its point by a thin line if it is far. A text
    None is a point without a label, only in the way of the others."""
    renderer, px = fig.canvas.get_renderer(), fig.dpi / 72  # pixels per point
    style = dict(fontsize=fontsize, color="0.15", textcoords="offset points", ha="left", va="bottom")
    for ax, items in labels.items():
        frame = ax.get_window_extent(renderer).extents
        points = [ax.transData.transform((x, y)) for x, y, _ in items]
        markers = [(p[0] - 5 * px, p[1] - 5 * px, p[0] + 5 * px, p[1] + 5 * px) for p in points]
        taken = [t.get_window_extent(renderer).extents for t in ax.texts]  # texts already there
        strokes = []  # points along the drawn lines (the fronts), every 3 pixels: a label should not cross them
        for line in ax.get_lines():
            if line.get_linestyle() in ("None", "", " "):
                continue
            xy = ax.transData.transform(np.column_stack(line.get_data()))
            for a, b in zip(xy[:-1], xy[1:]):
                steps = max(1, int(np.hypot(*(b - a)) / (3 * px)))
                strokes += [a + (b - a) * s for s in np.linspace(0, 1, steps + 1)]
        strokes = np.array(strokes).reshape(-1, 2)
        crowd = [sum(np.hypot(*(p - q)) < 40 * px for q in points) for p in points]
        for i in sorted(range(len(items)), key=lambda i: -crowd[i]):
            x, y, text = items[i]
            if text is None:
                continue
            probe = ax.annotate(text, (x, y), xytext=(0, 0), **style)
            box = probe.get_window_extent(renderer)
            width, height = box.width, box.height
            probe.remove()
            best = None
            for radius in (7, 11, 16, 22, 30, 40, 52, 66, 82, 100, 120):
                for angle in np.deg2rad(np.arange(0, 360, 15)):
                    cos, sin = np.cos(angle), np.sin(angle)
                    x0 = points[i][0] + radius * px * cos + (0 if cos > 0.3 else -width if cos < -0.3 else -width / 2)
                    y0 = points[i][1] + radius * px * sin + (0 if sin > 0.3 else -height if sin < -0.3 else -height / 2)
                    candidate = (x0, y0, x0 + width, y0 + height)
                    inside = (candidate[0] >= frame[0] and candidate[1] >= frame[1] and candidate[2] <= frame[2]
                              and candidate[3] <= frame[3])
                    area = width * height
                    cost = radius + 2000 * sum(overlap(candidate, t) for t in taken) / area \
                        + 500 * sum(overlap(candidate, m) for m in markers) / area + (0 if inside else 5000) \
                        + 40 * np.sum((strokes[:, 0] > candidate[0]) & (strokes[:, 0] < candidate[2])
                                      & (strokes[:, 1] > candidate[1]) & (strokes[:, 1] < candidate[3]))
                    if best is None or cost < best[0]:
                        best = (cost, candidate, radius)
            _, candidate, radius = best
            offset = ((candidate[0] - points[i][0]) / px, (candidate[1] - points[i][1]) / px)
            line = dict(arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6, shrinkA=0, shrinkB=3)) if radius > 16 else {}
            ax.annotate(text, (x, y), xytext=offset, zorder=4, **style, **line)
            taken.append(candidate)


def highlight(hits, name):
    """Ring the points of `name` on every plot and fade the others (name None: all as drawn); hits are (artist, x, y,
    name) of the drawn points. Only changes the artists, so a click does not lay out the figure again. Returns the
    rings, to remove them at the next call."""
    rings = []
    for artist, x, y, point_name in hits:
        artist.set_alpha(1 if name is None or point_name == name else 0.25)
        if point_name == name:
            rings += artist.axes.plot(*artist.get_data(), marker="o", ms=22, ls="", mfc="none", mec="black", mew=2,
                                      zorder=5)
    return rings
