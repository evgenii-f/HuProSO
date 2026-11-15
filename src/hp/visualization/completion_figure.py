"""Pictures of pose completions (as Fig. 4 of PoseRFM): the visible part of the body in blue, the hidden part, filled
by a model, in pink; K hypotheses of every method next to the ground truth.

The completions are those of the test protocol (hp/evaluation/completion.py), read from its cache: nothing is
recomputed, so any (method, occlusion) that scripts/evaluate_completion.py has run can be shown. The pink part is the
hidden part of the metric (the vertices of MPVPE, their smplx_vert_segmentation.json), so the picture shows what is
measured. Rendering: PyVista (hp/visualization/pv_render.py), SMPL-X neutral with betas = 0 (the body of the metrics).

    from hp.evaluation.completion import Body
    body = Body()
    rows = completion_rows(CACHE, ["DPoser: their full, DPoser method", "RSGM 512x4: their c10"], "legs", pose=123,
                           k=4, body=body)
    fig = completion_figure([rows], body)
"""
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hp.evaluation.completion import OCCLUSIONS, Body, cache_path

VISIBLE = (0.56, 0.72, 0.93)  # blue, as PoseRFM's figure
HIDDEN = (0.95, 0.64, 0.84)   # pink
GT = "ground truth"
CHOICES = ("first", "diverse", "best to worst")  # which K of the hypotheses of a pose are shown
PICKS = ("random", "hard for 1", "easy for 1", "1 better than 2", "2 better than 1")  # how a test pose is picked


def cached_labels(cache) -> List[str]:
    """The methods with completions in the cache: their labels (the folder names, "_" back to spaces)."""
    root = Path(cache).expanduser()
    if not root.is_dir():
        return []
    return sorted(p.name.replace("_", " ") for p in root.iterdir()
                  if p.is_dir() and any((p / f"{o}.pt").is_file() for o in OCCLUSIONS))


def load_completions(cache, label: str, occlusion: str) -> Optional[torch.Tensor]:
    """The cached completions [N, K, 21, 3, 3] of a method and an occlusion (None if not computed). Read as they are:
    no check of the settings they were made with (cached_run does that)."""
    path = cache_path(Path(cache), label, occlusion)
    return torch.load(path, weights_only=True)["pred"].float() if path.is_file() else None


def load_metrics(cache, label: str, occlusion: str) -> Optional[Dict[str, np.ndarray]]:
    """The cached metrics per pose of cached_metrics (None if not computed): mpvpe [N, K] (mm) and apd [N] (cm, over
    all K hypotheses of a pose)."""
    path = cache_path(Path(cache), label, occlusion).with_name(f"{occlusion}_metrics.pt")
    if not path.is_file():
        return None
    plain = torch.load(path, weights_only=True)["plain"]
    return {key: plain[key].numpy() for key in ("mpvpe", "apd") if key in plain}


def load_errors(cache, label: str, occlusion: str) -> Optional[np.ndarray]:
    """The cached MPVPE per pose and hypothesis [N, K] (mm), as computed by cached_metrics (None if not computed)."""
    metrics = load_metrics(cache, label, occlusion)
    return None if metrics is None else metrics["mpvpe"]


def method_summary(cache, label: str, occlusion: str) -> Optional[Dict[str, float]]:
    """The numbers of the table for a method and an occlusion, over all test poses (None if not computed): "mpvpe"
    (mean over the K hypotheses), "apd" (cm) and "best" (best of K), as summary() of hp.evaluation.completion."""
    metrics = load_metrics(cache, label, occlusion)
    if metrics is None:
        return None
    return {"mpvpe": float(metrics["mpvpe"].mean()), "best": float(metrics["mpvpe"].min(1).mean()),
            "apd": float(metrics["apd"].mean()) if "apd" in metrics else float("nan")}


def method_option(cache, label: str, occlusion: str) -> str:
    """The averages of a method over the test poses, then its label, for a list of methods to pick from: the numbers
    first, padded with figure spaces (as wide as a digit), so that they line up whatever the length of the labels."""
    s = method_summary(cache, label, occlusion)
    if s is None:
        numbers = "MPVPE  –  ·  APD  –  ·  best of 10  –"
        return f"{numbers}   {label}   (not computed for {occlusion})"
    pad = lambda x, spec: format(x, spec).replace(" ", " ")
    return (f"MPVPE {pad(s['mpvpe'], '6.1f')}  ·  APD {pad(s['apd'], '5.2f')}  ·  best of 10 {pad(s['best'], '6.1f')}"
            f"   {label}")


def hidden_vertices(body: Body, occlusion: str) -> np.ndarray:
    """A mask [V] of the vertices of the hidden part: those of the MPVPE of the occlusion."""
    mask = np.zeros(body.model.init_v_template.shape[1], dtype=bool)
    mask[body.vertices[occlusion]] = True
    return mask


def vertex_colors(mask: np.ndarray) -> np.ndarray:
    """RGB [V, 3]: the hidden vertices pink, the others blue."""
    return np.where(mask[:, None], np.array(HIDDEN), np.array(VISIBLE))


def choose(poses: torch.Tensor, k: int, how: str, errors: Optional[np.ndarray] = None, body: Optional[Body] = None):
    """Indices of k of the K hypotheses [K, 21, 3, 3] of one pose: "first" the first k; "diverse" farthest point
    sampling on the joints (the first one is the hypothesis closest to the mean of all); "best to worst" k spread
    over the hypotheses sorted by their error (`errors` [K]), the best one first."""
    n = len(poses)
    k = min(k, n)
    if how == "first":
        return list(range(k))
    if how == "best to worst":
        if errors is None:
            raise ValueError("'best to worst' needs the errors of the hypotheses.")
        order = np.argsort(errors)
        return [int(order[i]) for i in np.round(np.linspace(0, n - 1, k)).astype(int)]
    if how == "diverse":
        joints = body(poses)[1].reshape(n, -1).numpy()
        picked = [int(np.argmin(((joints - joints.mean(0)) ** 2).sum(1)))]
        distance = ((joints - joints[picked[0]]) ** 2).sum(1)
        while len(picked) < k:
            picked.append(int(np.argmax(distance)))
            distance = np.minimum(distance, ((joints - joints[picked[-1]]) ** 2).sum(1))
        return picked
    raise ValueError(f"Unknown choice '{how}'. Use one of {CHOICES}.")


def pick_pose(how: str, errors: Sequence[Optional[np.ndarray]], rng: np.random.Generator, top: int = 50) -> int:
    """A test pose: "random"; "hard for 1" / "easy for 1" one of the `top` poses with the highest / lowest mean error
    of method 1; "1 better than 2" / "2 better than 1" one of the `top` poses where the mean error of method 1 is the
    most below / above that of method 2. errors: the [N, K] MPVPE of the methods (load_errors)."""
    known = [e for e in errors if e is not None]
    if how == "random" or not known:
        n = len(known[0]) if known else 5000
        return int(rng.integers(n))
    mean = [None if e is None else e.mean(1) for e in errors]
    if how in ("hard for 1", "easy for 1"):
        if mean[0] is None:
            raise ValueError("method 1 has no metrics for this occlusion.")
        order = np.argsort(mean[0])
        candidates = order[-top:] if how == "hard for 1" else order[:top]
    elif how in ("1 better than 2", "2 better than 1"):
        if len(mean) < 2 or mean[0] is None or mean[1] is None:
            raise ValueError("comparing needs the metrics of methods 1 and 2.")
        gap = mean[0] - mean[1]
        order = np.argsort(gap)
        candidates = order[:top] if how == "1 better than 2" else order[-top:]
    else:
        raise ValueError(f"Unknown pick '{how}'. Use one of {PICKS}.")
    return int(rng.choice(candidates))


def completion_rows(cache, labels: Sequence[str], occlusion: str, pose: int, k: int, body: Body,
                    gt: Optional[torch.Tensor] = None, how: str = "first") -> Dict:
    """What a row of the figure shows: per method the chosen hypotheses of test pose `pose`, their errors, and the
    ground truth (from `gt` [N, 21, 3, 3], the test poses, if given). Returns {"occlusion", "pose", "methods":
    [{"label", "poses" [k, 21, 3, 3], "errors" [k] or None, "indices", "mpvpe" (mean), "best" (min) and "apd" of the
    pose over all K hypotheses (mm, cm; None if the metrics are not cached)}], "gt": [21, 3, 3] or None}."""
    methods = []
    for label in labels:
        pred = load_completions(cache, label, occlusion)
        if pred is None:
            methods.append({"label": label, "poses": None, "errors": None, "indices": [], "mpvpe": None, "best": None,
                            "apd": None})
            continue
        metrics = load_metrics(cache, label, occlusion)
        e = None if metrics is None else metrics["mpvpe"][pose]
        idx = choose(pred[pose], k, how, e, body)
        methods.append({"label": label, "poses": pred[pose, idx], "errors": None if e is None else e[idx],
                        "indices": idx, "mpvpe": None if e is None else float(e.mean()),
                        "best": None if e is None else float(e.min()),
                        "apd": None if metrics is None or "apd" not in metrics else float(metrics["apd"][pose])})
    return {"occlusion": occlusion, "pose": pose, "methods": methods, "gt": None if gt is None else gt[pose]}


def render_row(row: Dict, body: Body, azimuths=(0,), size=(300, 420), ghost: bool = False) -> Dict:
    """Images of a row (completion_rows): per method [k, views, h, w, 3] and the ground truth [views, h, w, 3], with
    one framing for the whole row (the same scale in every picture). ghost: the ground truth drawn faintly in grey
    behind every hypothesis."""
    from hp.visualization.pv_render import render_overlay

    faces = body.model.f.detach().cpu().numpy()
    colors = vertex_colors(hidden_vertices(body, row["occlusion"]))
    meshes = {}
    for m in row["methods"]:
        if m["poses"] is not None:
            meshes[m["label"]] = body(m["poses"])[0].numpy()
    gt_mesh = None if row["gt"] is None else body(row["gt"][None])[0].numpy()[0]
    every = np.concatenate([*meshes.values()] + ([gt_mesh[None]] if gt_mesh is not None else []))
    lo, hi = every.reshape(-1, 3).min(0), every.reshape(-1, 3).max(0)
    frame = dict(center=(hi + lo) / 2, radius=np.linalg.norm(hi - lo) / 2)
    width, height = size

    def draw(vertices):
        if ghost and gt_mesh is not None:
            return render_overlay(np.stack([gt_mesh, vertices]), faces, opacities=[0.25, 1.0], width=width,
                                  height=height, azimuths_deg=azimuths,
                                  colors=np.stack([np.full_like(colors, 0.7), colors]), **frame)
        return render_overlay(vertices, faces, width=width, height=height, azimuths_deg=azimuths, colors=colors,
                              **frame)

    out = {"methods": {label: np.stack([draw(v) for v in verts]) for label, verts in meshes.items()}}
    out["gt"] = None if gt_mesh is None else render_overlay(gt_mesh, faces, width=width, height=height,
                                                            azimuths_deg=azimuths, colors=colors, **frame)
    return out


def crop(images: np.ndarray, margin: float = 0.03) -> np.ndarray:
    """The same crop of the white background for all images [..., h, w, 3], so that they keep their scale."""
    used = (images.reshape(-1, *images.shape[-3:]) < 250).any(-1).any(0)
    rows, cols = np.where(used.any(1))[0], np.where(used.any(0))[0]
    if len(rows) == 0:
        return images
    h, w = used.shape
    r0, r1 = max(rows[0] - int(margin * h), 0), min(rows[-1] + int(margin * h) + 1, h)
    c0, c1 = max(cols[0] - int(margin * w), 0), min(cols[-1] + int(margin * w) + 1, w)
    return images[..., r0:r1, c0:c1, :]


def completion_figure(rows: Sequence[Dict], images: Sequence[Dict], names: Optional[Dict[str, str]] = None,
                      errors: bool = True, title: Optional[str] = None, layout: str = "rows",
                      panel_width: float = 1.3):
    """A matplotlib figure of the cases `rows` (completion_rows) and their pictures `images` (render_row); the views
    of a pose stacked vertically in its column, the MPVPE of every hypothesis under it if `errors`, `names` shorter
    names for the labels.

    layout "rows": every method on a line of its own (its hypotheses, then the ground truth), named on the left, the
    case (occlusion, pose) above its lines. "columns" (as Fig. 4 of PoseRFM): one line per case, the hypotheses of
    the methods side by side and the ground truth last, the methods named below. With `errors`, the name of a method
    carries the MPVPE (mean over all K hypotheses of the pose) and the APD (over all K) of the pose."""
    import matplotlib.pyplot as plt

    # a paper look whatever the theme of the notebook (a dark theme would hide the dark text on a dark background)
    with plt.style.context("default"), plt.rc_context({"figure.facecolor": "white", "savefig.facecolor": "white",
                                                       "text.color": "black"}):
        return _completion_figure(rows, images, names, errors, title, layout, panel_width)


def _completion_figure(rows, images, names, errors, title, layout, panel_width):
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    if layout not in ("rows", "columns"):
        raise ValueError(f"Unknown layout '{layout}'. Use 'rows' or 'columns'.")
    names = names or {}
    gt_name = "Ground truth (GT)"

    def named(m, sep):
        name = textwrap.fill(names.get(m["label"], m["label"]), 26)
        if not errors or m["mpvpe"] is None:
            return name
        return f"{name}{sep}MPVPE {m['mpvpe']:.0f} mm, APD {m['apd']:.1f} cm" if m["apd"] is not None else name

    def label_left(ax, text):  # the name in bold, the metrics of the pose (after the last line break) under it
        name, _, numbers = text.rpartition("\x00") if "\x00" in text else (text, "", "")
        ax.text(-0.08, 0.52, name, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")
        if numbers:
            ax.text(-0.08, 0.48, numbers, transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.3")

    lines = []  # per line: pictures [(image [views, h, w, 3], text under it, group)], name on the left, header
    for row, im in zip(rows, images):
        shown = [m for m in row["methods"] if m["label"] in im["methods"]]
        pictures = {m["label"]: [(img, "" if (not errors or m["errors"] is None) else f"{m['errors'][i]:.0f} mm",
                                  named(m, "\n")) for i, img in enumerate(im["methods"][m["label"]])]
                    for m in shown}
        gt = [] if im["gt"] is None else [(im["gt"], "GT" if layout == "rows" else "", gt_name)]
        case = f"{row['occlusion'].replace('_', ' ')}, pose {row['pose']}"
        if layout == "rows":
            for j, m in enumerate(shown):
                lines.append((pictures[m["label"]] + gt, named(m, "\x00"), case if j == 0 else None))
        else:
            lines.append(([p for m in shown for p in pictures[m["label"]]] + gt, case.replace(", ", "\n"), None))
    if not lines:
        raise ValueError("nothing to draw: no method has completions for these cases.")
    # one crop per case, so that every picture of a case has the same scale
    cropped, k = [], 0
    for row, im in zip(rows, images):
        n = sum(1 for m in row["methods"] if m["label"] in im["methods"]) if layout == "rows" else 1
        block = lines[k:k + n]
        widths = [len(line[0]) for line in block]
        stack = crop(np.concatenate([np.stack([p[0] for p in line[0]]) for line in block]))
        for line, (a, b) in zip(block, zip(np.cumsum([0] + widths[:-1]), np.cumsum(widths))):
            cropped.append(stack[a:b])
        k += n
    n_cols = max(len(line[0]) for line in lines)
    # the left margin fits the longest line of the names on the left (inches, ~0.075 per character at 9 pt)
    longest = max(len(part) for line in lines for part in line[1].replace("\x00", "\n").split("\n"))
    margin = max(1.0, 0.075 * longest + 0.3)
    views, h, w = cropped[0].shape[1:4]
    aspect = h * views / w
    header = 0.25 if layout == "rows" else 0.0
    fig, axes = plt.subplots(len(lines), n_cols, squeeze=False,
                             figsize=(panel_width * n_cols + margin, (panel_width * aspect + header) * len(lines) + 0.4))
    fig.subplots_adjust(left=margin / (panel_width * n_cols + margin), wspace=0.02,
                        hspace=(0.45 if layout == "rows" else 0.35) / views)  # before anything placed by the axes positions
    for r, ((pictures, left_name, case), images_r) in enumerate(zip(lines, cropped)):
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c >= len(pictures):
                continue
            ax.imshow(np.concatenate(list(images_r[c]), axis=0))
            ax.text(0.5, 0.0, pictures[c][1], transform=ax.transAxes, ha="center", va="top", fontsize=8)
        if layout == "rows":
            label_left(axes[r][0], left_name)
        else:
            axes[r][0].text(-0.08, 0.5, left_name, transform=axes[r][0].transAxes, ha="right", va="center", fontsize=9)
        if case:
            axes[r][0].text(0.0, 1.06, case, transform=axes[r][0].transAxes, ha="left", va="bottom", fontsize=9,
                            color="0.35")
        if pictures and pictures[-1][2] == gt_name and len(pictures) > 1:  # a thin line before the ground truth
            x = (axes[r][len(pictures) - 2].get_position().x1 + axes[r][len(pictures) - 1].get_position().x0) / 2
            box = axes[r][0].get_position()
            fig.add_artist(plt.Line2D([x, x], [box.y0, box.y1], color="0.85", lw=0.8))
        if layout == "columns":  # the methods named under their groups, thin lines between them
            start = 0
            for c in range(1, len(pictures) + 1):
                if c == len(pictures) or pictures[c][2] != pictures[start][2]:
                    a, b = axes[r][start].get_position(), axes[r][c - 1].get_position()
                    under = blended_transform_factory(fig.transFigure, axes[r][start].transAxes)
                    axes[r][start].text((a.x0 + b.x1) / 2, -0.11, pictures[start][2], transform=under,
                                        ha="center", va="top", fontsize=9, fontweight="bold", color="0.2")
                    if 0 < c < len(pictures) - 1:
                        x = (b.x1 + axes[r][c].get_position().x0) / 2
                        fig.add_artist(plt.Line2D([x, x], [a.y0, a.y1], color="0.85", lw=0.8))
                    start = c
    if title:
        fig.suptitle(title, fontsize=10)
    return fig
