"""Pictures of motion denoising (hp/evaluation/denoising.py), read from its cache: nothing is recomputed. A case is one
sequence of a split at one noise level; for it, the ground truth with the noisy joints (red dots), and the denoised
bodies of every method at a few frames, the vertices coloured by their error to the ground truth (the vertices of
MPVPE, toes at rest), and the joint error of every method over the frames.

    from hp.evaluation.denoising import Body
    body = Body()
    case = denoising_case(records, ["DPoser: their full, DPoser method", "RSGM 1536x6: their c10, DPoser method"],
                          "test", 0.04, sequence=12, body=body)
    fig = denoising_figure(case, render_case(case, body, frames=(0, 20, 40, 59)))
"""
import textwrap
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hp.evaluation.completion import TOES, at_rest
from hp.evaluation.denoising import Body, gt_joints, sequences
from hp.evaluation.denoising_results import find

MAX_ERROR = 100.0  # mm: the top of the colour scale of the vertex errors
GT_COLOR = (0.56, 0.72, 0.93)  # blue, as the visible part of the completion figures


def noisy_sequence(joints: torch.Tensor, sigma: float, index: int) -> torch.Tensor:
    """The noisy joints [T, 22, 3] of sequence `index` (the draw of hp.evaluation.denoising.noisy_joints)."""
    g = torch.Generator().manual_seed(index)
    return joints.float() + sigma * torch.randn(joints.shape, generator=g)


def denoising_case(records: Sequence[dict], labels: Sequence[str], split: str, sigma: float, sequence: int,
                   body: Body, sequences_file: str = "~/data/experiments/denoising/sequences.pt") -> Dict:
    """One sequence: the ground truth [T, 21, 3, 3], its joints and noisy joints [T, 22, 3], and per method (cache
    record of the label; missing ones left out) the denoised poses and the per-frame MPJPE and MPVPE (mm)."""
    gt_all, names = sequences(split, sequences_file)
    gt = gt_all[sequence]
    joints = gt_joints(gt[None], body)[0]
    methods = []
    for label in labels:
        r = find(records, label, sigma, split)
        if r is None:
            continue
        if r["names"] != names:
            raise ValueError(f"{label}: the cache was made on other sequences of '{split}'.")
        saved = torch.load(r["file"], weights_only=True)
        methods.append({"label": label, "poses": saved["pred"][sequence].float(),
                        "mpjpe": saved["metrics"]["mpjpe"][sequence].double().numpy(),
                        "mpvpe": saved["metrics"]["mpvpe"][sequence].double().numpy()})
    noisy = noisy_sequence(joints, sigma, sequence)
    return {"split": split, "sigma": sigma, "sequence": sequence, "name": names[sequence], "gt": gt,
            "joints": joints, "noisy": noisy, "input": (noisy.double() - joints).norm(dim=-1).mean(-1).numpy() * 1000,
            "methods": methods}


def vertex_errors(poses: torch.Tensor, gt: torch.Tensor, body: Body) -> np.ndarray:
    """[T, V] distance (mm) of every vertex to the ground truth, toes at rest (as MPVPE)."""
    v, _ = body(at_rest(poses, TOES))
    v_gt, _ = body(at_rest(gt, TOES))
    return ((v - v_gt).norm(dim=-1) * 1000).numpy()


def error_colors(errors: np.ndarray, cmap: str = "magma_r", top: float = MAX_ERROR) -> np.ndarray:
    """Per-vertex RGB [..., V, 3] of errors in mm, light for 0 and dark for `top` and more."""
    import matplotlib

    return matplotlib.colormaps[cmap](np.clip(errors / top, 0, 1) * 0.85)[..., :3]


def render_case(case: Dict, body: Body, frames: Sequence[int] = (0, 15, 30, 45, 59), azimuth: float = 0,
                size=(260, 380), top: float = MAX_ERROR) -> Dict:
    """Images [frames, h, w, 3] of the ground truth (blue, the noisy joints as red dots) and of every method (vertices
    coloured by their error, up to `top` mm), each frame framed alike in every row (one scale per frame)."""
    from hp.visualization.pv_render import render_overlay

    frames = list(frames)
    faces = body.model.f.detach().cpu().numpy()
    gt_v = body(case["gt"][frames])[0].numpy()
    meshes = {m["label"]: body(m["poses"][frames])[0].numpy() for m in case["methods"]}
    colors = {m["label"]: error_colors(vertex_errors(m["poses"][frames], case["gt"][frames], body), top=top)
              for m in case["methods"]}
    out = {"frames": frames, "gt": [], "methods": {label: [] for label in meshes}}
    for i, f in enumerate(frames):
        every = np.concatenate([gt_v[i][None]] + [v[i][None] for v in meshes.values()])
        lo, hi = every.reshape(-1, 3).min(0), every.reshape(-1, 3).max(0)
        frame = dict(center=(hi + lo) / 2, radius=np.linalg.norm(hi - lo) / 2 * 1.08)
        common = dict(width=size[0], height=size[1], azimuths_deg=(azimuth,), **frame)
        out["gt"].append(render_overlay(gt_v[i], faces, colors=np.tile(GT_COLOR, (len(gt_v[i]), 1)),
                                        points=case["noisy"][f].numpy(), point_radius=0.025, point_opacity=0.9,
                                        **common)[0])
        for label, v in meshes.items():
            out["methods"][label].append(render_overlay(v[i], faces, colors=colors[label][i], **common)[0])
    out["gt"] = np.stack(out["gt"])
    out["methods"] = {k: np.stack(v) for k, v in out["methods"].items()}
    return out


def denoising_figure(case: Dict, images: Dict, names: Optional[Dict[str, str]] = None, curves: bool = True,
                     panel_width: float = 1.25, top: float = MAX_ERROR):
    """A matplotlib figure: a line for the ground truth with the noisy joints and one per method (its MPJPE and MPVPE
    on this sequence on the left, the frames in columns), the colour bar of the vertex errors, and below, if
    `curves`, the joint error of every method and of the input over the frames."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    from hp.visualization.completion_figure import crop

    names = names or {}
    rows = [("Ground truth,\nnoisy joints", images["gt"], f"input MPJPE {case['input'].mean():.1f} mm")]
    for m in case["methods"]:
        if m["label"] in images["methods"]:
            rows.append((textwrap.fill(names.get(m["label"], m["label"]), 24), images["methods"][m["label"]],
                         f"MPJPE {m['mpjpe'].mean():.1f}, MPVPE {m['mpvpe'].mean():.1f} mm"))
    stack = crop(np.stack([r[1] for r in rows]).transpose(1, 0, 2, 3, 4))  # one crop per frame
    n_frames = len(images["frames"])
    h, w = stack.shape[-3:-1]
    margin = 2.2
    pic_h = panel_width * h / w
    curve_h = 2.2 if curves else 0
    with plt.style.context("default"), plt.rc_context({"figure.facecolor": "white", "savefig.facecolor": "white",
                                                       "text.color": "black"}):
        fig = plt.figure(figsize=(panel_width * n_frames + margin + 0.6, pic_h * len(rows) + curve_h + 0.9))
        total_h = pic_h * len(rows) + curve_h + 0.9
        grid = fig.add_gridspec(len(rows) + (1 if curves else 0), n_frames + 1,
                                height_ratios=[pic_h] * len(rows) + ([curve_h] if curves else []),
                                width_ratios=[panel_width] * n_frames + [0.12],
                                left=margin / (panel_width * n_frames + margin + 0.6), right=0.93,
                                top=1 - 0.45 / total_h, bottom=0.6 / total_h, hspace=0.12, wspace=0.03)
        for r, (name, _, numbers) in enumerate(rows):
            for c in range(n_frames):
                ax = fig.add_subplot(grid[r, c])
                ax.imshow(stack[c, r])
                ax.axis("off")
                if r == 0:
                    ax.set_title(f"frame {images['frames'][c]}", fontsize=8)
                if c == 0:
                    ax.text(-0.08, 0.52, name, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                            fontweight="bold")
                    ax.text(-0.08, 0.48, numbers, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                            color="0.3")
        bar = fig.add_subplot(grid[1:len(rows), n_frames])
        norm = colors.Normalize(0, top)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colors.ListedColormap(error_colors(np.linspace(0, top, 256),
                                                                                         top=top))),
                     cax=bar, label="vertex error (mm)")
        if curves:
            ax = fig.add_subplot(grid[len(rows), :n_frames])
            ax.plot(case["input"], color="0.6", lw=1, label="input (noisy joints)")
            for m in case["methods"]:
                ax.plot(m["mpjpe"], lw=1.5, label=names.get(m["label"], m["label"]))
            for f in images["frames"]:
                ax.axvline(f, color="0.85", lw=0.8, zorder=0)
            ax.set_xlabel("frame")
            ax.set_ylabel("MPJPE (mm)")
            ax.set_xlim(0, len(case["input"]) - 1)
            ax.legend(fontsize=7, ncol=2, frameon=False)
            ax.grid(alpha=0.3)
        fig.suptitle(f"{case['name']}  ({case['split']}, sequence {case['sequence']}, "
                     f"noise {round(case['sigma'] * 1000)} mm)", fontsize=9, y=1 - 0.1 / total_h)
    return fig


def per_frame_errors(records: Sequence[dict], labels: Sequence[str], split: str, sigma: float,
                     metric: str = "mpjpe") -> List[Optional[np.ndarray]]:
    """[N, T] errors of each label (None if not cached), for hp.visualization.completion_figure.pick_pose."""
    out = []
    for label in labels:
        r = find(records, label, sigma, split)
        out.append(None if r is None else torch.load(r["file"], weights_only=True)["metrics"][metric].double().numpy())
    return out
