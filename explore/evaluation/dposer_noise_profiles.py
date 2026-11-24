"""The noise profiles of four DPoser runs (full data, seed 10): published (d = 1), d learned by the ELBO (phase 2 of
docs/elbo_noise_plan.md), and the RSGM-like noise_scale "joint" and "iso" of the reverse test (score report, section
16). Two figures: on the SMPL-X body (vertices colored by the joint that moves them most; row 1 the axis-angle noise
per joint at std(t) = 1, row 2 log2 d rescaled to mean d^2 = 1, i.e. the shape relative to DPoser), and per axis-angle
coordinate. Also prints the correlation and slope of log d (learned) against log d (iso, joint). From the root:

    PYTHONPATH=src python explore/evaluation/dposer_noise_profiles.py docs/experiments/figures
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hp.data.amass_cfg import AmassJoints4Learning
from hp.evaluation.completion import Body
from hp.models.io import load_model
from hp.visualization.pv_render import render_overlay

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
RUNS = {
    "control (d = 1)": "dposer_posedflow_s10",
    "learned (ELBO d)": "dposer_posedflow_learned_s10",
    "joint (RSGM-like)": "dposer_posedflow_joint_s10",
    "iso (degraded)": "dposer_posedflow_iso_s10",
}
ROOT = Path("~/data/experiments/full").expanduser()
JOINTS = list(AmassJoints4Learning)
GRAY = np.array([0.82, 0.82, 0.82])

prof = {}
for label, run in RUNS.items():
    m = load_model(ROOT / run / "run_0001", device="cpu", weights="weights_ema")
    d = m.noise_d.double().numpy()
    raw = np.degrees(m.data_std.double().numpy() * d)  # axis-angle noise at std(t) = 1, degrees
    prof[label] = dict(d=d, raw=raw)

body = Body()
verts = body.model(pose_body=torch.zeros(1, 63)).v[0].detach().numpy()
faces = body.model.f.numpy()
owner = body.model.weights.argmax(1).numpy()  # SMPL-X joint that moves each vertex most
index = {j.value: i for i, j in enumerate(JOINTS)}


def colors_of(per_joint, cmap, norm):
    c = np.tile(GRAY, (len(verts), 1))
    for jv, i in index.items():
        c[owner == jv] = cmap(norm(per_joint[i]))[:3]
    return c


def render(c):
    img = render_overlay(verts, faces, colors=c, width=520, height=520, azimuths_deg=(0,))[0]
    keep = (img < 250).any(-1)
    cols, rows = np.where(keep.any(0))[0], np.where(keep.any(1))[0]
    return img[max(rows[0] - 5, 0):rows[-1] + 5, max(cols[0] - 5, 0):cols[-1] + 5]


labels = list(RUNS)
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
# row 1: absolute noise per joint (RMS over x/y/z), shared log scale
raw_j = {k: np.sqrt((v["raw"].reshape(-1, 3) ** 2).mean(1)) for k, v in prof.items()}
lo, hi = min(r.min() for r in raw_j.values()), max(r.max() for r in raw_j.values())
seq = plt.get_cmap("viridis")
seqn = matplotlib.colors.LogNorm(lo, hi)
for ax, k in zip(axes[0], labels):
    ax.imshow(render(colors_of(raw_j[k], seq, seqn)))
    ax.set_title(f"{k}\nmean d² = {np.mean(prof[k]['d'] ** 2):.2f}", fontsize=10)
    ax.axis("off")
cb = fig.colorbar(matplotlib.cm.ScalarMappable(seqn, seq), ax=axes[0], fraction=0.02, pad=0.01)
cb.set_label("axis-angle noise per joint at std(t) = 1 (deg, RMS of x/y/z)")
# row 2: shape relative to control: log2 of d normalized to mean d^2 = 1 (RMS over x/y/z in log)
div = plt.get_cmap("RdBu_r")
divn = matplotlib.colors.TwoSlopeNorm(0, -1.6, 1.6)
for ax, k in zip(axes[1], labels):
    d = prof[k]["d"] / np.sqrt(np.mean(prof[k]["d"] ** 2))
    ld = np.log2(d).reshape(-1, 3).mean(1)
    ax.imshow(render(colors_of(ld, div, divn)))
    ax.set_title(f"{k}: relative to control", fontsize=10)
    ax.axis("off")
cb = fig.colorbar(matplotlib.cm.ScalarMappable(divn, div), ax=axes[1], fraction=0.02, pad=0.01)
cb.set_label("log2 d per joint (mean over x/y/z), d rescaled to mean d² = 1\nred: more noise than DPoser, blue: less")
fig.suptitle("DPoser noise profiles (gray: joints not modeled)", fontsize=12)
fig.savefig(OUT / "dposer_noise_profiles_body.png", dpi=130, bbox_inches="tight")

# per coordinate
fig, ax = plt.subplots(figsize=(15, 4.8))
n = len(JOINTS) * 3
x = np.arange(n)
w = 0.2
style = {"control (d = 1)": "#444444", "learned (ELBO d)": "#d1495b", "joint (RSGM-like)": "#3b8bc2",
         "iso (degraded)": "#edae49"}
for i, k in enumerate(labels):
    ax.bar(x + (i - 1.5) * w, prof[k]["raw"], w, color=style[k], label=k)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([c for _ in JOINTS for c in "xyz"], fontsize=6)
for i, j in enumerate(JOINTS):
    ax.text(3 * i + 1, ax.get_ylim()[0] * 0.55, j.name, ha="center", va="top", rotation=40, fontsize=7.5)
    if i:
        ax.axvline(3 * i - 0.5, color="0.85", lw=0.6)
ax.set_xlim(-0.6, n - 0.4)
ax.set_ylabel("noise at std(t) = 1 (deg)")
ax.set_title("Axis-angle noise per coordinate (data std x d)")
ax.legend(ncol=4, fontsize=8, loc="upper right")
fig.subplots_adjust(bottom=0.25)
fig.savefig(OUT / "dposer_noise_profiles_coords.png", dpi=130, bbox_inches="tight")
logd = {k: np.log(v["d"]) for k, v in prof.items()}
for other in ("iso (degraded)", "joint (RSGM-like)"):
    a, b = logd[other] - logd[other].mean(), logd["learned (ELBO d)"] - logd["learned (ELBO d)"].mean()
    print(f"log d learned vs {other}: corr {np.corrcoef(a, b)[0, 1]:.3f}, slope {np.polyfit(a, b, 1)[0]:.3f}")
