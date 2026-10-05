"""Helpers to browse raw AMASS (SMPL-X) sequences and decide what can be cropped.

Used by explore/inspect_amass.ipynb. Works on the extracted *_stageii.npz files
("amass_raw" in config/paths.yaml), where sequence boundaries are still known.
"""
import io
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from scipy.spatial.transform import Rotation

from hp.data.amass_cfg import AmassJoints4Learning
from hp.data.compression import compress, npy_len as _npy_len
from hp.utils.paths import get_path

N_BODY = 22  # root + 21 body joints in SMPL-X pose order


def list_datasets(raw_dir: str = None) -> List[str]:
    raw_dir = get_path("amass_raw", override=raw_dir)
    return sorted(d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)))


def scan_dataset(dataset: str, raw_dir: str = None) -> List[Dict]:
    """One entry per sequence: subject, name, path, n_frames, fps (header reads only)."""
    raw_dir = get_path("amass_raw", override=raw_dir)
    entries = []
    for subject in sorted(os.listdir(os.path.join(raw_dir, dataset))):
        sub_dir = os.path.join(raw_dir, dataset, subject)
        if not os.path.isdir(sub_dir):
            continue
        for name in sorted(os.listdir(sub_dir)):
            if not name.endswith("_stageii.npz"):
                continue
            path = os.path.join(sub_dir, name)
            with zipfile.ZipFile(path) as z:
                n_frames = _npy_len(z, "poses.npy")
                fps = float(np.load(z.open("mocap_frame_rate.npy")))
            entries.append(
                dict(subject=subject, name=name, path=path, n_frames=n_frames, fps=fps)
            )
    return entries


def load_sequence(path: str) -> Dict:
    d = np.load(path, allow_pickle=True)
    return dict(
        poses=d["poses"].astype(np.float64),
        trans=d["trans"].astype(np.float64),
        fps=float(d["mocap_frame_rate"]),
        gender=str(d["gender"]),
    )


def load_skeleton(body_models_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Parents and rest-pose joint positions (mean body) of the first 22 body joints."""
    path = os.path.join(get_path("body_models", override=body_models_dir), "smpl_skeleton.npz")
    skel = np.load(path)
    return skel["parents"][:N_BODY].astype(int), skel["p3d0"][0, :N_BODY].astype(np.float64)


def motion_energy(poses: np.ndarray, fps: float) -> np.ndarray:
    """Mean angular speed of the 21 body joints in deg/s, one value per frame.
    The first frame repeats the second; the root rotation is not included."""
    n = len(poses)
    if n < 2:
        return np.zeros(n)
    aa = poses[:, 3 * 1 : 3 * N_BODY].reshape(n, N_BODY - 1, 3)
    r0 = Rotation.from_rotvec(aa[:-1].reshape(-1, 3))
    r1 = Rotation.from_rotvec(aa[1:].reshape(-1, 3))
    step = (r0.inv() * r1).magnitude().reshape(n - 1, N_BODY - 1)
    speed = np.rad2deg(step.mean(axis=1)) * fps
    return np.concatenate([speed[:1], speed])


def root_speed(trans: np.ndarray, fps: float) -> np.ndarray:
    """Speed of the root translation in m/s, one value per frame."""
    n = len(trans)
    if n < 2:
        return np.zeros(n)
    speed = np.linalg.norm(np.diff(trans, axis=0), axis=1) * fps
    return np.concatenate([speed[:1], speed])


def suggest_crop(energy: np.ndarray, fps: float, thr_deg_s: float, smooth_s: float = 0.5) -> Optional[Tuple[int, int]]:
    """First and last frame where the smoothed motion energy is above the threshold."""
    if len(energy) == 0:
        return None
    k = max(1, int(round(smooth_s * fps)))
    smooth = np.convolve(energy, np.ones(k) / k, mode="same")
    active = np.flatnonzero(smooth > thr_deg_s)
    if len(active) == 0:
        return None
    return int(active[0]), int(active[-1])


def skeleton_joints(pose: np.ndarray, trans: np.ndarray, parents: np.ndarray, p3d0: np.ndarray) -> np.ndarray:
    """Forward kinematics of the 22 body joints for one frame -> [22, 3] in the AMASS world frame (z up)."""
    rot_local = Rotation.from_rotvec(pose[: 3 * N_BODY].reshape(N_BODY, 3)).as_matrix()
    rot_global = np.zeros_like(rot_local)
    joints = np.zeros((N_BODY, 3))
    for i in range(N_BODY):
        if parents[i] < 0:
            rot_global[i] = rot_local[i]
            joints[i] = p3d0[i] + trans
        else:
            p = parents[i]
            rot_global[i] = rot_global[p] @ rot_local[i]
            joints[i] = joints[p] + rot_global[p] @ (p3d0[i] - p3d0[p])
    return joints


def _kept_frame(comp: Dict, frame: int) -> Optional[int]:
    """Position in the compressed sequence shown at original frame `frame` (last kept frame so far)."""
    a, b = comp["range"]
    if frame < a or frame >= b:
        return None
    return int(np.searchsorted(comp["index"], frame, side="right") - 1)


def _png(fig: Figure) -> bytes:
    fig.canvas.draw()
    buf = io.BytesIO()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


class SkeletonView:
    """Original (gray) and compressed (blue) skeleton side by side. The figure is built once;
    png(frame) only moves the bones, which keeps playback fast. World frame, z up."""

    def __init__(self, parents: np.ndarray, p3d0: np.ndarray, figsize=(9, 4.2), dpi=72):
        self.parents, self.p3d0 = parents, p3d0
        self.fig = Figure(figsize=figsize, dpi=dpi)
        FigureCanvasAgg(self.fig)
        self.axes, self.bones = [], []
        for k, color in enumerate(("0.4", "C0")):
            ax = self.fig.add_subplot(1, 2, k + 1, projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z (up)")
            ax.set_box_aspect((1, 1, 1))
            self.bones.append([ax.plot([0, 0], [0, 0], [0, 0], "-o", ms=3, color=color)[0] for _ in range(N_BODY - 1)])
            self.axes.append(ax)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.9)

    def _pose(self, k: int, joints: np.ndarray, title: str):
        ax = self.axes[k]
        for i, line in enumerate(self.bones[k], start=1):
            a, b = joints[i], joints[self.parents[i]]
            line.set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
            line.set_visible(True)
        c = joints[0]
        ax.set_xlim(c[0] - 1, c[0] + 1)
        ax.set_ylim(c[1] - 1, c[1] + 1)
        ax.set_zlim(c[2] - 1, c[2] + 1)
        ax.set_title(title, fontsize=9)

    def png(self, orig: Dict, comp: Dict, frame: int) -> bytes:
        n, fps = len(orig["poses"]), orig["fps"]
        frame = int(np.clip(frame, 0, n - 1))
        jo = skeleton_joints(orig["poses"][frame], orig["trans"][frame], self.parents, self.p3d0)
        self._pose(0, jo, f"original, {fps:.0f} Hz   frame {frame}/{n - 1}   t = {frame / fps:.2f} s")
        k = _kept_frame(comp, frame)
        if k is None:
            for line in self.bones[1]:
                line.set_visible(False)
            self.axes[1].set_title("compressed: cropped", fontsize=9)
        else:
            jc = skeleton_joints(comp["poses"][k], comp["trans"][k], self.parents, self.p3d0)
            self._pose(1, jc, f"compressed, every {comp['stride']}th frame, {comp['fps']:.1f} Hz   frame {comp['index'][k]}")
        return _png(self.fig)


class SeriesView:
    """Joint speed and root speed over time, original vs compressed, with a cursor at the current frame."""

    def __init__(self, orig: Dict, comp: Dict, thr_deg_s: float, title: str = "", figsize=(9, 3.8), dpi=72):
        n, fps = len(orig["poses"]), orig["fps"]
        a, b = comp["range"]
        self.fps = fps
        energy_o, speed_o = motion_energy(orig["poses"], fps), root_speed(orig["trans"], fps)
        energy_c, speed_c = motion_energy(comp["poses"], comp["fps"]), root_speed(comp["trans"], comp["fps"])
        t_o, t_c = np.arange(n) / fps, comp["index"] / fps
        sug = suggest_crop(energy_o, fps, thr_deg_s)

        self.fig = Figure(figsize=figsize, dpi=dpi)
        FigureCanvasAgg(self.fig)
        axe = self.fig.add_subplot(2, 1, 1)
        axs = self.fig.add_subplot(2, 1, 2, sharex=axe)
        self.cursors = []
        for ax, yo, yc, label in ((axe, energy_o, energy_c, "joint speed [deg/s]"), (axs, speed_o, speed_c, "root speed [m/s]")):
            ax.plot(t_o, yo, color="0.6", lw=0.7, label="original")
            ax.plot(t_c, yc, color="C0", lw=1.0, marker="." if len(t_c) < 300 else None, ms=3, label="compressed")
            ax.axvspan(0, a / fps, color="red", alpha=0.12)
            ax.axvspan(b / fps, n / fps, color="red", alpha=0.12)
            self.cursors.append(ax.axvline(0, color="k", lw=1))
            ax.set_ylabel(label)
        axe.axhline(thr_deg_s, color="gray", ls="--", lw=0.8)
        if sug is not None:
            for ax in (axe, axs):
                ax.axvline(sug[0] / fps, color="green", ls=":")
                ax.axvline(sug[1] / fps, color="green", ls=":")
        axe.legend(loc="upper right", fontsize=7)
        axs.set_xlabel("time [s]")
        m = len(comp["poses"])
        self.fig.suptitle(
            f"{title}  |  {n} frames @ {fps:.0f} Hz -> {m} @ {comp['fps']:.1f} Hz ({100 * m / n:.1f}% kept)  |  "
            f"mean joint speed in kept range: original {energy_o[a:b].mean():.1f}, compressed {energy_c.mean():.1f} deg/s",
            fontsize=8,
        )
        self.fig.tight_layout()

    def png(self, frame: int) -> bytes:
        for c in self.cursors:
            c.set_xdata([frame / self.fps] * 2)
        return _png(self.fig)


# ---- Compression quality: nearest-neighbour recall and per-joint Wasserstein -------------

LEARNING_JOINTS = [j.value for j in AmassJoints4Learning]  # indices into the 22 body joints


def joint_matrices(poses: np.ndarray) -> np.ndarray:
    """[N, 165] axis-angle -> [N, 19, 3, 3] rotation matrices of the joints HuProSO models."""
    aa = np.stack([poses[:, 3 * j : 3 * j + 3] for j in LEARNING_JOINTS], axis=1)
    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(len(poses), len(LEARNING_JOINTS), 3, 3)


def nn_recall(ref: np.ndarray, cand: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """For every reference pose, the distance to its nearest candidate pose, where the distance is
    the geodesic angle summed over the joints (the HuProSO recall). ref, cand: [N, J, 3, 3]. Exact."""
    import torch

    n_ref, n_joints = ref.shape[:2]
    ref_t = torch.from_numpy(ref.reshape(n_ref, n_joints, 9)).float()
    best = torch.full((n_ref,), float("inf"))
    for start in range(0, len(cand), chunk):
        c = torch.from_numpy(cand[start : start + chunk].reshape(-1, n_joints, 9)).float()
        total = torch.zeros(n_ref, len(c))
        for j in range(n_joints):
            trace = ref_t[:, j] @ c[:, j].T  # trace(R_i^T R_k) = <R_i, R_k>
            total += torch.arccos(((trace - 1) / 2).clamp(-1, 1))
        best = torch.minimum(best, total.min(dim=1).values)
    return best.numpy()


def per_joint_wasserstein(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> np.ndarray:
    """W1 between a and b (uniform samples of n poses, geodesic cost) for every joint, in degrees."""
    from hp.metrics.similarity_measures import wasserstein_distance_so3

    rng = np.random.default_rng(seed)
    n = min(n, len(a), len(b))
    ia, ib = rng.choice(len(a), n, replace=False), rng.choice(len(b), n, replace=False)
    return np.array(
        [np.rad2deg(wasserstein_distance_so3(a[ia, j], b[ib, j])) for j in range(a.shape[1])]
    )


def gather(
    seqs: List[Dict],
    stride: int,
    crop_start_s: float,
    crop_end_s: float,
    auto_thr_deg_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation matrices of all frames (original) and of the compressed frames of the sequences.
    With `auto_thr_deg_s` each sequence is cropped by its own idle start/end instead of the fixed times."""
    orig, comp = [], []
    for seq in seqs:
        cs, ce = crop_start_s, crop_end_s
        if auto_thr_deg_s is not None:
            n, fps = len(seq["poses"]), seq["fps"]
            sug = suggest_crop(motion_energy(seq["poses"], fps), fps, auto_thr_deg_s)
            cs, ce = (0.0, 0.0) if sug is None else (sug[0] / fps, (n - 1 - sug[1]) / fps)
        c = compress(seq, stride, cs, ce)
        orig.append(joint_matrices(seq["poses"]))
        comp.append(joint_matrices(c["poses"]))
    return np.concatenate(orig), np.concatenate(comp)


def compression_report(orig: np.ndarray, comp: np.ndarray, n_ref: int = 2000, n_w: int = 1000, seed: int = 0) -> Dict:
    """NN recall and per-joint Wasserstein of the compressed poses against the original ones."""
    rng = np.random.default_rng(seed)
    ref = orig[rng.choice(len(orig), min(n_ref, len(orig)), replace=False)]
    rand = orig[rng.choice(len(orig), min(len(comp), len(orig)), replace=False)]  # same size, random frames
    n_joints = orig.shape[1]
    half = rng.permutation(len(orig))
    k = min(n_w, len(comp), len(orig) // 2)  # same sample size for W1 and its noise floor
    return dict(
        n_orig=len(orig),
        n_comp=len(comp),
        recall=nn_recall(ref, comp) / n_joints,  # radians per joint
        recall_random=nn_recall(ref, rand) / n_joints,
        w1=per_joint_wasserstein(orig, comp, k, seed),
        w1_floor=per_joint_wasserstein(orig[half[:k]], orig[half[k : 2 * k]], k, seed),
    )


def plot_report(rep: Dict, figsize=(9, 3.6), dpi=72) -> bytes:
    """CDF of the nearest-neighbour distance (compressed vs random frames of the same size) and
    per-joint Wasserstein next to its noise floor. Returns PNG bytes."""
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 2, 1)
    for key, label, color in (("recall", "compressed", "C0"), ("recall_random", "random frames, same size", "0.6")):
        d = np.sort(np.rad2deg(rep[key]))
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), label=label, color=color)
    ax.set_xlabel("distance to nearest kept pose [deg per joint]")
    ax.set_ylabel("fraction of original poses")
    ax.set_title("recall (nearest neighbour)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax = fig.add_subplot(1, 2, 2)
    x = np.arange(len(rep["w1"]))
    ax.bar(x - 0.2, rep["w1"], 0.4, label="original vs compressed", color="C0")
    ax.bar(x + 0.2, rep["w1_floor"], 0.4, label="noise floor (original vs original)", color="0.6")
    ax.set_xticks(x)
    ax.set_xticklabels([j.name for j in AmassJoints4Learning], rotation=90, fontsize=6)
    ax.set_ylabel("W1 [deg]")
    ax.set_title("per-joint Wasserstein", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _png(fig)


def report_table(rep: Dict) -> str:
    """HTML summary of the report."""

    def stats(d):
        deg = np.rad2deg(d)
        return [f"{deg.mean():.2f}", f"{np.median(deg):.2f}", f"{np.percentile(deg, 95):.2f}", f"{np.percentile(deg, 99):.2f}", f"{deg.max():.2f}", f"{100 * (deg <= 5).mean():.1f}%"]

    rows = [("compressed", stats(rep["recall"])), ("random frames, same size", stats(rep["recall_random"]))]
    head = "".join(f"<th>{h}</th>" for h in ("", "mean", "median", "p95", "p99", "max", "within 5&deg;"))
    body = "".join("<tr><td>" + name + "</td>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>" for name, vals in rows)
    return (
        f"<b>{rep['n_orig']}</b> original frames &rarr; <b>{rep['n_comp']}</b> compressed ({100 * rep['n_comp'] / rep['n_orig']:.1f}%).<br>"
        f"<b>Recall</b>: distance from an original pose to its nearest kept pose, in degrees per joint (summed distance / 19).<br>"
        f"<table style='border-collapse:collapse' border=1 cellpadding=3><tr>{head}</tr>{body}</table>"
        f"<b>Wasserstein</b> (mean over joints): original vs compressed {rep['w1'].mean():.2f}&deg;, "
        f"noise floor {rep['w1_floor'].mean():.2f}&deg;."
    )
