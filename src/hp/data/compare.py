"""Compare raw AMASS sequences with their compressed version (see hp.data.compression).

For one dataset: integrity of the compressed files, how well the compressed poses cover the
original ones (nearest-neighbour recall, per-joint Wasserstein) and how much of the motion speed
is kept. Used by explore/compare_datasets.ipynb.
"""
import os
from typing import Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hp.data.compression import find_sequences, npz_len
from hp.data.explorer import _png, joint_matrices, motion_energy, nn_recall, per_joint_wasserstein, root_speed
from hp.data.amass_cfg import AmassJoints4Learning

PERCENTILES = (50, 95, 99, 99.9)


def sequences(raw_dir: str, comp_dir: str, dataset: str) -> Dict[str, List[str]]:
    """Relative paths of the sequences in both folders, and those present in only one of them."""
    raw = {rel for _, rel in find_sequences(raw_dir, [dataset])}
    comp = {rel for _, rel in find_sequences(comp_dir, [dataset])} if os.path.isdir(os.path.join(comp_dir, dataset)) else set()
    return dict(common=sorted(raw & comp), missing=sorted(raw - comp), extra=sorted(comp - raw))


def overview(raw_dir: str, comp_dir: str, datasets: List[str]) -> List[Dict]:
    """Counts per dataset from the file headers only (fast)."""
    rows = []
    for ds in datasets:
        seq = sequences(raw_dir, comp_dir, ds)
        n_orig = sum(npz_len(os.path.join(raw_dir, r)) for r in seq["common"])
        n_comp = sum(npz_len(os.path.join(comp_dir, r), "frame_index.npy") for r in seq["common"])
        fps = float(np.load(os.path.join(raw_dir, seq["common"][0]))["mocap_frame_rate"]) if seq["common"] else float("nan")
        rows.append(dict(dataset=ds, sequences=len(seq["common"]), missing=len(seq["missing"]), extra=len(seq["extra"]),
                         frames_orig=n_orig, frames_comp=n_comp, hours=n_orig / fps / 3600 if seq["common"] else 0.0))
    return rows


def verify(raw_dir: str, comp_dir: str, rels: List[str], n: int = 20, seed: int = 0) -> Dict:
    """Check that the compressed poses are exactly the original rows at frame_index."""
    rng = np.random.default_rng(seed)
    pick = [rels[i] for i in rng.choice(len(rels), min(n, len(rels)), replace=False)] if rels else []
    bad = []
    for rel in pick:
        a, b = np.load(os.path.join(raw_dir, rel), allow_pickle=True), np.load(os.path.join(comp_dir, rel), allow_pickle=True)
        idx = b["frame_index"]
        ok = np.array_equal(b["poses"], a["poses"][idx]) and np.array_equal(b["trans"], a["trans"][idx])
        ok = ok and np.isclose(float(b["mocap_frame_rate"]) * (idx[1] - idx[0] if len(idx) > 1 else 1), float(a["mocap_frame_rate"]))
        if not ok:
            bad.append(rel)
    return dict(checked=len(pick), bad=bad)


def _speeds(poses, trans, fps):
    return motion_energy(poses, fps), root_speed(trans, fps)


def dataset_report(raw_dir: str, comp_dir: str, dataset: str, n_ref: int = 2000, n_w: int = 1000,
                   n_seq: int = 100, n_verify: int = 20, seed: int = 0) -> Dict:
    """Fidelity of the compressed dataset. Recall references are frames that were dropped."""
    rng = np.random.default_rng(seed)
    seq = sequences(raw_dir, comp_dir, dataset)
    rels = seq["common"]
    if not rels:
        raise ValueError(f"No sequences of {dataset} in both {raw_dir} and {comp_dir}.")
    raw = lambda r: os.path.join(raw_dir, r)
    comp = lambda r: os.path.join(comp_dir, r)
    n_orig = np.array([npz_len(raw(r)) for r in rels])
    n_comp = np.array([npz_len(comp(r), "frame_index.npy") for r in rels])
    cum_all, cum_drop = np.cumsum(n_orig), np.cumsum(n_orig - n_comp)

    # frames to look at, as (sequence, local index) draws over the whole dataset
    def draw(cum, k):
        pos = rng.choice(int(cum[-1]), min(k, int(cum[-1])), replace=False)
        s = np.searchsorted(cum, pos, side="right")
        return s, pos - np.concatenate([[0], cum[:-1]])[s]

    ref_s, ref_rank = draw(cum_drop, n_ref)  # rank among the dropped frames of the sequence
    a_s, a_i = draw(cum_all, n_w)  # uniform over all original frames (W1 and noise floor)
    b_s, b_i = draw(cum_all, n_w)
    base_s, base_i = draw(cum_all, int(n_comp.sum()))  # random frames, same count as the compressed set

    ref, wa, wb, base, cand = [], [], [], [], []
    ref_set = set()
    for s in np.unique(np.concatenate([ref_s, a_s, b_s, base_s])):
        poses = np.load(raw(rels[s]))["poses"]
        kept = np.load(comp(rels[s]))["frame_index"]
        dropped = np.setdiff1d(np.arange(len(poses)), kept)
        ref_frames = dropped[ref_rank[ref_s == s]]
        ref_set.update((int(s), int(i)) for i in ref_frames)
        base_frames = [i for i in base_i[base_s == s] if (int(s), int(i)) not in ref_set]  # never a reference
        for out, frames in ((ref, ref_frames), (wa, a_i[a_s == s]), (wb, b_i[b_s == s]), (base, base_frames)):
            if len(frames):
                out.append(joint_matrices(poses[np.asarray(frames, dtype=int)]))
    for r in rels:
        cand.append(joint_matrices(np.load(comp(r))["poses"]))
    ref, wa, wb, base, cand = (np.concatenate(x) for x in (ref, wa, wb, base, cand))
    n_joints = ref.shape[1]

    # dynamics on a sample of sequences
    sj_o, sj_c, rs_o, rs_c = [], [], [], []
    for i in rng.choice(len(rels), min(n_seq, len(rels)), replace=False):
        a, b = np.load(raw(rels[i])), np.load(comp(rels[i]))
        eo, ro = _speeds(a["poses"], a["trans"], float(a["mocap_frame_rate"]))
        ec, rc = _speeds(b["poses"], b["trans"], float(b["mocap_frame_rate"]))
        sj_o.append(eo), sj_c.append(ec), rs_o.append(ro), rs_c.append(rc)
    sj_o, sj_c, rs_o, rs_c = (np.concatenate(x) for x in (sj_o, sj_c, rs_o, rs_c))

    return dict(
        dataset=dataset, n_seq=len(rels), missing=len(seq["missing"]), extra=len(seq["extra"]),
        n_orig=int(n_orig.sum()), n_comp=int(n_comp.sum()),
        verify=verify(raw_dir, comp_dir, rels, n_verify, seed),
        recall=nn_recall(ref, cand) / n_joints, recall_random=nn_recall(ref, base) / n_joints,
        w1=per_joint_wasserstein(wa, cand, n_w, seed), w1_floor=per_joint_wasserstein(wa, wb, n_w, seed),
        joint_speed=(sj_o, sj_c), root_speed=(rs_o, rs_c),
    )


def speed_retained(pair, q: float = 99) -> float:
    """Percentile q of the compressed speeds as a share of the original one."""
    return float(np.percentile(pair[1], q) / max(np.percentile(pair[0], q), 1e-12))


def plot_dataset_report(rep: Dict, figsize=(10, 6.5), dpi=72) -> bytes:
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(2, 2, 1)
    for key, label, color in (("recall", "compressed", "C0"), ("recall_random", "random frames, same size", "0.6")):
        d = np.sort(np.rad2deg(rep[key]))
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), label=label, color=color)
    ax.set_xlabel("distance to nearest kept pose [deg per joint]", fontsize=8)
    ax.set_ylabel("fraction of dropped frames", fontsize=8)
    ax.set_title("recall (dropped frames only)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax = fig.add_subplot(2, 2, 2)
    x = np.arange(len(rep["w1"]))
    ax.bar(x - 0.2, rep["w1"], 0.4, label="original vs compressed", color="C0")
    ax.bar(x + 0.2, rep["w1_floor"], 0.4, label="noise floor", color="0.6")
    ax.set_xticks(x)
    ax.set_xticklabels([j.name for j in AmassJoints4Learning], rotation=90, fontsize=6)
    ax.set_ylabel("W1 [deg]", fontsize=8)
    ax.set_title("per-joint Wasserstein", fontsize=9)
    ax.legend(fontsize=7)
    for k, (key, label) in enumerate((("joint_speed", "joint speed [deg/s]"), ("root_speed", "root speed [m/s]"))):
        ax = fig.add_subplot(2, 2, 3 + k)
        for arr, name, color in zip(rep[key], ("original", "compressed"), ("0.6", "C0")):
            d = np.sort(arr)
            ax.plot(d, np.arange(1, len(d) + 1) / len(d), label=name, color=color)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("fraction of frames", fontsize=8)
        ax.set_title(f"{label.split(' [')[0]} distribution", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(rep["dataset"], fontsize=10)
    fig.tight_layout()
    return _png(fig)


def overview_table(rows: List[Dict]) -> str:
    head = "".join(f"<th>{h}</th>" for h in ("dataset", "sequences", "missing", "extra", "frames original", "frames compressed", "kept", "hours"))
    body = ""
    for r in rows:
        kept = 100 * r["frames_comp"] / max(1, r["frames_orig"])
        body += (f"<tr><td>{r['dataset']}</td><td>{r['sequences']}</td><td>{r['missing']}</td><td>{r['extra']}</td>"
                 f"<td>{r['frames_orig']:,}</td><td>{r['frames_comp']:,}</td><td>{kept:.1f}%</td><td>{r['hours']:.1f}</td></tr>")
    return f"<table style='border-collapse:collapse' border=1 cellpadding=3><tr>{head}</tr>{body}</table>"


def report_table(reports: List[Dict]) -> str:
    head = "".join(f"<th>{h}</th>" for h in ("dataset", "integrity", "recall mean", "recall p95", "recall max", "W1", "W1 floor", "joint speed p99 kept", "root speed p99 kept"))
    body = ""
    for r in reports:
        deg = np.rad2deg(r["recall"])
        v = r["verify"]
        integrity = f"{v['checked'] - len(v['bad'])}/{v['checked']} ok" + (f", {r['missing']} missing" if r["missing"] else "")
        body += (f"<tr><td>{r['dataset']}</td><td>{integrity}</td><td>{deg.mean():.2f}&deg;</td><td>{np.percentile(deg, 95):.2f}&deg;</td>"
                 f"<td>{deg.max():.2f}&deg;</td><td>{r['w1'].mean():.2f}&deg;</td><td>{r['w1_floor'].mean():.2f}&deg;</td>"
                 f"<td>{100 * speed_retained(r['joint_speed']):.0f}%</td><td>{100 * speed_retained(r['root_speed']):.0f}%</td></tr>")
    return f"<table style='border-collapse:collapse' border=1 cellpadding=3><tr>{head}</tr>{body}</table>"
