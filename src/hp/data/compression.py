"""Crop and subsample raw AMASS sequences (e.g. to get a smaller dataset for quick experiments).

Every *_stageii.npz of the extracted AMASS data is cropped (seconds cut from the start and from the
end) and then reduced to every n-th frame. The output keeps the folder layout, so the preprocessing
can run on it unchanged. Command line: scripts/preprocess/compress_amass.py
"""
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib import format as npf
from tqdm import tqdm

from hp.utils.paths import get_path

# Arrays with one row per frame, and arrays copied as they are. Markers are dropped.
FRAME_KEYS = ("poses", "trans", "root_orient", "pose_body", "pose_hand", "pose_jaw", "pose_eye")
COPY_KEYS = ("betas", "gender", "surface_model_type", "num_betas")
MANIFEST = "compression.json"


def kept_frames(n: int, fps: float, stride: int, crop_start_s: float, crop_end_s: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Original frame numbers that are kept, and the kept range [a, b) before the stride.
    At least one frame is always kept, however long the crops are."""
    a = min(int(round(crop_start_s * fps)), n - 1)
    b = max(n - int(round(crop_end_s * fps)), a + 1)
    return np.arange(a, b, max(1, int(stride))), (a, b)


def compress(seq: Dict, stride: int = 1, crop_start_s: float = 0.0, crop_end_s: float = 0.0) -> Dict:
    """In-memory version for a loaded sequence (poses, trans, fps, gender). Returns the shortened
    sequence plus `index` (original frame numbers), `range` (kept [a, b)) and `stride`."""
    idx, rng = kept_frames(len(seq["poses"]), seq["fps"], stride, crop_start_s, crop_end_s)
    stride = max(1, int(stride))
    return dict(
        poses=seq["poses"][idx],
        trans=seq["trans"][idx],
        fps=seq["fps"] / stride,
        gender=seq["gender"],
        index=idx,
        range=rng,
        stride=stride,
    )


def npy_len(z: zipfile.ZipFile, name: str) -> int:
    """Length of an array of an open .npz without loading it."""
    with z.open(name) as f:
        version = npf.read_magic(f)
        read = npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0
        return read(f)[0][0]


def npz_len(path: str, name: str = "poses.npy") -> int:
    with zipfile.ZipFile(path) as z:
        return npy_len(z, name)


def compress_file(src: str, dst: str, stride: int, crop_start_s: float, crop_end_s: float) -> Tuple[int, int]:
    """Compress one *_stageii.npz. Returns the number of frames before and after."""
    data = np.load(src, allow_pickle=True)
    n, fps = len(data["poses"]), float(data["mocap_frame_rate"])
    idx, _ = kept_frames(n, fps, stride, crop_start_s, crop_end_s)
    out = {k: data[k][idx] for k in FRAME_KEYS if k in data.files}
    out.update({k: data[k] for k in COPY_KEYS if k in data.files})
    out["mocap_frame_rate"] = np.array(fps / max(1, int(stride)))
    out["frame_index"] = idx  # original frame numbers
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, **out)
    return n, len(idx)


def _job(args) -> Tuple[str, int, int]:
    dataset, src, dst, stride, cs, ce = args
    return (dataset, *compress_file(src, dst, stride, cs, ce))


def find_sequences(raw_dir: str, datasets: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """(dataset, path relative to raw_dir) of every *_stageii.npz, in sorted order."""
    datasets = sorted(datasets or [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    found = []
    for dataset in datasets:
        for root, _, files in sorted(os.walk(os.path.join(raw_dir, dataset))):
            found += [(dataset, os.path.relpath(os.path.join(root, f), raw_dir)) for f in sorted(files) if f.endswith("_stageii.npz")]
    return found


def run(cfg) -> Dict:
    """Compress the sequences described by the config (see config/compress.yaml) and write the manifest."""
    raw_dir = get_path("amass_raw", override=cfg.get("raw_dir"))
    out_dir = os.path.expanduser(cfg.output_dir)
    params = dict(stride=int(cfg.stride), crop_start_s=float(cfg.crop_start_s), crop_end_s=float(cfg.crop_end_s))
    if params["stride"] < 1 or params["crop_start_s"] < 0 or params["crop_end_s"] < 0:
        raise ValueError(f"Invalid parameters {params}: stride must be >= 1 and the crops >= 0.")
    if os.path.abspath(out_dir) == os.path.abspath(raw_dir):
        raise ValueError("output_dir must differ from the raw AMASS folder.")

    manifest_path = os.path.join(out_dir, MANIFEST)
    if os.path.isfile(manifest_path) and not cfg.get("overwrite", False):
        with open(manifest_path) as f:
            previous = json.load(f)["params"]
        if previous != params:
            raise ValueError(
                f"{out_dir} holds data compressed with {previous}, not {params}. "
                "Use another output_dir or set overwrite=true."
            )

    jobs, done = [], {}
    for dataset, rel in find_sequences(raw_dir, list(cfg.datasets) if cfg.get("datasets") else None):
        src, dst = os.path.join(raw_dir, rel), os.path.join(out_dir, rel)
        stats = done.setdefault(dataset, dict(sequences=0, frames_before=0, frames_after=0))
        if os.path.isfile(dst) and not cfg.get("overwrite", False):  # resume: count what is already there
            stats["sequences"] += 1
            stats["frames_before"] += npz_len(src)
            stats["frames_after"] += npz_len(dst, "frame_index.npy")
        else:
            jobs.append((dataset, src, dst, params["stride"], params["crop_start_s"], params["crop_end_s"]))

    workers = int(cfg.get("num_workers", 1))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(_job, jobs, chunksize=8), total=len(jobs), desc="compressing"))
    else:
        results = [_job(j) for j in tqdm(jobs, desc="compressing")]
    for dataset, before, after in results:
        done[dataset]["sequences"] += 1
        done[dataset]["frames_before"] += before
        done[dataset]["frames_after"] += after

    total = {k: sum(d[k] for d in done.values()) for k in ("sequences", "frames_before", "frames_after")}
    manifest = dict(params=params, raw_dir=raw_dir, datasets=done, total=total)
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def summary(manifest: Dict) -> str:
    rows = [f"{'dataset':22s}{'sequences':>10s}{'before':>12s}{'after':>12s}{'kept':>8s}"]
    for name, d in {**manifest["datasets"], "TOTAL": manifest["total"]}.items():
        kept = 100 * d["frames_after"] / max(1, d["frames_before"])
        rows.append(f"{name:22s}{d['sequences']:10d}{d['frames_before']:12d}{d['frames_after']:12d}{kept:7.1f}%")
    return "\n".join(rows)
