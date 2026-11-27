"""Verify Stage 1 joins, signal support, independent partitions and optional embedding galleries."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from .cache import RATE, SECONDS, VISION_REV, env_path, file_hash, read_rows
from .partition import normalize_window
from .prepare import continuous_eeg


def verify_split(folder: Path, base_hash: str, expected: set, records: dict) -> dict:
    """Check membership, participant isolation and train-only statistics for a saved partition."""
    spec = json.loads((folder / "split.json").read_text())
    groups = spec["groups"]
    if spec["cache_sha256"] != base_hash:
        raise ValueError(f"Split refers to a different signal/target cache: {folder}")
    assignments = read_rows(folder / "assignments.csv")
    keys = set()
    for a in assignments:
        key = a["recording_id"], a["moment_id"]
        group = records[key[0]]["group"]
        if key in keys or a["group"] != group or a["role"] != groups[group]:
            raise ValueError(f"Repeated or inconsistent group assignment: {key}")
        keys.add(key)
    if keys != expected or {a["role"] for a in assignments} != {"train", "val", "test"}:
        raise ValueError(f"Unexpected fold membership in {folder}")
    with np.load(folder / "train_stats.npz") as stats:
        ntrain = sum(a["role"] == "train" for a in assignments)
        if int(stats["samples"]) != ntrain * RATE * SECONDS or not np.all(stats["std_uv"] > 0):
            raise ValueError(f"Invalid training statistics in {folder}")
    return groups


def verify(root: Path, eeg: Path, stimuli: Path, base_only: bool) -> dict:
    """Validate every query, reconstruct one source pair per film and check every available target row."""
    base = root / "base"
    info = json.loads((base / "cache.json").read_text())
    base_hash = file_hash(base / "cache.json")
    for name, digest in info["manifest_hashes"].items():
        if file_hash(base / name) != digest:
            raise ValueError(f"Changed base manifest: {name}")
    channels = json.loads((base / "channels.json").read_text())
    positions = np.load(base / "positions.npy")
    if len(set(channels)) != 64 or positions.shape != (64, 3) or not np.isfinite(positions).all():
        raise ValueError("Invalid montage")
    records = {r["recording_id"]: r for r in read_rows(base / "recordings.csv")}
    windows = read_rows(base / "windows.csv")
    moments = read_rows(base / "moments.csv")
    gallery = {m["moment_id"]: m for m in moments}
    if len(gallery) != len(moments):
        raise ValueError("Duplicate visual target")
    for r in records.values():
        if r["eeg_path"]:
            path = base / r["eeg_path"]
            x = np.load(path, mmap_mode="r")
            if x.shape != (64, int(r["cache_stop"])) or x.dtype != np.float32 or file_hash(path) != r["cache_sha256"]:
                raise ValueError(f"Invalid cached recording {r['recording_id']}")
    expected, observed = set(), set()
    for w in windows:
        key = w["recording_id"], w["moment_id"]
        r, m = records[key[0]], gallery[key[1]]
        start, stop = int(w["start"]), int(w["stop"])
        if (
            key in observed
            or r["film"] != m["film"]
            or w["film"] != m["film"]
            or start != int(m["start_s"]) * RATE
            or stop != start + RATE * SECONDS
            or stop > int(r["cache_stop"])
            or stop / RATE > float(r["picture_s"])
            or (w["eligible"] == "1") != (not r["reason"] and not w["qc"])
        ):
            raise ValueError(f"Incorrect EEG/target join: {key}")
        observed.add(key)
        if w["eligible"] == "1" and int(w["film"]) not in (6, 9):
            expected.add(key)
    partitions = [verify_split(root / "splits" / name, base_hash, expected, records) for name in ("seed-42", "seed-43")]
    if partitions[0] == partitions[1]:
        raise ValueError("Partition smoke check produced identical assignments")
    with np.load(root / "splits" / "seed-42" / "train_stats.npz") as stats:
        mean, std = stats["mean_uv"], stats["std_uv"]
    clips = {}
    for film, media in info["representatives"].items():
        rows = [m for m in moments if m["film"] == film]
        middle = rows[len(rows) // 2]
        r = next(
            r
            for r in records.values()
            if r["film"] == film and r["eeg_path"] and int(r["cache_stop"]) >= RATE * int(middle["stop_s"])
        )
        path = eeg / f"{r['condition']}_condition" / r["participant"] / f"video{film}.npz"
        with np.load(base / "frames" / f"{film}.npz") as frames:
            scale = Fraction(*map(int, frames["time_base"]))
            original, _ = continuous_eeg(path, channels, int(frames["stop_pts"]) * scale, r)
            cached = np.load(base / r["eeg_path"], mmap_mode="r")
            start = RATE * int(middle["start_s"])
            sample = cached[:, start : start + SECONDS * RATE]
            if not np.array_equal(original[:, start : start + SECONDS * RATE], sample):
                raise ValueError(f"Source sample check failed for film {film}")
            x, clips[film] = normalize_window(sample, mean, std)
            if not np.isfinite(x).all():
                raise ValueError(f"Invalid normalized EEG for film {film}")
            pts = int(frames["pts"][int(middle["onset_frame"])])
        with av.open(str(stimuli / media)) as movie:
            stream = movie.streams.video[0]
            movie.seek(pts, stream=stream)
            frame = next(f for f in movie.decode(stream) if f.pts >= pts)
            gray = cv2.cvtColor(frame.to_ndarray(format="rgb24"), cv2.COLOR_RGB2GRAY)
            if frame.pts != pts or not np.isclose(gray.mean() / 255, float(middle["luminance"])):
                raise ValueError(f"Source frame check failed for film {film}")
        if not base_only:
            target = root / "vision" / f"film-{film}"
            model = json.loads((target / "model.json").read_text())
            index = read_rows(target / "index.csv")
            matrix = np.load(target / "targets.npy")
            if (
                model["revision"] != VISION_REV
                or model["cache_sha256"] != base_hash
                or file_hash(target / "targets.npy") != model["targets_sha256"]
                or matrix.shape != (len(rows), 1408)
                or not np.isfinite(matrix).all()
                or [(int(r["row"]), r["moment_id"]) for r in index] != list(enumerate(r["moment_id"] for r in rows))
            ):
                raise ValueError(f"Invalid target gallery for film {film}")
    return dict(
        recordings=len(records),
        windows=len(windows),
        unique_moments=len(moments),
        checked_partitions=2,
        checked_vision=not base_only,
        sample_clipped_fraction=clips,
    )


def main() -> None:
    """Verify default split manifests and either base preparation or the completed stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    result = verify(
        env_path("STAGE1_DIR", "data/stage1"), env_path("VIDEO_DATA_DIR"), env_path("STIMULUS_DIR"), args.base_only
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
