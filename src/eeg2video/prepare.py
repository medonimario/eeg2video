"""Prepare split-independent EEG and stimulus caches using the Stage 0 audit."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import resample_poly

from .audit import EXPORT_HZ, FILMS, inspect_movie, write_rows
from .cache import POSITION_MODEL, POSITION_REV, RATE, SECONDS, env_path, file_hash, montage, new_directory, read_rows
from .stimulus import audio_envelope, frame_table


def continuous_eeg(path: Path, names: list[str], duration: Fraction, audit: dict) -> tuple[np.ndarray, int]:
    """Validate the source, trim at the joint duration, and resample only supported EEG."""
    with np.load(path, allow_pickle=True) as source:
        data = source["data"]
        order = source["ch_names"].tolist()
        if (
            data.shape != (int(audit["samples"]), len(names))
            or set(order) != set(names)
            or len(order) != len(names)
            or source["sfreq"].item() != EXPORT_HZ
            or source["video_number"].item() != int(audit["film"])
            or source["start_sample"].item() != int(audit["start_sample"])
            or source["end_sample"].item() != int(audit["end_sample"])
            or hashlib.sha256(np.ascontiguousarray(data).data).hexdigest() != audit["signal_sha256"]
        ):
            raise ValueError(f"Recording differs from audited input: {path}")
    supported = min(Fraction(len(data), EXPORT_HZ), duration)
    native_n = int(supported * EXPORT_HZ)
    ordered = data[:native_n, [order.index(ch) for ch in names]].T
    signal = resample_poly(ordered * 1e6, 25, 32, axis=-1)
    n = min(int(supported * RATE), native_n * RATE // EXPORT_HZ)
    result = np.ascontiguousarray(signal[:, :n], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite supported EEG in {path}")
    return result, native_n


def recording_reason(row: dict) -> str:
    """Return conservative default exclusions while retaining source flags for reassessment."""
    reasons = [row["prior_flag"]] if row["prior_flag"] else []
    if row["overlaps_film"] or row["same_signal"] != "True" or row["mne_matches"] != "1":
        reasons.append("provenance_review")
    if row["duplicate_signal"] == "True":
        reasons.append("duplicate_signal")
    if row["finite"] != "True":
        reasons.append("nonfinite")
    return "+".join(reasons)


def window_qc(x: np.ndarray) -> str:
    """Flag gross residual artifacts in a model-rate window expressed in microvolts."""
    reasons = []
    if np.abs(x).max() > 200:
        reasons.append("peak")
    if np.abs(np.diff(x, axis=-1)).max() > 200:
        reasons.append("jump")
    if x.std(axis=-1).min() < 0.01:
        reasons.append("flat")
    return "+".join(reasons)


def prepare(eeg: Path, stimuli: Path, audit: Path, output: Path) -> None:
    """Publish the continuous signals, frame and audio features, and unpartitioned indexes."""
    cv2.setNumThreads(1)
    media = read_rows(audit / "media.csv")
    audited = read_rows(audit / "recordings.csv")
    lookup = {r["media"]: r for r in media}
    if len(lookup) != len(media):
        raise ValueError("Duplicate media paths in the audit")
    with new_directory(output) as work:
        for name in ("eeg", "frames", "audio"):
            (work / name).mkdir()
        channels = montage(work)
        moments, windows, recordings = [], [], []
        grids, durations, representatives = {}, {}, {}
        for film in FILMS:
            variants = [m for m in media if int(m["film"]) == film]
            if not variants:
                continue
            if len({(m["video_sha256"], m["pts_sha256"], m["time_base"]) for m in variants}) != 1:
                raise ValueError(f"Multiple visual edits found for film {film}")
            for m in variants:
                path = stimuli / m["media"]
                current = inspect_movie(path)
                if any(str(current[k]) != m[k] for k in current):
                    raise ValueError(f"Media differs from audit: {path}")
                envfile = work / "audio" / f"{m['audio_sha256']}.npz"
                if not envfile.exists():
                    audio_envelope(path, envfile)
            primary = variants[0]
            representatives[film] = primary["media"]
            grid, duration = frame_table(stimuli / primary["media"], work / "frames" / f"{film}.npz", primary)
            grids[film], durations[film] = grid, duration
            moments.extend(grid)
            print(f"Film {film}: {len(grid)} unique moments", flush=True)
        seen = set()
        for row in audited:
            film = int(row["film"])
            if film not in FILMS or film not in grids:
                raise ValueError(f"Unexpected or unmapped experimental film {film}")
            key = f"{row['participant']}_video{film}"
            version = lookup[row["media"]]
            if key in seen or (row["film"], row["profile"], row["config"]) != (
                version["film"],
                version["profile"],
                version["config"],
            ):
                raise ValueError(f"Duplicate recording or inconsistent stimulus mapping: {key}")
            seen.add(key)
            record = dict(
                row,
                recording_id=key,
                audio_sha256=version["audio_sha256"],
                reason=recording_reason(row),
                eeg_path="",
                native_stop=0,
                cache_stop=0,
                cache_sha256="",
            )
            recordings.append(record)
            if row["finite"] != "True":
                continue
            source = eeg / f"{row['condition']}_condition" / row["participant"] / f"video{film}.npz"
            signal, native_n = continuous_eeg(source, channels, durations[film], row)
            filename = f"eeg/{key}.npy"
            np.save(work / filename, signal)
            record.update(
                eeg_path=filename,
                native_stop=native_n,
                cache_stop=signal.shape[1],
                cache_sha256=file_hash(work / filename),
            )
            for moment in grids[film]:
                start = moment["start_s"] * RATE
                stop = moment["stop_s"] * RATE
                if stop > signal.shape[1]:
                    break
                flags = window_qc(signal[:, start:stop])
                windows.append(
                    dict(
                        recording_id=key,
                        moment_id=moment["moment_id"],
                        film=film,
                        start=start,
                        stop=stop,
                        native_start=moment["start_s"] * EXPORT_HZ,
                        native_stop=moment["stop_s"] * EXPORT_HZ,
                        qc=flags,
                        eligible=int(not record["reason"] and not flags),
                    )
                )
        write_rows(work / "recordings.csv", recordings)
        write_rows(work / "media.csv", media)
        write_rows(work / "moments.csv", moments)
        write_rows(work / "windows.csv", windows)
        rec_map = {r["recording_id"]: r for r in recordings}
        census = defaultdict(list)
        for w in windows:
            r = rec_map[w["recording_id"]]
            census[(w["film"], r["condition"], r["noise"])].append(w)
        counts = []
        for (film, condition, noise), subset in sorted(census.items()):
            keep = [w for w in subset if w["eligible"]]
            counts.append(
                dict(
                    film=film,
                    condition=condition,
                    noise=noise,
                    windows=len(subset),
                    eligible=len(keep),
                    moments=len({w["moment_id"] for w in keep}),
                    viewers=len({rec_map[w["recording_id"]]["participant"] for w in keep}),
                    groups=len({rec_map[w["recording_id"]]["group"] for w in keep}),
                )
            )
        write_rows(work / "census.csv", counts)
        summary = dict(
            version=1,
            rate_hz=RATE,
            units="uV",
            window_seconds=SECONDS,
            onset_shift_s=0,
            onset="reported photodiode-corrected onset",
            positions_model=POSITION_MODEL,
            positions_revision=POSITION_REV,
            representatives=representatives,
            recordings=len(recordings),
            windows=len(windows),
            unique_moments=len(moments),
            eligible_windows=sum(w["eligible"] for w in windows),
            exclusions=dict(Counter(r["reason"] for r in recordings if r["reason"])),
            audit_hashes={n: file_hash(audit / n) for n in ("media.csv", "recordings.csv")},
            manifest_hashes={
                n: file_hash(work / n)
                for n in [
                    "windows.csv",
                    "moments.csv",
                    "recordings.csv",
                    "media.csv",
                    "channels.json",
                    "positions.npy",
                    *(
                        str(p.relative_to(work))
                        for sub in ("frames", "audio")
                        for p in sorted((work / sub).glob("*.npz"))
                    ),
                ]
            },
        )
        (work / "cache.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    """Build the CPU cache from local or cluster paths in .env."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("reports/stage0"))
    args = parser.parse_args()
    prepare(
        env_path("VIDEO_DATA_DIR"), env_path("STIMULUS_DIR"), args.audit, env_path("STAGE1_DIR", "data/stage1") / "base"
    )


if __name__ == "__main__":
    main()
