"""Regression coverage for clipping movie support, temporal indexing and split isolation."""

import hashlib
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from eeg2video.audit import write_rows
from eeg2video.cache import read_rows
from eeg2video.partition import build_partition, group_roles, normalize_window
from eeg2video.prepare import continuous_eeg, recording_reason, window_qc
from eeg2video.stimulus import frame_table


def test_trim_precedes_resampling_and_preserves_channel_order(tmp_path: Path) -> None:
    """A large post-movie transient cannot bleed into the last eligible EEG window."""
    names = [f"E{i}" for i in range(64)]
    raw = np.tile(np.linspace(0, 63e-6, 64)[::-1], (640, 1))
    raw[512:] = 10
    path = tmp_path / "video3.npz"
    np.savez(path, data=raw, ch_names=names[::-1], sfreq=256, video_number=3, start_sample=99, end_sample=5219)
    audit = dict(
        samples=640, film=3, start_sample=99, end_sample=5219, signal_sha256=hashlib.sha256(raw.data).hexdigest()
    )
    eeg, native_n = continuous_eeg(path, names, Fraction(2), audit)
    assert native_n == 512
    assert eeg.shape == (64, 400) and eeg.dtype == np.float32
    assert np.max(np.abs(eeg)) < 100
    np.testing.assert_allclose(eeg[:, 200], np.arange(64), atol=0.06)
    short, _ = continuous_eeg(path, names, Fraction(10), audit)
    assert short.shape == (64, 500)
    with pytest.raises(ValueError, match="differs"):
        continuous_eeg(path, names, Fraction(2), dict(audit, signal_sha256="changed"))


def test_probes_follow_presentation_times_at_the_window_boundary(tmp_path: Path) -> None:
    """Use the frame on screen at onset, its motion context and only complete two-second targets."""
    path = tmp_path / "source.mp4"
    with av.open(str(path), "w") as movie:
        stream = movie.add_stream("libx264", rate=Fraction(7, 2))
        stream.width = 64
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(15):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), i * 10, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                movie.mux(packet)
        for packet in stream.encode():
            movie.mux(packet)
    with av.open(str(path)) as movie:
        video = movie.streams.video[0]
        base = video.time_base
        spans = sorted((p.pts, p.duration) for p in movie.demux(video) if p.size)
    version = dict(
        film="8",
        video_sha256="test",
        time_base=str(base),
        frames=len(spans),
        pts_sha256=hashlib.sha256(np.array(spans, dtype="<i8").tobytes()).hexdigest(),
    )
    rows, duration = frame_table(path, tmp_path / "frames.npz", version)
    assert 4 < duration < 5
    assert [r["onset_frame"] for r in rows] == [0, 7]
    assert rows[1]["probe_first_pts"] == spans[6][0]
    assert rows[1]["luminance"] > rows[0]["luminance"]
    for r in rows:
        assert len(r["teacher_frames"].split()) == 4
        assert Fraction(r["teacher_first_pts"]) * base >= r["start_s"] - Fraction(2, 7)


def test_split_fits_only_eligible_training_viewers_and_development_films(tmp_path: Path) -> None:
    """Known channel offsets make held-out leakage detectable in the saved normalization statistics."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "cache.json").write_text("{}")
    rows, windows = [], []
    for i in range(10):
        group = f"pair{i // 2}"
        for film in (3, 6, 9):
            key = f"s{i}-f{film}"
            signal = np.tile(np.arange(400, dtype="float32"), (64, 1)) + i * 1000 + (99999 if film != 3 else 0)
            np.save(base / f"{key}.npy", signal)
            rows.append(dict(recording_id=key, group=group, condition="together", profile="A", eeg_path=f"{key}.npy"))
            windows.append(dict(recording_id=key, moment_id=f"{film}-0", film=film, start=0, stop=400, eligible=1))
    write_rows(base / "recordings.csv", rows)
    write_rows(base / "windows.csv", windows)
    out = tmp_path / "split"
    build_partition(base, out, 42)
    assigned = read_rows(out / "assignments.csv")
    assert {r["moment_id"] for r in assigned} == {"3-0"}
    for group in {r["group"] for r in assigned}:
        assert len({r["role"] for r in assigned if r["group"] == group}) == 1
    training = [np.load(base / f"{r['recording_id']}.npy") for r in assigned if r["role"] == "train"]
    with np.load(out / "train_stats.npz") as s:
        np.testing.assert_allclose(s["mean_uv"], np.concatenate(training, axis=-1).mean(axis=1))
    available = {r["recording_id"] for r in rows}
    rotations = [group_roles(rows, available, 42, k) for k in range(5)]
    for group in rotations[0]:
        assert sum(r[group] == "test" for r in rotations) == 1


def test_quality_and_normalization_leave_raw_input_intact() -> None:
    """Report clipped samples and keep ambiguous recordings explicitly flagged."""
    x = np.zeros((2, 400), dtype=np.float32)
    x[0, 200] = 1000
    y, fraction = normalize_window(x, np.zeros(2), np.ones(2))
    assert fraction == 1 / 800
    assert x[0, 200] == 1000 and np.isfinite(y).all()
    assert set(window_qc(x).split("+")) == {"peak", "jump", "flat"}
    flags = dict(
        prior_flag="", overlaps_film="0", same_signal="", mne_matches="0", duplicate_signal="False", finite="True"
    )
    assert recording_reason(flags) == "provenance_review"
