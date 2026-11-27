"""Regression checks for the three clocks and for missing copies in the engagement export."""

from pathlib import Path

import numpy as np
import pytest

from eeg2video.audit import inspect_recording, mark_overlaps, pair_differences


def test_missing_mne_preserves_independent_clocks(tmp_path: Path) -> None:
    """A missing MNE copy must not lose a valid NPZ or turn a duration difference into an onset shift."""
    path = tmp_path / "video3.npz"
    np.savez(
        path,
        data=np.ones((512, 64)),
        ch_names=[f"ch{i}" for i in range(64)],
        sfreq=256,
        video_number=3,
        video_start_code=40,
        video_end_code=160,
        start_sample=20480,
        end_sample=24579,
    )
    row = inspect_recording(path, 3, {}, {"media": "example.mp4", "picture_s": 1.5, "container_s": 1.55})
    assert row["export_s"] == 2
    assert row["export_minus_trigger_ms"] == pytest.approx(-3000 / 2048)
    assert row["trigger_minus_picture_s"] == pytest.approx(0.5 + 3 / 2048)
    assert row["mne_matches"] == 0
    assert row["same_signal"] is None
    assert row["annotation_count"] is None


def test_pair_timing_is_within_the_same_session_and_film() -> None:
    """Equal shifts of both markers preserve the pair's common stimulus interval."""
    first = dict(group="pair1", film=3, member="1", start_sample=100, end_sample=4196, samples=512)
    second = dict(first, member="2", start_sample=128, end_sample=4224)
    other = dict(first, film=4)
    assert pair_differences([second, other, first]) == [
        dict(group="pair1", film=3, start_delta=28, end_delta=28, sample_count_delta=0)
    ]


def test_practice_overlap_is_reported_without_adding_a_trial(tmp_path: Path) -> None:
    """Use the practice bracket to flag conflicts without including its signal in the census."""
    np.savez(tmp_path / "video0.npz", start_sample=100, end_sample=200)
    rows = [dict(film=10, start_sample=50, end_sample=150), dict(film=3, start_sample=200, end_sample=300)]
    mark_overlaps(rows, tmp_path)
    assert [r["overlaps_film"] for r in rows] == ["0", ""]
    assert [r["film"] for r in rows] == [10, 3]
