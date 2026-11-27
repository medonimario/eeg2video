"""Extract timestamped picture features and configuration-specific mixed-audio envelopes."""

import hashlib
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from .cache import SECONDS


def frame_table(path: Path, output: Path, version: dict) -> tuple[list[dict], Fraction]:
    """Decode one visual edit in presentation order and index probe and teacher support."""
    pts, luma, contrast, motion = [], [], [], []
    previous = None
    with av.open(str(path)) as movie:
        stream = movie.streams.video[0]
        base = Fraction(stream.time_base)
        for frame in movie.decode(stream):
            if frame.pts is None or frame.time_base != base:
                raise ValueError(f"Invalid frame timestamp in {path}")
            rgb = frame.to_ndarray(format="rgb24")
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            mean, std = cv2.meanStdDev(gray)
            small = cv2.resize(
                gray, (160, max(1, round(160 * gray.shape[0] / gray.shape[1]))), interpolation=cv2.INTER_AREA
            )
            energy = 0.0
            if previous is not None:
                flow = cv2.calcOpticalFlowFarneback(previous, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                energy = float(cv2.magnitude(flow[..., 0], flow[..., 1]).mean())
            pts.append(frame.pts)
            luma.append(mean.item() / 255)
            contrast.append(std.item() / 255)
            motion.append(energy)
            previous = small
        stop_pts = stream.start_time + stream.duration
    times = np.array(pts, dtype=np.int64)
    edges = np.append(times, stop_pts)
    intervals = np.column_stack((times, np.diff(edges))).astype("<i8")
    if (
        len(times) != int(version["frames"])
        or str(base) != version["time_base"]
        or times[0] != 0
        or np.any(np.diff(edges) <= 0)
        or hashlib.sha256(intervals.tobytes()).hexdigest() != version["pts_sha256"]
    ):
        raise ValueError(f"Decoded picture timeline disagrees with audit for {path}")
    duration = stop_pts * base
    np.savez(
        output,
        pts=times,
        stop_pts=stop_pts,
        time_base=[base.numerator, base.denominator],
        luminance=np.float32(luma),
        contrast=np.float32(contrast),
        motion=np.float32(motion),
    )
    rows = []
    for anchor in range(int(duration // SECONDS)):
        start = anchor * SECONDS
        onset = int(times.searchsorted(start / base, side="right") - 1)
        midpoints = [Fraction(4 * start + j, 4) for j in (1, 3, 5, 7)]
        samples = [int(times.searchsorted(t / base, side="right") - 1) for t in midpoints]
        rows.append(
            dict(
                film=version["film"],
                moment_id=f"{version['film']}-{version['video_sha256']}-{anchor}",
                anchor=anchor,
                start_s=start,
                stop_s=start + SECONDS,
                onset_frame=onset,
                luminance=luma[onset],
                contrast=contrast[onset],
                motion=motion[onset],
                probe_first_pts=pts[max(0, onset - 1)],
                probe_stop_pts=int(edges[onset + 1]),
                teacher_frames=" ".join(map(str, samples)),
                teacher_first_pts=pts[samples[0]],
                teacher_stop_pts=int(edges[samples[-1] + 1]),
            )
        )
    return rows, duration


def audio_envelope(path: Path, output: Path) -> None:
    """Store 100 Hz mixed-audio RMS in file time, respecting AAC timestamps and end padding."""
    with av.open(str(path)) as movie:
        stream = movie.streams.audio[0]
        start = stream.start_time * stream.time_base
        stop = start + stream.duration * stream.time_base
        if start < 0:
            raise ValueError(f"Negative movie audio origin in {path}")
        length = int(np.ceil(float(stop) * 100))
        total, counts = np.zeros(length), np.zeros(length, dtype=np.int64)
        for frame in movie.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Missing audio timestamp in {path}")
            samples = frame.to_ndarray()
            if not frame.format.is_planar:
                samples = samples.reshape(-1, len(frame.layout.channels)).T
            if samples.dtype.kind != "f":
                raise ValueError(f"Expected floating-point AAC decoder output in {path}")
            origin = frame.pts * frame.time_base * frame.sample_rate
            if origin.denominator != 1:
                raise ValueError(f"Fractional audio sample origin in {path}")
            indices = int(origin) + np.arange(frame.samples)
            keep = (indices >= int(np.ceil(start * frame.sample_rate))) & (
                indices < int(np.ceil(stop * frame.sample_rate))
            )
            bins = indices[keep] * 100 // frame.sample_rate
            powers = np.mean(samples[:, keep].astype(np.float64) ** 2, axis=0)
            np.add.at(total, bins, powers)
            np.add.at(counts, bins, 1)
    rms = np.sqrt(np.divide(total, counts, out=np.zeros_like(total), where=counts > 0))
    np.savez(output, rms=rms.astype("float32"), valid=counts > 0, rate_hz=100)
