"""Audit stimulus timing and recording quality in the trusted engagement study exports."""

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import numpy as np
from dotenv import load_dotenv
from matplotlib.figure import Figure

FILMS = dict(
    enumerate(
        (
            "Sundays_at_Roccos",
            "John_and_Joe",
            "To_R.P._Salazar_with_Love",
            "Eyes_on_the_Stars",
            "Marking_the_Distance",
            "A_Good_Man",
            "The_Icing_on_the_Cake",
            "The_Nature_of_War",
        ),
        start=3,
    )
)
SOURCE_HZ = 2048
EXPORT_HZ = 256
PRIOR_FLAGS = {
    "solo11_A_1": "prior_faulty",
    "pair4_together_A_4_p2": "prior_provisional",
    "pair9_together_A_9_p2": "prior_provisional",
}


def write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Write an audit table with a header, including when no findings were recorded."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_movie(path: Path) -> dict:
    """Read encoded packet presentation times and hash the separate picture and sound streams."""
    with av.open(str(path)) as asset:
        (video,) = asset.streams.video
        (audio,) = asset.streams.audio
        video_hash, audio_hash = hashlib.sha256(), hashlib.sha256()
        intervals = []
        for packet in asset.demux(video, audio):
            if not packet.size:
                continue
            if packet.stream.index == video.index:
                if packet.pts is None or packet.duration <= 0:
                    raise ValueError(f"Missing video presentation interval in {path}")
                intervals.append((packet.pts, packet.duration))
                video_hash.update(packet)
            else:
                audio_hash.update(packet)
        intervals.sort()
        if not intervals or any(a + d != b for (a, d), (b, _) in zip(intervals, intervals[1:])):
            raise ValueError(f"Empty or discontinuous picture timeline in {path}")
        times = np.asarray(intervals, dtype="<i8")
        first = intervals[0][0]
        stop = sum(intervals[-1])
        return dict(
            video_sha256=video_hash.hexdigest(),
            audio_sha256=audio_hash.hexdigest(),
            pts_sha256=hashlib.sha256(times.tobytes()).hexdigest(),
            time_base=str(video.time_base),
            frames=len(intervals),
            picture_start_s=float(first * video.time_base),
            picture_s=float((stop - first) * video.time_base),
            audio_s=float(audio.duration * audio.time_base),
            container_s=asset.duration / av.time_base,
        )


def movie_catalog(root: Path) -> dict:
    """Join each configuration CSV to the matching media files using validated title IDs."""
    by_title = {name: film for film, name in FILMS.items()}
    catalog = {}
    for manifest in sorted(root.glob("counterbalance_*/config_*/*.csv")):
        profile = manifest.parent.parent.name.removeprefix("counterbalance_").upper()
        config = int(manifest.parent.name.removeprefix("config_"))
        with manifest.open() as stream:
            for entry in csv.DictReader(stream):
                film = by_title[Path(entry["video"]).stem]
                path = root / entry["file_path"]
                key = (profile, config, film)
                if key in catalog or path.parent.resolve() != manifest.parent.resolve():
                    raise ValueError(f"Ambiguous movie configuration in {manifest}: {key}")
                catalog[key] = dict(
                    profile=profile, config=config, film=film, media=entry["file_path"], **inspect_movie(path)
                )
    if not catalog:
        raise ValueError(f"No movie configurations in {root}")
    return catalog


def compare_mne(matches: list, signal: np.ndarray, channels: list[str], start: int) -> dict:
    """Report missing or ambiguous MNE copies separately from measured mismatches."""
    fields = dict(
        mne_matches=len(matches),
        first_samp_mne=None,
        mne_first_minus_trigger_ms=None,
        annotation_count=None,
        annotations=None,
        channel_types=None,
        bad_channels=None,
        same_signal=None,
    )
    if len(matches) == 1:
        raw = matches[0]
        if channels != raw.ch_names or raw.info["sfreq"] != EXPORT_HZ:
            raise ValueError("MNE and NPZ channel order or sampling rate differ")
        fields.update(
            first_samp_mne=raw.first_samp,
            mne_first_minus_trigger_ms=1000 * (raw.first_samp / EXPORT_HZ - start / SOURCE_HZ),
            annotation_count=len(raw.annotations),
            annotations=" | ".join(map(str, raw.annotations.description)),
            channel_types=",".join(sorted(set(raw.get_channel_types()))),
            bad_channels=",".join(raw.info["bads"]),
            same_signal=np.array_equal(raw.get_data().T, signal, equal_nan=True),
        )
    return fields


def inspect_recording(path: Path, film: int, raw_by_length: dict, movie: dict) -> dict:
    """Compare an NPZ recording with its trigger bracket, matched MNE copy and picture timeline."""
    with np.load(path, allow_pickle=True) as archive:
        signal = archive["data"]
        channels = archive["ch_names"].tolist()
        if int(archive["video_number"]) != film or float(archive["sfreq"]) != EXPORT_HZ:
            raise ValueError(f"Unexpected film or sample rate in {path}")
        if (
            signal.ndim != 2
            or signal.shape[1] != 64
            or len(channels) != 64
            or len(set(channels)) != 64
            or not len(signal)
        ):
            raise ValueError(f"Unexpected EEG layout in {path}")
        start, end, onset_code, offset_code = (
            int(archive[k]) for k in ("start_sample", "end_sample", "video_start_code", "video_end_code")
        )
    if not 0 <= start < end:
        raise ValueError(f"Invalid trigger bracket in {path}")
    finite = bool(np.isfinite(signal).all())
    export_s = len(signal) / EXPORT_HZ
    trigger_s = (end - start) / SOURCE_HZ
    return dict(
        film=film,
        start_code=onset_code,
        end_code=offset_code,
        start_sample=start,
        end_sample=end,
        samples=len(signal),
        trigger_s=trigger_s,
        export_s=export_s,
        export_minus_trigger_ms=1000 * (export_s - trigger_s),
        media=movie["media"],
        picture_s=movie["picture_s"],
        trigger_minus_picture_s=trigger_s - movie["picture_s"],
        export_minus_picture_s=export_s - movie["picture_s"],
        trigger_minus_container_s=trigger_s - movie["container_s"],
        **compare_mne(raw_by_length.get(len(signal), []), signal, channels, start),
        finite=finite,
        rms_uv=float(np.sqrt(np.mean(signal**2)) * 1e6) if finite else None,
        peak_uv=float(np.max(np.abs(signal)) * 1e6) if finite else None,
        flat_channels=",".join(ch for ch, span in zip(channels, np.ptp(signal, axis=0)) if span == 0),
        signal_sha256=hashlib.sha256(np.ascontiguousarray(signal).data).hexdigest(),
    )


def mark_overlaps(rows: list[dict], subject: Path) -> None:
    """Check trigger coordinates within one folder, using practice only as a timing reference."""
    spans = [(r["film"], r["start_sample"], r["end_sample"]) for r in rows]
    practice = subject / "video0.npz"
    if practice.is_file():
        with np.load(practice) as archive:
            spans.append((0, int(archive["start_sample"]), int(archive["end_sample"])))
    for row in rows:
        row["overlaps_film"] = ",".join(
            str(film)
            for film, start, end in spans
            if film != row["film"] and start < row["end_sample"] and end > row["start_sample"]
        )


def scan_eeg(root: Path, catalog: dict) -> tuple[list[dict], list[dict]]:
    """Audit every experimental film and account for absent participant-film recordings."""
    recordings, missing = [], []
    for condition in ("solo", "together"):
        folder = root / f"{condition}_condition"
        subjects = sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
        if not subjects:
            raise ValueError(f"No participant folders in {folder}")
        pattern = r"(solo\d+)_([AB])_(\d+)" if condition == "solo" else r"(pair\d+)_together_([AB])_(\d+)_p([12])"
        for subject in subjects:
            match = re.fullmatch(pattern, subject.name)
            if not match:
                raise ValueError(f"Unrecognized participant ID {subject.name}")
            group, profile, config, *member = match.groups()
            identity = dict(
                participant=subject.name,
                group=group,
                member=member[0] if member else "",
                condition=condition,
                profile=profile,
                config=int(config),
            )
            with (subject / f"{subject.name}_eeg_processed.pkl").open("rb") as stream:
                raws = pickle.load(stream)
            raw_by_length = defaultdict(list)
            for raw in raws:
                raw_by_length[raw.n_times].append(raw)
            subject_rows = []
            for film in FILMS:
                path = subject / f"video{film}.npz"
                if not path.exists():
                    missing.append(dict(participant=subject.name, film=film))
                    continue
                movie = catalog[profile, int(config), film]
                row = inspect_recording(path, film, raw_by_length, movie)
                subject_rows.append(
                    dict(
                        **identity,
                        **row,
                        noise="low" if (film <= 6) == (profile == "A") else "high",
                        prior_flag=PRIOR_FLAGS.get(subject.name, ""),
                    )
                )
            mark_overlaps(subject_rows, subject)
            recordings.extend(subject_rows)
    if not recordings:
        raise ValueError("No experimental EEG recordings found")
    return recordings, missing


def pair_differences(records: list[dict]) -> list[dict]:
    """Measure same-session differences in native sample coordinates for the two viewers."""
    pairs = defaultdict(dict)
    for row in records:
        if row["member"]:
            members = pairs[row["group"], row["film"]]
            if row["member"] in members:
                raise ValueError(f"Duplicate dyad member in {row['group']}, film {row['film']}")
            members[row["member"]] = row
    results = []
    for (group, film), members in sorted(pairs.items()):
        if members.keys() == {"1", "2"}:
            a, b = members["1"], members["2"]
            results.append(
                dict(
                    group=group,
                    film=film,
                    start_delta=b["start_sample"] - a["start_sample"],
                    end_delta=b["end_sample"] - a["end_sample"],
                    sample_count_delta=b["samples"] - a["samples"],
                )
            )
    return results


def film_summary(records: list[dict], media: list[dict]) -> list[dict]:
    """Summarize measured duration differences and independent picture timelines by title."""
    results = []
    for film, title in FILMS.items():
        rows = [r for r in records if r["film"] == film]
        if not rows:
            continue
        variants = [r for r in media if r["film"] == film]
        diffs = [r["trigger_minus_picture_s"] for r in rows]
        codes = sorted({(r["start_code"], r["end_code"]) for r in rows})
        results.append(
            dict(
                film=film,
                title=title,
                recordings=len(rows),
                code_pairs=str(codes),
                visual_versions=len({(r["video_sha256"], r["time_base"], r["pts_sha256"]) for r in variants}),
                distinct_audio_streams=len({r["audio_sha256"] for r in variants}),
                min_picture_s=min(r["picture_s"] for r in variants),
                full_2s_windows=int(min(r["picture_s"] for r in variants) // 2),
                min_excess_s=min(diffs),
                median_excess_s=float(np.median(diffs)),
                max_excess_s=max(diffs),
            )
        )
    return results


def timing_plot(records: list[dict], path: Path) -> None:
    """Show film-level variation in the measured trigger-to-picture duration difference."""
    fig = Figure(figsize=(8.4, 4.9), layout="constrained")
    ax = fig.subplots()
    colors = {"solo": "#176a68", "together": "#b15832"}
    for condition, offset in (("solo", -0.15), ("together", 0.15)):
        for y, film in enumerate(FILMS):
            values = [
                r["trigger_minus_picture_s"]
                for r in records
                if r["film"] == film and r["condition"] == condition and r["member"] != "2"
            ]
            if values:
                ax.hlines(y + offset, min(values), max(values), color=colors[condition], linewidth=2, alpha=0.65)
                ax.scatter(
                    np.median(values),
                    y + offset,
                    color=colors[condition],
                    zorder=3,
                    s=34,
                    label=condition.capitalize() if y == 0 else None,
                )
    ax.set_yticks(range(len(FILMS)), [n.replace("_", " ") for n in FILMS.values()])
    ax.invert_yaxis()
    ax.set_xlabel("Trigger interval minus picture duration (seconds)")
    ax.set_title("Recorded interval exceeds picture duration in all eight films", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    fig.text(
        0.02,
        -0.005,
        "Dots show medians. Lines span the observed range. Each session contributes once per film.",
        color="#36464c",
        fontsize=9,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.14)


def run(eeg: Path, stimuli: Path, output: Path) -> None:
    """Publish the audit tables and timing plot together after all source reads succeed."""
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    catalog = movie_catalog(stimuli)
    records, missing = scan_eeg(eeg, catalog)
    pairs = pair_differences(records)
    counts = Counter(r["signal_sha256"] for r in records)
    for row in records:
        row["duplicate_signal"] = counts[row["signal_sha256"]] > 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as work:
        result = Path(work) / "result"
        result.mkdir()
        write_rows(result / "recordings.csv", records)
        write_rows(result / "media.csv", list(catalog.values()))
        write_rows(result / "films.csv", film_summary(records, list(catalog.values())))
        write_rows(result / "missing.csv", missing, ["participant", "film"])
        write_rows(result / "pairs.csv", pairs, ["group", "film", "start_delta", "end_delta", "sample_count_delta"])
        timing_plot(records, result / "timing.png")
        summary = dict(
            recordings=len(records),
            participants=len({r["participant"] for r in records}),
            groups=len({r["group"] for r in records}),
            missing=missing,
            complete_pair_films=len(pairs),
            overlapping_experimental_rows=sum(bool(r["overlaps_film"]) for r in records),
            source_hz=SOURCE_HZ,
            export_hz=EXPORT_HZ,
            media_variants=len(catalog),
            nonfinite=sum(not r["finite"] for r in records),
            duplicate_signals=sum(r["duplicate_signal"] for r in records),
            mne_unmatched=sum(r["mne_matches"] != 1 for r in records),
            mne_signal_mismatches=sum(r["same_signal"] is False for r in records),
            annotated_segments=sum(bool(r["annotation_count"]) for r in records),
        )
        (result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        result.rename(output)


def main() -> None:
    """Run the local audit using paths from the environment or the checkout's .env file."""
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/stage0"))
    args = parser.parse_args()
    for key in ("VIDEO_DATA_DIR", "STIMULUS_DIR"):
        if not os.environ.get(key):
            parser.error(f"Set {key} in the environment or the project .env")
    run(Path(os.environ["VIDEO_DATA_DIR"]), Path(os.environ["STIMULUS_DIR"]), args.output)


if __name__ == "__main__":
    main()
