"""Build independent-viewer partitions and fit REVE input scaling on eligible training windows."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audit import FILMS, write_rows
from .cache import env_path, file_hash, new_directory, read_rows


def group_roles(records: list[dict], eligible_ids: set[str], seed: int, fold: int | None) -> dict[str, str]:
    """Stratify independent groups by social setup and counterbalance, keeping partners together."""
    strata = defaultdict(set)
    identities = {}
    for r in records:
        if r["recording_id"] in eligible_ids:
            group = r["group"]
            stratum = (r["condition"], r["profile"])
            if identities.setdefault(group, stratum) != stratum:
                raise ValueError(f"Conflicting stratum for {group}")
            strata[stratum].add(group)
    rng = np.random.default_rng(seed)
    roles = {}
    for stratum, groups in sorted(strata.items()):
        ordered = rng.permutation(sorted(groups)).tolist()
        if fold is None:
            if len(groups) < 3:
                raise ValueError(f"Need at least three independent groups in {stratum}")
            count = max(1, round(0.15 * len(groups)))
            for i, group in enumerate(ordered):
                roles[group] = "test" if i < count else "val" if i < 2 * count else "train"
        else:
            if len(groups) < 5 or fold not in range(5):
                raise ValueError("Five-fold rotation requires five groups per stratum and fold 0..4")
            for i, group in enumerate(ordered):
                part = i % 5
                roles[group] = "test" if part == fold else "val" if part == (fold + 1) % 5 else "train"
    return roles


def normalize_window(signal: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, float]:
    """Clip using training-channel statistics, then z-score each channel within the current window."""
    lo, hi = (mean - 15 * std)[:, None], (mean + 15 * std)[:, None]
    fraction = float(np.mean((signal < lo) | (signal > hi)))
    clipped = np.clip(signal, lo, hi)
    scale = clipped.std(axis=1, keepdims=True)
    return ((clipped - clipped.mean(axis=1, keepdims=True)) / np.maximum(scale, 1e-6)).astype("float32"), fraction


def build_partition(
    base: Path, output: Path, seed: int, fold: int | None = None, validation_film: int = 6, test_film: int = 9
) -> None:
    """Write assignments and fold-specific statistics without modifying the continuous or visual caches."""
    if validation_film == test_film or not {validation_film, test_film} <= FILMS.keys():
        raise ValueError("Choose two different experimental films to reserve")
    recordings = read_rows(base / "recordings.csv")
    windows = read_rows(base / "windows.csv")
    by_id = {r["recording_id"]: r for r in recordings}
    eligible = [w for w in windows if w["eligible"] == "1" and int(w["film"]) not in (validation_film, test_film)]
    roles = group_roles(recordings, {w["recording_id"] for w in eligible}, seed, fold)
    by_recording = defaultdict(list)
    for w in eligible:
        by_recording[w["recording_id"]].append(w)
    count = 0
    total, squares = np.zeros(64), np.zeros(64)
    assignments = []
    for key, rows in by_recording.items():
        group = by_id[key]["group"]
        role = roles[group]
        signal = None
        if role == "train":
            signal = np.load(base / by_id[key]["eeg_path"], mmap_mode="r")
        for w in rows:
            assignments.append(dict(recording_id=key, moment_id=w["moment_id"], group=group, role=role))
            if signal is not None:
                x = signal[:, int(w["start"]) : int(w["stop"])].astype(np.float64)
                total += x.sum(axis=1)
                squares += np.square(x).sum(axis=1)
                count += x.shape[1]
    if set(roles.values()) != {"train", "val", "test"} or count == 0:
        raise ValueError("Empty partition role")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    if not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Invalid training-channel variance")
    with new_directory(output) as work:
        write_rows(work / "assignments.csv", assignments)
        np.savez(work / "train_stats.npz", mean_uv=mean, std_uv=std, samples=count)
        meta = dict(
            seed=seed,
            fold=fold,
            task="new_groups_familiar_moments",
            reserved_validation_film=validation_film,
            reserved_test_film=test_film,
            groups=roles,
            cache_sha256=file_hash(base / "cache.json"),
            windows={role: sum(r["role"] == role for r in assignments) for role in ("train", "val", "test")},
        )
        (work / "split.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    """Make a reproducible partition, optionally rotating one of five outer group folds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--validation-film", type=int, choices=FILMS, default=6)
    parser.add_argument("--test-film", type=int, choices=FILMS, default=9)
    args = parser.parse_args()
    root = env_path("STAGE1_DIR", "data/stage1")
    name = f"seed-{args.seed}" + (f"-fold-{args.fold}" if args.fold is not None else "")
    if (args.validation_film, args.test_film) != (6, 9):
        name += f"-valfilm-{args.validation_film}-testfilm-{args.test_film}"
    build_partition(root / "base", root / "splits" / name, args.seed, args.fold, args.validation_film, args.test_film)


if __name__ == "__main__":
    main()
