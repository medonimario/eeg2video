"""Shared file and montage conventions for Stage 1."""

import csv
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import mne
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors.numpy import load_file

POSITION_MODEL = "brain-bzh/reve-positions"
POSITION_REV = "befa5b57a455b77cf302daf610c2e9ed8140bace"
VISION_MODEL = "facebook/vjepa2-vitg-fpc64-384"
VISION_REV = "12ca91694b230e0d4b5b0078af6f4ae1d51e933d"
RATE = 200
SECONDS = 2


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read one manifest, preserving identifiers as strings."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def file_hash(path: Path) -> str:
    """Fingerprint a file without reading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def env_path(key: str, default: str = "") -> Path:
    """Resolve a location configured on the execution host."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    value = os.environ.get(key) or default
    if not value:
        raise ValueError(f"Set {key} in the environment or repository .env")
    return Path(value).expanduser().resolve()


def model_cache_dir() -> Path:
    """Use the shared Hub cache configured for this host, including in a repository .env."""
    return env_path("HF_HUB_CACHE", HF_HUB_CACHE)


@contextmanager
def new_directory(destination: Path) -> Iterator[Path]:
    """Publish complete results atomically on the same filesystem."""
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        result = Path(temp) / "result"
        result.mkdir()
        yield result
        result.rename(destination)


def montage(output: Path) -> list[str]:
    """Look up the BioSemi64 channels in the released REVE position bank in signal order."""
    names = mne.channels.make_standard_montage("biosemi64").ch_names
    cache = model_cache_dir()
    config = Path(hf_hub_download(POSITION_MODEL, "config.json", revision=POSITION_REV, cache_dir=cache))
    weights = hf_hub_download(POSITION_MODEL, "model.safetensors", revision=POSITION_REV, cache_dir=cache)
    bank_names = json.loads(config.read_text())["position_names"]
    coords = load_file(weights)["embedding"][[bank_names.index(name) for name in names]]
    if coords.shape != (64, 3) or not np.isfinite(coords).all():
        raise ValueError("Incomplete REVE electrode coordinates")
    np.save(output / "positions.npy", coords.astype("float32"))
    (output / "channels.json").write_text(json.dumps(names, indent=2) + "\n")
    return names
