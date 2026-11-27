"""Generate uncentered frozen V-JEPA-2 targets once per film and time interval."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from transformers import VJEPA2Model

from .audit import FILMS, inspect_movie, write_rows
from .cache import VISION_MODEL, VISION_REV, env_path, file_hash, model_cache_dir, new_directory, read_rows


def image_tensor(frame: av.VideoFrame) -> torch.Tensor:
    """Use the checkpoint's RGB resize, center crop and channel normalization."""
    image = frame.to_image().convert("RGB")
    w, h = image.size
    size = (int(w * 438 / min(w, h)), int(h * 438 / min(w, h)))
    image = image.resize(size, Image.Resampling.BILINEAR)
    x, y = (size[0] - 384) // 2, (size[1] - 384) // 2
    rgb = np.asarray(image.crop((x, y, x + 384, y + 384)), dtype=np.float32) / 255
    rgb -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    rgb /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


@torch.inference_mode()
def encode_frames(model: VJEPA2Model, frames: list[torch.Tensor], device: str) -> np.ndarray:
    """Mean-pool spatial tokens for each frame, then mean-pool the four frame embeddings."""
    pixels = torch.stack(frames).unsqueeze(1).to(device)
    tokens = model.get_vision_features(pixels)
    target = tokens.float().mean(dim=1).mean(dim=0).cpu().numpy()
    if target.shape != (model.config.hidden_size,) or not np.isfinite(target).all():
        raise ValueError("Invalid V-JEPA-2 target")
    return target


def extract_film(base: Path, stimuli: Path, output: Path, film: int, model: VJEPA2Model, device: str) -> None:
    """Decode selected frames, verify source identity and publish a keyed target matrix."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    info = json.loads((base / "cache.json").read_text())
    src = info["representatives"][str(film)]
    manifest = next(r for r in read_rows(base / "media.csv") if r["media"] == src)
    path = stimuli / src
    fingerprint = inspect_movie(path)
    if any(str(fingerprint[k]) != manifest[k] for k in ("video_sha256", "pts_sha256", "time_base")):
        raise ValueError(f"Changed source movie: {path}")
    rows = [r for r in read_rows(base / "moments.csv") if int(r["film"]) == film]
    with np.load(base / "frames" / f"{film}.npz") as frame_data:
        pts = frame_data["pts"]
    indices = [[int(i) for i in row["teacher_frames"].split()] for row in rows]
    targets, frames = [], {}
    index = 0
    with av.open(str(path)) as movie:
        for i, frame in enumerate(movie.decode(video=0)):
            if i in indices[index]:
                if frame.pts != pts[i]:
                    raise ValueError(f"Frame {i} PTS mismatch in {path}")
                frames[i] = image_tensor(frame)
            if i == indices[index][-1]:
                targets.append(encode_frames(model, [frames[j] for j in indices[index]], device))
                index += 1
                frames.clear()
                if index == len(indices):
                    break
    if len(targets) != len(rows) or model.config.hidden_size != 1408:
        raise ValueError(f"Incomplete targets for film {film}")
    with new_directory(output) as work:
        np.save(work / "targets.npy", np.stack(targets))
        write_rows(work / "index.csv", [dict(moment_id=r["moment_id"], row=i) for i, r in enumerate(rows)])
        provenance = dict(
            model=VISION_MODEL,
            revision=VISION_REV,
            film=film,
            video_sha256=manifest["video_sha256"],
            cache_sha256=file_hash(base / "cache.json"),
            targets_sha256=file_hash(work / "targets.npy"),
            frame_offsets_s=[0.25, 0.75, 1.25, 1.75],
            pooling="mean spatial tokens then mean frames",
            centered=False,
        )
        (work / "model.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    """Extract all eight films or one film assigned by an LSF array job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--film", type=int, choices=FILMS)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    root, stimuli = env_path("STAGE1_DIR", "data/stage1"), env_path("STIMULUS_DIR")
    model = VJEPA2Model.from_pretrained(VISION_MODEL, revision=VISION_REV, cache_dir=model_cache_dir())
    model.eval().requires_grad_(False).to(args.device)
    for film in [args.film] if args.film else FILMS:
        extract_film(root / "base", stimuli, root / "vision" / f"film-{film}", film, model, args.device)


if __name__ == "__main__":
    main()
