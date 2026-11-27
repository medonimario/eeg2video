# EEG to video retrieval

Stage 0 audits the engagement dataset. Stage 1 prepares aligned EEG, visual and audio features for films 3–10. The working assumption is that the reported photodiode-corrected onset corresponds to movie time zero. The cache drops EEG beyond the jointly supported duration **before resampling**. Original recordings and trigger coordinates are preserved. Film 0 is the practice trial.

## Set up a fresh cluster checkout

Run these commands from the cloned repository on a host with network access. Use a conda installation and shared filesystem accessible from the compute nodes.

```bash
conda create -n eeg2video python=3.11 pip -y
conda activate eeg2video
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
cp .env.example .env
```

Edit the four paths in `.env` for the cluster before submitting. The EEG root contains `solo_condition/` and `together_condition/`. The stimulus root contains `counterbalance_a/` and `counterbalance_b/`, including their configuration CSVs and MP4s. Point `STAGE1_DIR` and `HF_HUB_CACHE` at two writable directories on the shared data volume, for example `/shared/project/stage1` and `/shared/project/model-weights`. Every CPU and GPU job can then reuse the same files. Each host has its own ignored `.env`. The Python environment can be checked with `python -m pip check`. To run the tests, install `requirements_dev.txt` and run `pytest tests/`.

Allow about **6 GB** in `STAGE1_DIR` for prepared data. Stage 1's pinned downloads in `HF_HUB_CACHE` total **4.14 GB** (3.85 GiB). V-JEPA-2 accounts for almost all of that. The REVE electrode-position bank uses 14.3 kB including its config. The later EEG encoding stage adds about 277 MB for REVE-base, bringing the checkpoint total to about **4.42 GB**. These are file sizes for one revision of each model. [REVE-base file listing](https://huggingface.co/brain-bzh/reve-base/tree/main)

On the login node, export the settings from the edited `.env` so the `hf` command uses the same cache as the project. Run the two pinned downloads once, then enable offline mode for the submitted jobs. On a fresh account, Hugging Face otherwise defaults to `~/.cache/huggingface/hub`.

```bash
set -a
. ./.env
set +a
hf download brain-bzh/reve-positions config.json model.safetensors --revision befa5b57a455b77cf302daf610c2e9ed8140bace
hf download facebook/vjepa2-vitg-fpc64-384 config.json model.safetensors --revision 12ca91694b230e0d4b5b0078af6f4ae1d51e933d
export HF_HUB_OFFLINE=1
```

## Run Stage 1 on LSF

Submit from the repository root with the `eeg2video` environment active. The CPU job reads the audited data, prepares the cache, creates the development partition and a second partition for validation of the splitting code, and checks both. If `reports/stage0/` is missing on a fresh clone, the job runs the audit first. This requires the trusted MNE pickle exports alongside the NPZs. Alternatively, copy the completed Stage 0 audit to the cluster and set `AUDIT_DIR` to its directory. Stage 1 checks the NPZ signals, trigger bounds and movie signatures against the audit.

```bash
mkdir -p outputs errors
bsub < src/eeg2video/hpc/prepare.lsf
```

After the CPU job succeeds, submit the eight-film GPU array. The wrapper requests the prior project's `gpuv100` queue and one GPU per film. When all eight jobs finish, run the final verification on a CPU node with access to the original data.

```bash
bsub < src/eeg2video/hpc/vision.lsf
# After the array completes, on the CPU node
python -m eeg2video.verify
```

A completed film is published independently. To retry a failed film, run `python -m eeg2video.vision --device cuda --film ID` on a GPU node. A new data version or preprocessing policy needs a fresh `STAGE1_DIR`. New partitions can reuse the existing cache. CPU and GPU commands operate on the execution host's paths. No checkpoint training is performed in Stage 1.

## Prepared data

| Location under `STAGE1_DIR` | Contents |
| --- | --- |
| `base/eeg/` | One continuous float32 array in microvolts per finite participant–film recording, in BioSemi64 order. Signals are trimmed at the common end, transposed, and polyphase-resampled from 256 to 200 Hz. Trailing partial windows remain available in the continuous cache. |
| `base/channels.json`, `base/positions.npy` | Ordered electrode names and the matching 64 × 3 coordinates from the official pinned REVE position bank. |
| `base/frames/`, `base/moments.csv` | Per-frame presentation timestamps, luminance, RMS contrast and Farnebäck motion energy. Probe values use the frame on screen at onset. Motion is mean optical-flow magnitude in pixels per frame at 160-pixel width. Frame zero has motion value zero. Moment rows store the actual frame support for probes and the teacher. |
| `base/audio/`, `base/media.csv` | Configuration-specific mixed-soundtrack RMS at 100 Hz, with valid-bin masks. Recordings reference the audio stream hash. |
| `base/recordings.csv`, `base/windows.csv` | Source and cached signal hashes, original triggers, half-open native and resampled bounds, stable movie-moment keys, exclusions and QC. Each single-viewer window is 64 × 400. |
| `base/census.csv`, `base/cache.json` | Counts by film, setup and noise level, plus cache and input fingerprints. |
| `vision/film-ID/` | One uncentered 1,408-D V-JEPA-2 target per unique movie moment, with a row index, source revision and output hash. |
| `splits/seed-42/` | Viewer-group assignments and training-only clipping statistics. Seed 43 checks that the same data can support a different partition. |

**Default eligibility.** The primary split omits the previously flagged `solo11_A_1`, the two provisional paired viewers (`pair4_together_A_4_p2` and `pair9_together_A_9_p2`), and film 10 for `solo31_B_1` because its provenance is uncertain. Valid partners and the latter viewer's other films remain eligible. Finite flagged recordings are retained in the continuous cache. The initial gross-artifact screen flags a two-second window if any channel has an absolute sample or single-sample jump above 200 µV, or a within-window standard deviation below 0.01 µV. Flags preserve the original movie timeline.

**Visual target.** Four frames are sampled at 0.25, 0.75, 1.25 and 1.75 seconds into each window. Each RGB frame is resized bilinearly to a 438-pixel short edge, center-cropped to 384, and normalized with the checkpoint's ImageNet statistics. The frozen encoder treats each frame as a single-image input (duplicated internally for its two-frame tubelet). Spatial tokens are averaged per frame, then the four frame vectors are averaged. This is our explicit sampling choice for the paper's frame-to-window target. A future model fits the target mean using training moments. [V-JEPA-2 checkpoint](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384)

**Partitions and model inputs.** The first task holds out viewers, stratified by solo/together and A/B. A dyad stays in one role. Six development films are available in all three roles. Films 6 and 9 are reserved for later film validation and test, respectively. All eight titles are present in the base cache. Changing the seed or using `python -m eeg2video.partition --seed 42 --fold 0` creates a separate partition. Five-fold rotation accepts indices 0–4. Use `--validation-film` and `--test-film` to reserve different titles in a separately named partition. The fitted mean and standard deviation use only eligible training samples. `normalize_window` clips at ±15 training-channel standard deviations, reports the clipped fraction, then z-scores each channel over the current window.

## Run individual steps

```bash
python -m eeg2video.audit --output reports/stage0
python -m eeg2video.prepare --audit reports/stage0
python -m eeg2video.partition --seed 42
python -m eeg2video.partition --seed 43
python -m eeg2video.verify --base-only
python -m eeg2video.vision --device cuda
python -m eeg2video.verify
```

The final check verifies signal and feature fingerprints, every EEG-window-to-target join, two group partitions, and the indices and contents of all embedding galleries. It also rebuilds one EEG interval per film from its source and checks the on-screen frame against its cached luminance.
