# CLAUDE.md

Guidance for AI coding assistants and contributors working on layerlab.

## What layerlab is

layerlab takes a video file and separates it into layers:

1. **Dialogue**: speech, with transcript and timing
2. **Music**: background and featured music beds
3. **Sound effects**: discrete SFX events
4. **On-screen graphics**: overlays such as titles, logos, lower thirds
5. **Edit timeline**: cuts, SFX hits, and overlay appearances, derived from the layers above

Everything runs locally. Do not add code that uploads media or calls a remote inference API by default. A remote backend, if ever added, must be opt-in and clearly labelled.

## Current state

Pre-alpha. The pipeline core exists: `layerlab run`, the job folder, the manifest, resumable steps, and the `extract`, `audio` and `analyze` steps. The `inpaint` and `export` steps are **placeholders**: they declare their inputs and record `"placeholder": true` but do no processing yet.

## Architecture

layerlab is a **pipeline of independent steps**. Each step:

- reads its inputs from the **job folder**,
- writes its outputs to the job folder,
- records what it did, and its outputs, in **`manifest.json`** inside the job folder.

Steps never call each other directly. The pipeline runner decides the order by reading the manifest. A step can be re-run on its own, and a failed step must not corrupt another step's outputs.

### Job folder

`layerlab run input.mp4 --out ./output` creates (or reopens) `./output/<first 16 hex chars of the input's SHA-256>/`:

```
<job>/
  manifest.json          # source of truth: source info, one record per step
  steps/
    extract/             # audio.wav (48 kHz stereo PCM), video_only.mp4 (stream copy)
    audio/               # dialogue.wav, music.wav, effects.wav (32-bit float), sfx_events.json
    analyze/             # shots.json, overlays.json, masks/<track>/<sample>.png, [debug_preview.mp4]
    <step>/              # one folder per step, written only by that step
    <step>.tmp/          # exists only while that step runs
```

- The original video is never copied or modified. The manifest records its absolute path, size and SHA-256.
- Paths in the manifest are POSIX paths relative to the job folder.
- A step writes into `job.work_dir` (`steps/<step>.tmp/`). On success the runner swaps it into `steps/<step>/`; on failure it deletes it. Partial outputs are never mistaken for complete ones.
- Outputs are plain, inspectable formats where practical: JSON for structured data, WAV or FLAC for audio, PNG for frames, MP4 or MOV for video.

### Manifest

`manifest.json` (`src/layerlab/manifest.py`) records, per step: status (`pending`, `running`, `succeeded`, `failed`, `skipped`), step version, input hash, outputs, step `data` (for example the probed fps, resolution and duration from `extract`), timings, and the error message on failure. It also records the run's compute `device` (`name`, `detail`, `requested`). Writes are atomic. Changing the schema requires bumping `SCHEMA_VERSION`, keeping older versions readable, and adding a line to the schema history in the module docstring (currently version 2).

### Steps

Steps run in the order listed in `src/layerlab/steps/__init__.py`: `extract`, `audio`, `analyze`, `inpaint`, `export`. Each is a module defining:

- `NAME` and `VERSION`. Bump `VERSION` whenever a change alters the step's outputs.
- `inputs(job) -> list[Path]`: the files it reads. Use `job.source_path` or `job.output("<step>", "<key>")` to get an earlier step's outputs through the manifest.
- `run(job) -> StepResult`: writes files into `job.work_dir` and returns `StepResult(outputs={key: filename}, data={...})`. Report progress with `job.report_progress(fraction)`.

The runner (`src/layerlab/pipeline.py`):

- **Resumes.** The input hash is the SHA-256 of the step name, version and the hashes of its input files. If the manifest records a success with the same hash and the outputs still exist, the step is reported `cached` and not run. Changing an upstream output therefore re-runs only the steps that read it. `--force` re-runs everything.
- **Skips.** `--skip <step>` (repeatable) does not run that step. An earlier success is kept so later steps can still use its outputs; otherwise the step is recorded `skipped`, and any step that needs its outputs fails with a message saying so.
- **Fails safely.** A step that raises gets `status: failed` and `error` in the manifest. Earlier outputs are kept, the pipeline stops, and `layerlab run` exits 1. Raise `LayerlabError` with a user-facing message for expected failures; anything else is recorded as `ExceptionType: message`.
- **Shows progress** with one Rich progress bar per step.

Keep steps free of CLI code so they can be tested directly.

### Model backends behind interfaces

Model code is never called directly from a step. Each capability is defined by a small interface (a `Protocol` in `src/layerlab/backends/`). Concrete backends implement the interface. Steps depend only on the interface, so a backend can be swapped without touching step logic.

Run options such as the backend choice reach steps as `job.options` (set by `run_pipeline(options=...)`). A step that reads an option must return it from `config(job)`, which goes into its input hash, so changing the option re-runs the step. Steps report what they are doing with `job.report_status(text)`, for example while a model downloads.

### Audio step and separation backends

`src/layerlab/steps/audio.py` separates `extract`'s `audio.wav` into stems through the `Separator` protocol (`src/layerlab/backends/separation.py`). Backends are listed in `src/layerlab/backends/catalog.py` and chosen with `--audio-backend`:

- `tiger` (default, `backends/tiger_dnr.py`): TIGER-DnR as the int8-weight ONNX export `audiojs/tiger-dnr`, on onnxruntime. The graph takes one mono 12 s, 44.1 kHz segment as an STFT (n_fft 2048, hop 512, periodic Hann, centred, reflect-padded; `audio/dsp.py` matches `torch.stft`) and returns dialogue, music and effects spectrograms. One model run takes about 35 s per segment on a laptop CPU, so the model runs once per chunk on the mid channel and its stems become power-ratio masks applied to each channel's STFT. Stems stay stereo and sum to the mix up to resampling error.
- `demucs` (`backends/htdemucs.py`, extra `layerlab[demucs]`): htdemucs_ft, four checkpoints verified by the model manager before `torch.load`. vocals map to `dialogue`; drums, bass and other to `music`; `effects` is reported unavailable. The torch/demucs imports are inside `load` so the module imports without the extra.

`audio/chunking.py` streams the WAV through the separator in overlapping chunks (`chunk_seconds`, `overlap_seconds` per backend) with a sin²/cos² crossfade, writing stems as it goes, so memory does not grow with file length. It also accumulates the residual `mix - sum(stems)`. The step records it in `data.mix_check` (`residual_db` relative to the mix, `None` when exact or silent; `within_tolerance` against -20 dB). An out-of-tolerance residual is flagged, not a failure.

`audio/sfx.py` runs librosa onset detection on `effects.wav` in blocks, refines each onset to the attack within one 512-sample hop, and writes `sfx_events.json`: `time`, `duration`, `peak_dbfs`, `confidence` (onset strength relative to the file's strongest). With no effects stem it writes `"available": false` and no events.

When adding a backend:

- Add its weights to the model registry (below), with the licence. Call `get_model("<name>")` to get the file path; it downloads on first use.
- Prefer `safetensors`, ONNX, or GGML/GGUF weights. Never load pickled checkpoints from untrusted sources.
- Use `job.device` (`cuda`, `rocm`, `mps` or `cpu`) to choose where to run. Do not detect the device yourself.
- Keep heavy imports (`torch`, model libraries) inside the backend module so `layerlab --version` and `layerlab doctor` stay fast and work without them.

### Model weights and the registry

**Weights are never bundled in the repository or the package.** `.gitignore` and `[tool.hatch.build] exclude` block weight file types, and `tests/test_registry.py` fails if one is found in the repo.

- `src/layerlab/models/registry.toml` lists each model: `name`, `task`, `description`, `url`, `sha256`, `size`, `license` (SPDX, or `NOASSERTION` if the publisher states none), `commercial_use`, `homepage`. Pin Hugging Face URLs to a commit (`/resolve/<40-hex sha>/`). Take the sha256 and size from the publisher (the Hugging Face API lists the LFS `oid` and `size`), not from a guess. Set `commercial_use = true` only when the licence clearly allows it.
- `$LAYERLAB_MODEL_REGISTRY` points at a different registry file.
- Downloads (`src/layerlab/models/download.py`) stream into `<file>.part`, resume with an HTTP Range request, and only move the file into place once its size and SHA-256 match. A checksum mismatch deletes the partial file. Only `https` URLs are allowed (plain `http` only for localhost, which the tests use).
- The cache is `platformdirs.user_cache_dir("layerlab")/models/<name>/<file>`, or `$LAYERLAB_CACHE_DIR/models/...`. A cached file is trusted if its size matches; its hash was verified when it was downloaded.
- `layerlab models list` shows the registry with licences. `layerlab models fetch <name>...` downloads ahead of time.
- There is no download lock yet. Two processes fetching the same model at once can corrupt the `.part` file (the checksum then catches it).

### Devices

`src/layerlab/devices.py` picks `cuda`, `rocm`, `mps` or `cpu`, in that order of preference. If PyTorch is installed, its view wins (a CPU-only PyTorch build means `cpu` even on a GPU machine). Otherwise it looks for `nvidia-smi`, ROCm tools or `/dev/kfd`, and Apple Silicon. `layerlab run --device <name>` overrides detection; an override that was not detected is still used but flagged in `detail`. The choice is printed and stored in the manifest's `device`. The device is not part of a step's input hash, so changing it does not invalidate cached outputs.

### Analyze step and detectors

`src/layerlab/steps/analyze.py` only reads `extract`'s `video_only.mp4`. It checks the file's size and mtime are unchanged afterwards and records `video_unchanged`. Each detector is its own module under `src/layerlab/analyze/`, with its own section of thresholds in `analyze/defaults.toml`. `--analyze-config my.toml` overrides any subset (unknown keys are rejected, and the CLI validates the file before any step runs). The resolved settings go into the step's input hash.

- `cuts.py`: PySceneDetect `ContentDetector` (`[cuts]`). Writes `shots.json` with `start`, `end` (exclusive), `duration` and frame numbers, plus the list of cut times. Handles both PySceneDetect 0.6 and 0.7 timecode APIs.
- `frames.py`: frames streamed from an ffmpeg pipe, all frames or sampled at `[sampling] fps` (sample k is the frame nearest k / fps s), scaled to at most `max_side`.
- Text: `backends/text_detection.py` runs PaddleOCR's PP-OCRv3 DB detector (`paddleocr-det-v3`, ONNX, onnxruntime; detection only, no recognition, no PaddlePaddle) behind the `TextDetector` protocol. `text_overlays.py` groups per-sample boxes into tracks by IoU with gap bridging and drops tracks shorter than `min_samples` (`[text]`).
- `logos.py`: per-pixel stability over the samples at `analysis_width`. It keeps a histogram of quantised luma plus running mean and variance. A pixel is static if it is within one level of its mode in at least `stable_fraction` (0.8) of samples. Static regions are filtered by area, edge density and span (bars are not logos). A second pass finds the samples each logo is present in (`[logos]`).
- `overlays.py`: tracks, mask PNG sequences and `overlays.json`. Masks are 8-bit single-channel PNGs at video resolution, one per sample a track covers, at `masks/<track id>/<sample:06d>.png`. Each track lists its masks with their times.
- `preview.py`: `--debug-preview` decodes every frame, draws track boxes (text green, logo orange), cut markers and the shot counter, and encodes `debug_preview.mp4` (mpeg4, so it works with any ffmpeg build).

The text detector is loaded through `analyze.load_text_detector`, which tests replace with a scripted fake.

### Media tools

ffmpeg and ffprobe are the only required external tools. Call them through `src/layerlab/ffmpeg.py` (`probe`, `run`), which takes argument lists, never shell strings, and raises `FFmpegError` with a clear message when a tool is missing or fails.

## Code layout

```
src/layerlab/
  __init__.py     # __version__
  __main__.py     # python -m layerlab
  cli.py          # Typer app: --version, doctor, run
  doctor.py       # environment checks, dependency-free
  errors.py       # LayerlabError: expected failures with user-facing messages
  hashing.py      # SHA-256 helpers
  manifest.py     # Manifest, StepRecord: manifest.json schema and atomic writes
  job.py          # Job: job folder paths, step outputs, hash cache, progress
  step.py         # StepResult, StepSpec: the runner/step contract
  pipeline.py     # run_pipeline: ordering, caching, skipping, failures, progress
  ffmpeg.py       # ffprobe/ffmpeg wrappers
  steps/          # extract, audio (real); analyze, inpaint, export (placeholders)
  audio/          # dsp (STFT, resampling), chunking (crossfaded streaming), sfx (onsets)
  backends/       # separation protocol, catalog, tiger_dnr, htdemucs, text_detection
  analyze/        # defaults.toml, config, frames, cuts, text_overlays, logos, overlays, preview
  devices.py      # compute device detection and selection
  models/         # registry.toml, registry loader, resumable verified downloads, cache
tests/
  conftest.py         # synthetic_video / silent_video fixtures; fake backends for every test
  synthetic_audio.py  # mixes with known stems, BandSplitSeparator, IdentitySeparator
  synthetic_video.py  # video with known cuts (2 s, 4 s), a caption (1-4.5 s) and a corner logo
```

## Tests and media

Never commit binary media. Tests that need video use the `synthetic_video` fixture (colour bars, a 440 Hz sine tone, a drawtext overlay) or `silent_video`, both generated by ffmpeg into a temp folder. They skip when ffmpeg is not on PATH, except in CI, where `LAYERLAB_REQUIRE_FFMPEG=1` makes them fail instead. Runner behaviour is tested with fake steps in `tests/test_pipeline.py` and needs no ffmpeg. Downloads are tested against a local HTTP server in `tests/test_downloads.py`.

Tests never download or run real weights by default. A session-scoped autouse fixture in `conftest.py` points `LAYERLAB_CACHE_DIR` at a temp folder and replaces the `tiger` and `demucs` backends with `BandSplitSeparator`. It is session-scoped because module-scoped fixtures also run the pipeline. Audio tests use `synthetic_audio.make_mix()`: dialogue, music and effects in separate frequency bands, with noise bursts at known times. The fake separator splits by band, so stems, residuals and SFX onsets can be checked against the known answer. Tests marked `@pytest.mark.real_models` restore the real backends and the user's cache, and run only with `LAYERLAB_REAL_MODEL_TESTS=1` (about a minute on CPU; Demucs also needs the extra).

## Conventions

- Python 3.10+. Use `from __future__ import annotations` only when needed for forward references.
- Type everything. `mypy --strict` must pass. Avoid `Any` at module boundaries.
- Lint and format with ruff (line length 100). Run `ruff check .` and `ruff format .` before committing.
- Tests use pytest. Mock `nvidia-smi` and other optional tools. ffmpeg is installed in CI.
- CI runs lint, typecheck, and tests on Linux, macOS, and Windows with Python 3.10 and 3.12. Code must work on all three. Use `pathlib` and never hard-code `/` separators.
- Keep the CLI thin: parse arguments, call library code, print results. Library code must be importable without the CLI.
- Print plain ASCII in CLI output. Windows consoles may not render other characters.

## Commands

```bash
python -m pip install -e ".[dev]"   # install with dev tools
ruff check . && ruff format --check .
mypy
pytest
layerlab --version
layerlab doctor
layerlab run input.mp4 --out ./output [--skip <step>] [--force] [--device auto|cuda|rocm|mps|cpu]
layerlab models list
layerlab models fetch <name>...
layerlab run input.mp4 --audio-backend demucs        # needs pip install -e ".[demucs]"
layerlab run input.mp4 --debug-preview --analyze-config my.toml
LAYERLAB_REAL_MODEL_TESTS=1 pytest -m real_models    # runs the real models
pre-commit install
```

## Things not to do

- Do not upload media or call remote APIs by default.
- Do not commit model weights, footage, or job outputs (see `.gitignore`).
- Do not make steps depend on each other's internal code; go through the manifest and job folder.
- Do not import heavy ML libraries at module import time in CLI or doctor code paths.
- Do not commit or push unless asked.
