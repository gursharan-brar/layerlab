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

Pre-alpha. The pipeline core exists: `layerlab run`, the job folder, the manifest, resumable steps, and the `extract` step. The `audio`, `analyze`, `inpaint` and `export` steps are **placeholders**: they declare their inputs and record `"placeholder": true` but do no processing yet.

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
    <step>/              # one folder per step, written only by that step
    <step>.tmp/          # exists only while that step runs
```

- The original video is never copied or modified. The manifest records its absolute path, size and SHA-256.
- Paths in the manifest are POSIX paths relative to the job folder.
- A step writes into `job.work_dir` (`steps/<step>.tmp/`). On success the runner swaps it into `steps/<step>/`; on failure it deletes it. Partial outputs are never mistaken for complete ones.
- Outputs are plain, inspectable formats where practical: JSON for structured data, WAV or FLAC for audio, PNG for frames, MP4 or MOV for video.

### Manifest

`manifest.json` (`src/layerlab/manifest.py`) records, per step: status (`pending`, `running`, `succeeded`, `failed`, `skipped`), step version, input hash, outputs, step `data` (for example the probed fps, resolution and duration from `extract`), timings, and the error message on failure. Writes are atomic. Changing the schema requires bumping `SCHEMA_VERSION` and a migration note.

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

Model code is never called directly from a step. Each capability is defined by a small interface (for example a `Protocol` in `src/layerlab/backends/`), such as a speech transcriber or a source separator. Concrete backends implement the interface. Steps depend only on the interface, so a backend can be swapped without touching step logic.

When adding a backend:

- Record the model's **licence** for both code and weights in the backend's docstring and in `NOTICE` if it is bundled.
- Prefer `safetensors`, ONNX, or GGUF weights. Never load pickled checkpoints from untrusted sources.
- Declare hardware needs (CPU, CUDA, Apple Silicon) so `layerlab doctor` can report them.
- Keep heavy imports (`torch`, model libraries) inside the backend module so `layerlab --version` and `layerlab doctor` stay fast and work without them.

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
  steps/          # extract (real), audio, analyze, inpaint, export (placeholders)
tests/
  conftest.py     # synthetic_video / silent_video fixtures generated by ffmpeg
```

## Tests and media

Never commit binary media. Tests that need video use the `synthetic_video` fixture (colour bars, a 440 Hz sine tone, a drawtext overlay) or `silent_video`, both generated by ffmpeg into a temp folder. They skip when ffmpeg is not on PATH, except in CI, where `LAYERLAB_REQUIRE_FFMPEG=1` makes them fail instead. Runner behaviour is tested with fake steps in `tests/test_pipeline.py` and needs no ffmpeg.

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
layerlab run input.mp4 --out ./output [--skip <step>] [--force]
pre-commit install
```

## Things not to do

- Do not upload media or call remote APIs by default.
- Do not commit model weights, footage, or job outputs (see `.gitignore`).
- Do not make steps depend on each other's internal code; go through the manifest and job folder.
- Do not import heavy ML libraries at module import time in CLI or doctor code paths.
- Do not commit or push unless asked.
