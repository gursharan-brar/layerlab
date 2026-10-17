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

Pre-alpha. The package skeleton, the `layerlab` CLI (`--version`, `doctor`), and tests exist. **No processing logic exists yet.** Do not add processing steps without an issue or a plan agreed with the maintainers.

## Architecture

layerlab is a **pipeline of independent steps**. Each step:

- reads its inputs from the **job folder**,
- writes its outputs to the job folder,
- records what it did, and its outputs, in **`manifest.json`** inside the job folder.

Steps never call each other directly. The pipeline runner decides the order by reading the manifest. A step can be re-run on its own, and a failed step must not corrupt another step's outputs.

### Job folder

```
<job>/
  manifest.json          # source of truth: job metadata, steps, status, output paths
  input/                 # the original video, referenced by the manifest, never modified
  steps/
    <step-name>/         # one folder per step, written only by that step
      ...outputs...
```

- The original file is never modified. Paths in the manifest are relative to the job folder.
- A step writes to a temporary location and renames on success, so partial outputs are never mistaken for complete ones.
- Outputs are plain, inspectable formats where practical: JSON for structured data, WAV or FLAC for audio, PNG for frames, MP4 or MOV for video.

### Manifest

`manifest.json` records, for each step: its name, version, status (`pending`, `running`, `succeeded`, `failed`), input and output paths, the backend used, timings, and any error message. Changing the manifest schema requires a version bump and a migration note.

### Steps

Planned steps map to the layers above (dialogue, music, sound effects, on-screen graphics, edit timeline). Each step is a module under `src/layerlab/steps/` (to be created) with a single entry point that takes the job folder and returns a result. Keep steps free of CLI code so they can be tested directly.

### Model backends behind interfaces

Model code is never called directly from a step. Each capability is defined by a small interface (for example a `Protocol` in `src/layerlab/backends/`), such as a speech transcriber or a source separator. Concrete backends implement the interface. Steps depend only on the interface, so a backend can be swapped without touching step logic.

When adding a backend:

- Record the model's **licence** for both code and weights in the backend's docstring and in `NOTICE` if it is bundled.
- Prefer `safetensors`, ONNX, or GGUF weights. Never load pickled checkpoints from untrusted sources.
- Declare hardware needs (CPU, CUDA, Apple Silicon) so `layerlab doctor` can report them.
- Keep heavy imports (`torch`, model libraries) inside the backend module so `layerlab --version` and `layerlab doctor` stay fast and work without them.

### Media tools

ffmpeg is the only required external tool. Call it through a single helper that takes an argument list, never a shell string. Check for it with `shutil.which` and report its absence through `layerlab doctor`.

## Code layout

```
src/layerlab/
  __init__.py     # __version__
  __main__.py     # python -m layerlab
  cli.py          # Typer app: --version, doctor (more commands added here)
  doctor.py       # environment checks, dependency-free
tests/            # pytest; mirror module names
```

## Conventions

- Python 3.10+. Use `from __future__ import annotations` only when needed for forward references.
- Type everything. `mypy --strict` must pass. Avoid `Any` at module boundaries.
- Lint and format with ruff (line length 100). Run `ruff check .` and `ruff format .` before committing.
- Tests use pytest. Mock external tools (ffmpeg, `nvidia-smi`) rather than requiring them in CI.
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
pre-commit install
```

## Things not to do

- Do not upload media or call remote APIs by default.
- Do not commit model weights, footage, or job outputs (see `.gitignore`).
- Do not make steps depend on each other's internal code; go through the manifest and job folder.
- Do not import heavy ML libraries at module import time in CLI or doctor code paths.
- Do not commit or push unless asked.
