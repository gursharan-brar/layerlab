# layerlab

**layerlab** takes a video and separates it into editable layers:

- **Dialogue**
- **Music**
- **Sound effects**
- **On-screen graphics**
- **Edit timeline** (cuts, SFX hits, overlay appearances)

Everything runs locally on your machine. Your footage is not uploaded anywhere.

> **Status: pre-alpha.** The pipeline core and the `extract` step work. The
> separation, analysis, inpainting and export steps are placeholders.

## Requirements

- Python 3.10 or newer
- [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`
- Optional: an NVIDIA GPU or Apple Silicon for faster model steps

## Install (from source)

```bash
git clone <repository-url> layerlab
cd layerlab
python -m pip install -e ".[dev]"
```

## Usage

```bash
layerlab doctor
layerlab run input.mp4 --out ./output
```

`layerlab doctor` checks the Python version, whether ffmpeg is on `PATH`, and whether a GPU is available.

`layerlab run` creates a job folder under `--out`, named from the input file's hash, and runs the steps `extract`, `audio`, `analyze`, `inpaint` and `export` in order. Each step's results are recorded in the job's `manifest.json`.

- Running again resumes: steps whose inputs have not changed are skipped.
- `--skip <step>` skips a step (repeatable). `--force` re-runs everything.
- If a step fails, the error is recorded in the manifest, earlier outputs are kept, and the command exits non-zero.

The `extract` step writes `audio.wav` (48 kHz stereo) and `video_only.mp4` (the video stream, copied without re-encoding), and records fps, resolution and duration.

## Development

```bash
pre-commit install
ruff check . && ruff format --check .
mypy
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and [CLAUDE.md](CLAUDE.md) for the architecture.

## Security

Please report vulnerabilities privately. See [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
