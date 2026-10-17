# layerlab

**layerlab** takes a video and separates it into editable layers:

- **Dialogue**
- **Music**
- **Sound effects**
- **On-screen graphics**
- **Edit timeline** (cuts, SFX hits, overlay appearances)

Everything runs locally on your machine. Your footage is not uploaded anywhere.

> **Status: pre-alpha.** The pipeline core and the `extract`, `audio` and
> `analyze` steps work. The inpainting and export steps are placeholders.

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

The `audio` step splits that audio into `dialogue.wav`, `music.wav` and `effects.wav`, and writes `sfx_events.json` with the time, duration, peak level and confidence of each sound-effect hit. It checks that the stems add back up to the original mix and records the residual in the manifest.

- Default backend: [TIGER-DnR](https://huggingface.co/JusperLee/TIGER-DnR) as an int8 ONNX model on onnxruntime (7 MB, CPU friendly). On a laptop CPU it runs at roughly 3-4x the length of the audio.
- `--audio-backend demucs`: Demucs htdemucs_ft. Install it with `pip install "layerlab[demucs]"` (this pulls in PyTorch). It produces dialogue (vocals) and music only. Effects stay mixed into music and are marked unavailable.

Long audio is processed in overlapping, crossfaded chunks, so memory use does not grow with the length of the video.

The `analyze` step reads the video (it never changes it) and writes:

- `shots.json`: cuts found by PySceneDetect, with each shot's start, end and duration.
- `overlays.json`: on-screen text (PaddleOCR text detection on frames sampled at 2 fps, grouped into tracks that persist in one region) and static logos (regions whose pixels stay nearly constant in 80% or more of the samples). Each track has a start, end and bounding box.
- `masks/<track>/`: a PNG mask per sampled frame for every track, ready for inpainting.
- `debug_preview.mp4`, with `--debug-preview`: the video with every detected box and cut drawn on.

Every detector threshold lives in [`src/layerlab/analyze/defaults.toml`](src/layerlab/analyze/defaults.toml). Override any of them with `--analyze-config my.toml`.

`--device` picks the compute device (`auto`, `cuda`, `rocm`, `mps` or `cpu`). By default layerlab uses the best one it finds, and records the choice in the manifest.

## Models

Model weights are not included in layerlab. They download on first use into your user cache folder, are checked against a SHA-256, and resume if interrupted.

```bash
layerlab models list            # registry with license and commercial-use columns
layerlab models fetch lama      # download ahead of time
```

Licenses are as stated by each model's publisher. Check them before commercial use. Set `LAYERLAB_CACHE_DIR` to move the cache.

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
