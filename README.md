# layerlab

**layerlab** takes a video and separates it into editable layers:

- **Dialogue**
- **Music**
- **Sound effects**
- **On-screen graphics**
- **Edit timeline** (cuts, SFX hits, overlay appearances)

Everything runs locally on your machine. Your footage is not uploaded anywhere.

> **Status: pre-alpha.** The package skeleton, CLI and environment checks exist.
> No processing is implemented yet.

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
layerlab --version
layerlab doctor
```

`layerlab doctor` checks the Python version, whether ffmpeg is on `PATH`, and whether a GPU is available.

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
