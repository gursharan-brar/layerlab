"""Shared fixtures. Test media is generated with ffmpeg; no binary media is committed."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SYNTHETIC_DURATION_SECONDS = 2
SYNTHETIC_FPS = 25
SYNTHETIC_SIZE = (320, 240)

# Candidate fonts for drawtext, so the overlay does not depend on fontconfig.
_FONT_CANDIDATES = {
    "win32": ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"],
    "darwin": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    "linux": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ],
}


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        message = "ffmpeg and ffprobe must be on PATH"
        # CI sets this so a missing ffmpeg fails the run instead of silently skipping.
        if os.environ.get("LAYERLAB_REQUIRE_FFMPEG"):
            pytest.fail(message)
        pytest.skip(message)
    return ffmpeg


def _drawtext_filter() -> str:
    options = [
        "text='layerlab'",
        "fontsize=32",
        "fontcolor=white",
        "box=1",
        "boxcolor=black@0.6",
        "x=(w-text_w)/2",
        "y=h-60",
    ]
    for candidate in _FONT_CANDIDATES.get(sys.platform, []):
        if Path(candidate).is_file():
            # Colons separate filter options, so the drive letter's colon is escaped.
            escaped = Path(candidate).as_posix().replace(":", "\\:")
            options.insert(0, f"fontfile='{escaped}'")
            break
    return "drawtext=" + ":".join(options)


def _generate(path: Path, *, with_audio: bool) -> Path:
    ffmpeg = _require_ffmpeg()
    width, height = SYNTHETIC_SIZE
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"smptebars=size={width}x{height}:rate={SYNTHETIC_FPS}:duration={SYNTHETIC_DURATION_SECONDS}",
    ]
    if with_audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=44100:duration={SYNTHETIC_DURATION_SECONDS}",
        ]
    command += ["-vf", _drawtext_filter(), "-c:v", "mpeg4", "-q:v", "5"]
    command += ["-c:a", "aac", "-b:a", "96k", "-shortest"] if with_audio else ["-an"]
    command.append(str(path))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"could not generate test video: {result.stderr.strip()}")
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two seconds of colour bars with a drawtext overlay and a 440 Hz mono tone."""
    return _generate(tmp_path_factory.mktemp("media") / "synthetic.mp4", with_audio=True)


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Colour bars with a drawtext overlay and no audio stream."""
    return _generate(tmp_path_factory.mktemp("media") / "silent.mp4", with_audio=False)


def ffprobe_streams(path: Path) -> list[dict[str, Any]]:
    """Streams of a media file as reported by ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    streams: list[dict[str, Any]] = json.loads(result.stdout)["streams"]
    return streams
