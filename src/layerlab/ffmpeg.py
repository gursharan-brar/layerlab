"""Thin wrappers around the ffmpeg and ffprobe command-line tools.

Commands are always built as argument lists and never passed through a shell.
"""

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from layerlab.errors import LayerlabError

_ERROR_TAIL_LINES = 5


class FFmpegError(LayerlabError):
    """ffmpeg or ffprobe is missing or failed."""


def find_tool(name: str) -> str:
    """Absolute path of `ffmpeg` or `ffprobe`, or a clear error if it is not on PATH."""
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(
            f"{name} not found on PATH. Install ffmpeg (it includes {name}) "
            "and check with `layerlab doctor`."
        )
    return path


def probe(path: Path) -> dict[str, Any]:
    """Return ffprobe's JSON description of the streams and container of `path`."""
    result = subprocess.run(
        [
            find_tool("ffprobe"),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe could not read {path}: {_tail(result.stderr)}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FFmpegError(f"ffprobe returned unexpected output for {path}")
    return data


def run(
    args: Sequence[str],
    *,
    duration: float | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Run ffmpeg with `args`, reporting progress as a fraction of `duration` seconds."""
    command = [
        find_tool("ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        *args,
    ]
    # stderr goes to a file so a chatty ffmpeg can never block on a full pipe.
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None:  # pragma: no cover - stdout is always piped above
            raise FFmpegError("could not read ffmpeg progress")
        for line in process.stdout:
            if on_progress is not None:
                _report(line, duration, on_progress)
        returncode = process.wait()
        if returncode != 0:
            stderr.seek(0)
            message = _tail(stderr.read().decode("utf-8", errors="replace"))
            raise FFmpegError(f"ffmpeg failed with exit code {returncode}: {message}")


def _report(line: str, duration: float | None, on_progress: Callable[[float], None]) -> None:
    key, _, value = line.strip().partition("=")
    if key == "progress" and value == "end":
        on_progress(1.0)
    elif key in ("out_time_us", "out_time_ms") and duration:
        # Both keys carry microseconds; out_time_ms is misnamed in ffmpeg.
        try:
            seconds = int(value) / 1_000_000
        except ValueError:
            return
        on_progress(min(seconds / duration, 1.0))


def _tail(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-_ERROR_TAIL_LINES:]) or "no error output"
