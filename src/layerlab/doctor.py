"""Environment checks behind `layerlab doctor`.

Checks are deliberately dependency-free: they inspect the interpreter, PATH and
vendor tools (`nvidia-smi`) rather than importing heavy ML libraries.
"""

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

MIN_PYTHON: tuple[int, int] = (3, 10)
_PROBE_TIMEOUT_SECONDS = 10

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    """Result of one environment check."""

    name: str
    status: Status
    detail: str


def check_python(version: tuple[int, ...] = tuple(sys.version_info[:3])) -> Check:
    """Verify the interpreter meets the minimum supported Python version."""
    found = ".".join(str(part) for part in version)
    if tuple(version[:2]) < MIN_PYTHON:
        required = ".".join(str(part) for part in MIN_PYTHON)
        return Check("python", "fail", f"{found} found; layerlab requires {required}+")
    return Check("python", "ok", found)


def check_ffmpeg() -> Check:
    """Verify an ffmpeg binary is on PATH and can report its version."""
    path = shutil.which("ffmpeg")
    if path is None:
        return Check(
            "ffmpeg",
            "fail",
            "not found on PATH; install ffmpeg from https://ffmpeg.org/download.html",
        )
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("ffmpeg", "fail", f"found at {path} but could not run it: {exc}")
    first_line = result.stdout.splitlines()[0] if result.stdout else "version unknown"
    return Check("ffmpeg", "ok", f"{first_line} ({path})")


def check_gpu() -> Check:
    """Report whether an NVIDIA or Apple Silicon GPU is available. Absence is a warning."""
    if shutil.which("nvidia-smi") is not None:
        names = _nvidia_gpus()
        if names:
            return Check("gpu", "ok", "; ".join(names))
    if _is_apple_silicon():
        return Check("gpu", "ok", "Apple Silicon GPU (Metal)")
    return Check(
        "gpu",
        "warn",
        "no GPU detected; model steps will run on CPU and be slow",
    )


def run_checks() -> list[Check]:
    """Run every doctor check in display order."""
    return [check_python(), check_ffmpeg(), check_gpu()]


def _nvidia_gpus() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"
