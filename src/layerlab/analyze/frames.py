"""Reading video frames through ffmpeg, without ever writing to the video.

Frames are streamed as raw BGR from an ffmpeg pipe, one at a time, so memory
does not depend on the video's length.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from layerlab import ffmpeg
from layerlab.errors import LayerlabError

Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float


@dataclass(frozen=True)
class SampledFrame:
    index: int
    time: float
    image: Frame


def probe_video(path: Path) -> VideoInfo:
    probed = ffmpeg.probe(path)
    video = next((s for s in probed.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise LayerlabError(f"{path.name} has no video stream")
    rate = Fraction(0)
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = Fraction(str(video.get(key)))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            break
    try:
        duration = float(probed.get("format", {}).get("duration") or video.get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    return VideoInfo(int(video["width"]), int(video["height"]), float(rate), duration)


def fit_size(info: VideoInfo, max_side: int) -> tuple[int, int]:
    """Even-sized (width, height) no larger than `max_side`, keeping the aspect ratio."""
    scale = min(1.0, max_side / max(info.width, info.height))
    width = max(2, round(info.width * scale / 2) * 2)
    height = max(2, round(info.height * scale / 2) * 2)
    return width, height


def iter_frames(
    path: Path, size: tuple[int, int], *, fps: float | None = None
) -> Iterator[SampledFrame]:
    """Frames of `path` scaled to `size`, every frame or `fps` samples per second.

    With `fps`, sample k is the frame nearest to k / fps seconds.
    """
    width, height = size
    filters = [f"scale={width}:{height}:flags=area"]
    if fps is not None:
        filters.insert(0, f"fps={fps}")
    command = [
        ffmpeg.find_tool("ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        ",".join(filters),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    frame_bytes = width * height * 3
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
        if process.stdout is None:  # pragma: no cover - stdout is always piped above
            raise LayerlabError("could not read frames from ffmpeg")
        try:
            index = 0
            while True:
                data = process.stdout.read(frame_bytes)
                if len(data) < frame_bytes:
                    break
                image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                time = index / fps if fps is not None else float("nan")
                yield SampledFrame(index, time, image)
                index += 1
        finally:
            process.stdout.close()
            returncode = process.wait()
        if returncode != 0:
            stderr.seek(0)
            message = stderr.read().decode("utf-8", errors="replace").strip()
            raise LayerlabError(f"ffmpeg could not decode {path.name}: {message[-500:]}")
