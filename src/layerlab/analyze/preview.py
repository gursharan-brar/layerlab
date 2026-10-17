"""The --debug-preview video: every detected box and cut drawn onto the footage.

The source video is only read. Frames are decoded through one ffmpeg pipe,
drawn on with OpenCV, and encoded by a second ffmpeg into a new MP4.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from layerlab import ffmpeg
from layerlab.analyze.config import PreviewConfig
from layerlab.analyze.cuts import Shot
from layerlab.analyze.frames import VideoInfo, fit_size, iter_frames
from layerlab.analyze.overlays import OverlayTrack
from layerlab.errors import LayerlabError

_COLORS = {"text": (0, 220, 0), "logo": (0, 160, 255)}  # BGR
_CUT_COLOR = (0, 0, 255)


def render_preview(
    video: Path,
    destination: Path,
    info: VideoInfo,
    shots: list[Shot],
    tracks: list[OverlayTrack],
    sample_fps: float,
    settings: PreviewConfig,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    import cv2

    width, height = fit_size(info, settings.max_side)
    scale_x, scale_y = width / info.width, height / info.height
    fps = info.fps or 25.0
    total = max(1, round(info.duration * fps))
    cuts = [shot.start for shot in shots[1:]]

    command = [
        ffmpeg.find_tool("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
        "-c:v",
        "mpeg4",
        "-q:v",
        "4",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    with tempfile.TemporaryFile() as stderr:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr)
        if encoder.stdin is None:  # pragma: no cover - stdin is always piped above
            raise LayerlabError("could not start the preview encoder")
        try:
            for frame in iter_frames(video, (width, height)):
                time = frame.index / fps
                image = frame.image.copy()
                sample = int(time * sample_fps)
                for track in tracks:
                    box = track.boxes.get(sample)
                    if box is None:
                        continue
                    x, y, w, h = box
                    p0 = (round(x * scale_x), round(y * scale_y))
                    p1 = (round((x + w) * scale_x), round((y + h) * scale_y))
                    color = _COLORS.get(track.kind, (255, 255, 255))
                    cv2.rectangle(image, p0, p1, color, 2)
                    cv2.putText(
                        image,
                        track.id,
                        (p0[0], max(12, p0[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                if any(cut <= time < cut + settings.cut_marker_seconds for cut in cuts):
                    cv2.rectangle(image, (0, 0), (width - 1, height - 1), _CUT_COLOR, 8)
                    cv2.putText(
                        image,
                        "CUT",
                        (width // 2 - 30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        _CUT_COLOR,
                        3,
                        cv2.LINE_AA,
                    )
                shot = sum(1 for cut in cuts if cut <= time)
                label = f"{time:7.2f}s  shot {shot + 1}/{len(shots)}"
                cv2.putText(
                    image,
                    label,
                    (8, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                encoder.stdin.write(image.tobytes())
                if on_progress is not None:
                    on_progress(min(frame.index + 1, total) / total)
        finally:
            encoder.stdin.close()
            returncode = encoder.wait()
        if returncode != 0:
            stderr.seek(0)
            message = stderr.read().decode("utf-8", errors="replace").strip()
            raise LayerlabError(f"ffmpeg could not write the debug preview: {message[-500:]}")
