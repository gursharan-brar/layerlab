"""A synthetic video with known cuts, a text overlay and a static logo, made by ffmpeg.

Three 2 s shots (moving test pattern, solid dark red, solid blue), so cuts are
at 2 s and 4 s. A caption is drawn from 1.0 s to 4.5 s (7 of 12 samples at
2 fps, so it is not static) and a small layered logo sits in the top-right
corner for the whole video. The moving test pattern also contains small
numbers, which the real text detector finds as short-lived text.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WIDTH, HEIGHT, FPS = 640, 360, 25
SHOT_SECONDS = 2
DURATION = 3 * SHOT_SECONDS
CUTS = (2.0, 4.0)
CAPTION = "BREAKING NEWS TONIGHT"
CAPTION_START, CAPTION_END = 1.0, 4.5
CAPTION_XY = (40, 290)
LOGO_BOX = (560, 12, 64, 40)  # x, y, width, height

_FONTS = {
    "win32": "C:/Windows/Fonts/arial.ttf",
    "darwin": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "linux": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}


def make_overlay_video(ffmpeg: str, path: Path) -> Path:
    size = f"{WIDTH}x{HEIGHT}"
    shot = f"rate={FPS}:duration={SHOT_SECONDS}"
    font = _FONTS.get(sys.platform)
    fontfile = ""
    if font and Path(font).is_file():
        fontfile = "fontfile='" + Path(font).as_posix().replace(":", "\\:") + "':"
    x, y = CAPTION_XY
    lx, ly, lw, lh = LOGO_BOX
    filters = ",".join(
        [
            "[0][1][2]concat=n=3:v=1:a=0",
            "format=yuv420p",
            f"drawtext={fontfile}text='{CAPTION}':fontsize=30:fontcolor=white:"
            f"box=1:boxcolor=black@0.85:boxborderw=8:x={x}:y={y}:"
            f"enable='between(t,{CAPTION_START},{CAPTION_END})'",
            # A layered logo: white plate, black inset, red mark.
            f"drawbox=x={lx}:y={ly}:w={lw}:h={lh}:color=white:t=fill",
            f"drawbox=x={lx + 8}:y={ly + 8}:w={lw - 16}:h={lh - 16}:color=black:t=fill",
            f"drawbox=x={lx + 20}:y={ly + 14}:w={lw - 40}:h={lh - 28}:color=red:t=fill",
        ]
    )
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:{shot}",
        "-f", "lavfi", "-i", f"color=c=0x601818:size={size}:{shot}",
        "-f", "lavfi", "-i", f"color=c=0x2a5db0:size={size}:{shot}",
        "-filter_complex", filters,
        "-c:v", "mpeg4", "-q:v", "3", "-an",
        str(path),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"could not generate overlay video: {result.stderr.strip()}")
    return path
