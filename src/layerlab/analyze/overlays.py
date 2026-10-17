"""Overlay tracks, their mask PNG sequences, and overlays.json.

Masks are single-channel 8-bit PNGs at the video's resolution: 255 where the
overlay is, 0 elsewhere. A track has one mask per sample it covers, named
`masks/<track id>/<sample index, 6 digits>.png`. Sample k starts at
k / sample_fps seconds and its mask applies until the next sample.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Box = tuple[int, int, int, int]
Mask = NDArray[np.uint8]


@dataclass
class OverlayTrack:
    id: str
    kind: str  # "text" or "logo"
    samples: list[int]
    bbox: Box
    confidence: float
    boxes: dict[int, Box] = field(default_factory=dict)  # per-sample boxes, video pixels
    details: dict[str, Any] = field(default_factory=dict)


def write_masks(
    root: Path,
    track: OverlayTrack,
    mask_for: Callable[[int], Mask],
) -> list[str]:
    """Write one PNG per sample of `track`; return their paths relative to `root`."""
    import cv2

    folder = root / "masks" / track.id
    folder.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    cache: dict[int, bytes] = {}
    for sample in track.samples:
        mask = mask_for(sample)
        key = hash(mask.tobytes())
        if key not in cache:
            ok, encoded = cv2.imencode(".png", mask)
            if not ok:  # pragma: no cover - PNG encoding of a uint8 image does not fail
                raise RuntimeError("could not encode mask PNG")
            cache[key] = encoded.tobytes()
        path = folder / f"{sample:06d}.png"
        path.write_bytes(cache[key])
        written.append(path.relative_to(root).as_posix())
    return written


def box_mask(box: Box, size: tuple[int, int], padding: int) -> Mask:
    """A filled rectangle mask, grown by `padding` and clipped to the frame."""
    width, height = size
    x, y, w, h = box
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(width, x + w + padding), min(height, y + h + padding)
    mask[y0:y1, x0:x1] = 255
    return mask


def track_document(
    track: OverlayTrack, sample_fps: float, duration: float, mask_files: list[str]
) -> dict[str, Any]:
    step = 1.0 / sample_fps
    start = track.samples[0] * step
    end = (
        min((track.samples[-1] + 1) * step, duration)
        if duration
        else (track.samples[-1] + 1) * step
    )
    x, y, w, h = track.bbox
    return {
        "id": track.id,
        "kind": track.kind,
        "start": round(start, 4),
        "end": round(end, 4),
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "confidence": track.confidence,
        "sample_count": len(track.samples),
        **track.details,
        "masks": [
            {"time": round(sample * step, 4), "file": path}
            for sample, path in zip(track.samples, mask_files, strict=True)
        ],
    }
