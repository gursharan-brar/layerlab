"""Cut detection with PySceneDetect's ContentDetector.

ContentDetector compares consecutive frames in HSV (optionally luma only) and
reports a cut where the weighted change exceeds `threshold`. Shots shorter
than `min_shot_seconds` are merged into their neighbours.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from layerlab.analyze.config import CutsConfig
from layerlab.analyze.frames import VideoInfo


@dataclass(frozen=True)
class Shot:
    index: int
    start: float
    end: float
    duration: float
    start_frame: int
    end_frame: int


def detect_shots(video: Path, info: VideoInfo, settings: CutsConfig) -> list[Shot]:
    """Shots covering the whole video, in order. One shot if there are no cuts."""
    from scenedetect import ContentDetector, detect

    min_frames = max(1, round(settings.min_shot_seconds * (info.fps or 25.0)))
    detector = ContentDetector(
        threshold=settings.threshold, min_scene_len=min_frames, luma_only=settings.luma_only
    )
    scenes = detect(str(video), detector)
    if not scenes:
        frames = round(info.duration * info.fps) if info.fps else 0
        return [Shot(0, 0.0, round(info.duration, 4), round(info.duration, 4), 0, frames)]
    shots = []
    for index, (start, end) in enumerate(scenes):
        start_s, end_s = _seconds(start), _seconds(end)
        shots.append(
            Shot(
                index=index,
                start=round(start_s, 4),
                end=round(end_s, 4),
                duration=round(end_s - start_s, 4),
                start_frame=_frame(start),
                end_frame=_frame(end),
            )
        )
    return shots


# PySceneDetect 0.7 replaced get_seconds()/get_frames() with properties; support both.


def _seconds(timecode: Any) -> float:
    value = getattr(timecode, "seconds", None)
    return float(value if value is not None else timecode.get_seconds())


def _frame(timecode: Any) -> int:
    value = getattr(timecode, "frame_num", None)
    return int(value if value is not None else timecode.get_frames())


def shots_document(shots: list[Shot], info: VideoInfo, settings: CutsConfig) -> dict[str, Any]:
    """The JSON written to shots.json. `end` is exclusive."""
    return {
        "version": 1,
        "fps": info.fps,
        "duration": info.duration,
        "detector": {"name": "PySceneDetect ContentDetector", **asdict(settings)},
        "cuts": [shot.start for shot in shots[1:]],
        "shots": [asdict(shot) for shot in shots],
    }
