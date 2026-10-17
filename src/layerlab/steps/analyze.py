"""Analyze step: cuts, text overlays and static logos. Reads the video, never changes it.

Outputs, in the step folder:

- `shots.json`: shots from PySceneDetect's ContentDetector (start, end, duration).
- `overlays.json`: overlay tracks. `text` tracks come from PaddleOCR text
  detection on frames sampled at `sampling.fps`; `logo` tracks from per-pixel
  temporal stability. Each has start, end, bounding box and its mask files.
- `masks/<track id>/<sample>.png`: one mask per sample a track covers.
- `debug_preview.mp4` (with `--debug-preview`): the video with boxes and cuts drawn on.

Thresholds come from `layerlab/analyze/defaults.toml`, overridable with
`--analyze-config`. Detectors live in `layerlab/analyze/` (cuts,
text_overlays, logos) and `layerlab/backends/text_detection.py`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from layerlab.analyze.config import AnalyzeConfig, TextConfig, load_config
from layerlab.analyze.overlays import Mask, OverlayTrack, box_mask
from layerlab.backends.text_detection import (
    MODELS as TEXT_MODELS,
)
from layerlab.backends.text_detection import (
    DBSettings,
    PaddleTextDetector,
    TextDetector,
)
from layerlab.job import Job
from layerlab.models import load_registry
from layerlab.step import StepResult

NAME = "analyze"
VERSION = 1

CONFIG_OPTION = "analyze_config"
PREVIEW_OPTION = "debug_preview"
SHOTS_FILE = "shots.json"
OVERLAYS_FILE = "overlays.json"
MASKS_DIR = "masks"
PREVIEW_FILE = "debug_preview.mp4"

# Shares of the step's progress bar.
_CUTS, _SAMPLES, _PRESENCE = 0.3, 0.5, 0.1


def load_text_detector(device: str, settings: TextConfig, on_status: Any) -> TextDetector:
    db = DBSettings(
        limit_side_len=settings.limit_side_len,
        pixel_threshold=settings.pixel_threshold,
        box_threshold=settings.box_threshold,
        unclip_ratio=settings.unclip_ratio,
    )
    return PaddleTextDetector.load(device, db, on_status)


def _settings(job: Job) -> AnalyzeConfig:
    override = job.options.get(CONFIG_OPTION)
    return load_config(Path(override) if override else None)


def _preview(job: Job) -> bool:
    return job.options.get(PREVIEW_OPTION, "") not in ("", "0", "false")


def config(job: Job) -> dict[str, Any]:
    registry = load_registry()
    return {
        "settings": _settings(job).to_dict(),
        "text_models": {name: registry.get(name).sha256 for name in TEXT_MODELS},
        "debug_preview": _preview(job),
    }


def inputs(job: Job) -> list[Path]:
    return [job.output("extract", "video")]


def run(job: Job) -> StepResult:
    import cv2

    from layerlab.analyze.cuts import detect_shots, shots_document
    from layerlab.analyze.frames import fit_size, iter_frames, probe_video
    from layerlab.analyze.logos import StabilityAccumulator
    from layerlab.analyze.overlays import track_document, write_masks
    from layerlab.analyze.text_overlays import TextTracker

    settings = _settings(job)
    video = job.output("extract", "video")
    before = video.stat()
    info = probe_video(video)
    out = job.work_dir
    sample_fps = settings.sampling.fps
    expected = max(1, math.ceil(info.duration * sample_fps))
    video_size = (info.width, info.height)

    job.report_status("detecting cuts")
    shots = detect_shots(video, info, settings.cuts)
    (out / SHOTS_FILE).write_text(
        json.dumps(shots_document(shots, info, settings.cuts), indent=2) + "\n", encoding="utf-8"
    )
    job.report_progress(_CUTS)

    job.report_status("loading text detector")
    detector = load_text_detector(job.device, settings.text, job.report_status)
    analysis = fit_size(info, settings.sampling.max_side)
    logo_size = fit_size(info, settings.logos.analysis_width)
    tracker = TextTracker(settings.text, (info.width / analysis[0], info.height / analysis[1]))
    stability = StabilityAccumulator(logo_size, settings.logos)

    job.report_status("detecting text and logos")
    samples = 0
    for frame in iter_frames(video, analysis, fps=sample_fps):
        tracker.update(frame.index, detector.detect(frame.image))
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, logo_size, interpolation=cv2.INTER_AREA)
        stability.update(np.asarray(small, dtype=np.uint8))
        samples = frame.index + 1
        job.report_progress(_CUTS + _SAMPLES * min(samples / expected, 1.0))

    tracks: list[OverlayTrack] = []
    for number, text in enumerate(tracker.tracks(), start=1):
        filled = text.filled()
        tracks.append(
            OverlayTrack(
                id=f"text-{number:03d}",
                kind="text",
                samples=sorted(filled),
                bbox=text.bbox,
                confidence=text.confidence,
                boxes=filled,
            )
        )

    regions = stability.regions()
    region_masks: dict[str, Any] = {}
    if regions:
        job.report_status("tracking logos")
        mode = stability.mode
        present: list[list[int]] = [[] for _ in regions]
        for frame in iter_frames(video, logo_size, fps=sample_fps):
            luma = np.asarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
            for index, region in enumerate(regions):
                if stability.presence(region, luma, mode):
                    present[index].append(frame.index)
            job.report_progress(
                _CUTS + _SAMPLES + _PRESENCE * min((frame.index + 1) / expected, 1.0)
            )
        sx, sy = info.width / logo_size[0], info.height / logo_size[1]
        number = 0
        for region, seen in zip(regions, present, strict=True):
            if not seen:
                continue
            number += 1
            x, y, w, h = region.bbox
            bbox = (
                math.floor(x * sx),
                math.floor(y * sy),
                math.ceil(w * sx),
                math.ceil(h * sy),
            )
            tracks.append(
                OverlayTrack(
                    id=f"logo-{number:03d}",
                    kind="logo",
                    samples=seen,
                    bbox=bbox,
                    confidence=region.stable_fraction,
                    boxes={sample: bbox for sample in seen},
                    details={
                        "stable_fraction": region.stable_fraction,
                        "temporal_std": region.temporal_std,
                        "edge_density": region.edge_density,
                    },
                )
            )
            region_masks[tracks[-1].id] = region.mask

    job.report_status("writing masks")
    documents = []
    for track in tracks:
        if track.kind == "text":
            mask_for = _text_masks(track, video_size, settings.text.mask_padding)
        else:
            mask = _region_mask(region_masks[track.id], video_size, settings.logos.mask_padding)
            mask_for = _constant_mask(mask)
        files = write_masks(out, track, mask_for)
        documents.append(track_document(track, sample_fps, info.duration, files))
    (out / MASKS_DIR).mkdir(exist_ok=True)

    overlays = {
        "version": 1,
        "video": asdict(info),
        "sample_fps": sample_fps,
        "sample_count": samples,
        "mask_format": "8-bit PNG at video resolution, 255 = overlay",
        "detectors": {
            "text": {**detector.describe(), **asdict(settings.text)},
            "logos": asdict(settings.logos),
        },
        "tracks": documents,
    }
    (out / OVERLAYS_FILE).write_text(json.dumps(overlays, indent=2) + "\n", encoding="utf-8")

    outputs = {"shots": SHOTS_FILE, "overlays": OVERLAYS_FILE, "masks": MASKS_DIR}
    if _preview(job):
        from layerlab.analyze.preview import render_preview

        job.report_status("rendering debug preview")
        render_preview(
            video,
            out / PREVIEW_FILE,
            info,
            shots,
            tracks,
            sample_fps,
            settings.preview,
            on_progress=lambda f: job.report_progress(_CUTS + _SAMPLES + _PRESENCE + 0.1 * f),
        )
        outputs["debug_preview"] = PREVIEW_FILE
    job.report_progress(1.0)

    return StepResult(
        outputs=outputs,
        data={
            "video_unchanged": _unchanged(video, before),
            "shot_count": len(shots),
            "cut_count": len(shots) - 1,
            "text_track_count": sum(1 for t in tracks if t.kind == "text"),
            "logo_track_count": sum(1 for t in tracks if t.kind == "logo"),
            "sample_count": samples,
            "sample_fps": sample_fps,
            "text_detector": detector.describe(),
        },
    )


def _text_masks(track: OverlayTrack, size: tuple[int, int], padding: int) -> Callable[[int], Mask]:
    def mask_for(sample: int) -> Mask:
        return box_mask(track.boxes[sample], size, padding)

    return mask_for


def _constant_mask(mask: Mask) -> Callable[[int], Mask]:
    def mask_for(_sample: int) -> Mask:
        return mask

    return mask_for


def _region_mask(small: Any, size: tuple[int, int], padding: int) -> Mask:
    """A logo region mask scaled from analysis size to video size and grown by `padding`."""
    import cv2
    import numpy as np

    mask = cv2.resize(small.astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
    if padding > 0:
        kernel = np.ones((2 * padding + 1, 2 * padding + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return np.asarray(mask, dtype=np.uint8)


def _unchanged(video: Path, before: Any) -> bool:
    """The step only reads the video; fail loudly if anything changed it."""
    from layerlab.errors import LayerlabError

    after = video.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise LayerlabError(f"{video} changed during analysis; it must only be read")
    return True
