"""Audio step: separate the soundtrack into dialogue, music and effects stems.

Backends (selected with `--audio-backend`, recorded in the input hash):

- `tiger` (default): TIGER-DnR, int8 ONNX, onnxruntime. dialogue, music, effects.
- `demucs`: htdemucs_ft via `layerlab[demucs]`. dialogue (vocals) and music;
  effects are reported unavailable.

The audio is streamed through the backend in overlapping chunks with a
crossfade, so memory use does not grow with the file length. The step then
checks that the stems add back up to the mix and records the residual, and
detects sound-effect events in effects.wav (written to sfx_events.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from layerlab.backends.catalog import DEFAULT_BACKEND, get_backend
from layerlab.job import Job
from layerlab.step import StepResult

NAME = "audio"
VERSION = 2

OPTION = "audio_backend"
EVENTS_FILE = "sfx_events.json"
RESIDUAL_TOLERANCE_DB = -20.0
"""Residual energy, relative to the mix, above which the stems are flagged."""

# Fractions of the step's progress bar.
_SEPARATION_SHARE = 0.9


def config(job: Job) -> dict[str, Any]:
    return get_backend(job.options.get(OPTION, DEFAULT_BACKEND)).config()


def inputs(job: Job) -> list[Path]:
    return [job.output("extract", "audio")]


def run(job: Job) -> StepResult:
    from layerlab.audio.chunking import separate_file
    from layerlab.audio.sfx import detect_events, events_document

    backend = get_backend(job.options.get(OPTION, DEFAULT_BACKEND))
    job.report_status(f"loading {backend.name}")
    separator = backend.load(job.device, job.report_status)
    job.report_status(f"separating ({backend.name})")

    mix = job.output("extract", "audio")
    destinations = {stem: job.work_dir / f"{stem}.wav" for stem in separator.stems}
    stats = separate_file(
        mix,
        destinations,
        separator,
        chunk_seconds=separator.chunk_seconds,
        overlap_seconds=separator.overlap_seconds,
        on_progress=lambda fraction: job.report_progress(fraction * _SEPARATION_SHARE),
    )

    residual_db = stats.residual_db
    residual: dict[str, Any] = {
        "residual_db": _finite_or_none(residual_db),
        "max_abs_residual": round(stats.max_abs_residual, 6),
        "tolerance_db": RESIDUAL_TOLERANCE_DB,
        "within_tolerance": residual_db is None or residual_db <= RESIDUAL_TOLERANCE_DB,
        "summed_stems": list(separator.stems),
    }

    stems: dict[str, Any] = {stem: {"available": True} for stem in separator.stems}
    stems.update(
        {
            stem: {"available": False, "reason": reason}
            for stem, reason in separator.unavailable.items()
        }
    )

    events_path = job.work_dir / EVENTS_FILE
    if "effects" in destinations:
        job.report_status("detecting SFX events")
        events, rate = detect_events(
            destinations["effects"],
            on_progress=lambda fraction: job.report_progress(
                _SEPARATION_SHARE + fraction * (1 - _SEPARATION_SHARE)
            ),
        )
        document = events_document(events, rate, "effects.wav")
    else:
        events = []
        document = {
            "version": 1,
            "source": None,
            "available": False,
            "reason": separator.unavailable.get("effects", "no effects stem"),
            "events": [],
        }
    events_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    job.report_progress(1.0)

    outputs = {stem: path.name for stem, path in destinations.items()}
    outputs["sfx_events"] = EVENTS_FILE
    return StepResult(
        outputs=outputs,
        data={
            "backend": backend.name,
            **separator.describe(),
            "sample_rate": stats.sample_rate,
            "channels": stats.channels,
            "duration_seconds": round(stats.frames / stats.sample_rate, 3)
            if stats.sample_rate
            else 0.0,
            "chunks": stats.chunks,
            "chunk_seconds": separator.chunk_seconds,
            "overlap_seconds": separator.overlap_seconds,
            "stems": stems,
            "mix_check": residual,
            "sfx_event_count": len(events),
        },
    )


def _finite_or_none(value: float | None) -> float | None:
    """JSON has no infinity: an exact reconstruction is recorded as None."""
    if value is None or value == float("-inf"):
        return None
    return round(value, 2)
