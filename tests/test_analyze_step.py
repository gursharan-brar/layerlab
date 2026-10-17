"""The analyze step end to end on an ffmpeg-generated video with known cuts, caption and logo.

Cuts and logos use the real detectors. Text detection uses a scripted fake that
reports the caption's box in the samples where it is on screen; the real
PaddleOCR model runs in the opt-in `real_models` test at the bottom.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from conftest import ScriptedTextDetector
from layerlab.backends.text_detection import TextBox
from layerlab.cli import app
from layerlab.job import Job
from layerlab.pipeline import run_pipeline
from layerlab.step import StepResult, StepSpec
from layerlab.steps import analyze
from synthetic_video import (
    CAPTION_END,
    CAPTION_START,
    CUTS,
    DURATION,
    HEIGHT,
    LOGO_BOX,
    WIDTH,
)

CAPTION_BOX = (39, 287, 398, 28)
SAMPLE_FPS = 2.0
CAPTION_SAMPLES = range(int(CAPTION_START * SAMPLE_FPS), int(CAPTION_END * SAMPLE_FPS))


def _caption_script() -> dict[int, list[TextBox]]:
    return {s: [TextBox(*CAPTION_BOX, score=0.99)] for s in CAPTION_SAMPLES}


def _steps(video: Path) -> tuple[StepSpec, StepSpec]:
    def extract(job: Job) -> StepResult:
        shutil.copy(video, job.work_dir / "video_only.mp4")
        return StepResult(outputs={"video": "video_only.mp4"})

    return (
        StepSpec("extract", 1, lambda job: [job.source_path], extract),
        StepSpec(analyze.NAME, analyze.VERSION, analyze.inputs, analyze.run, analyze.config),
    )


def _run(video: Path, out: Path, **options: str) -> Path:
    report = run_pipeline(
        video, out, steps=_steps(video), options=options, console=Console(quiet=True)
    )
    return report.job_root / "steps" / "analyze"


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analyze, "load_text_detector", lambda *_args: ScriptedTextDetector(_caption_script())
    )


@pytest.fixture(scope="module")
def analyzed(overlay_video: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            analyze, "load_text_detector", lambda *_a: ScriptedTextDetector(_caption_script())
        )
        return _run(overlay_video, tmp_path_factory.mktemp("analyze"), debug_preview="1")


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def test_shots_match_the_known_cuts(analyzed: Path) -> None:
    shots = _json(analyzed / "shots.json")
    assert shots["cuts"] == pytest.approx(list(CUTS), abs=0.05)
    assert [(s["start"], s["end"]) for s in shots["shots"]] == pytest.approx(
        [(0.0, 2.0), (2.0, 4.0), (4.0, float(DURATION))], abs=0.05
    )
    for shot in shots["shots"]:
        assert shot["duration"] == pytest.approx(shot["end"] - shot["start"])


def _track(analyzed: Path, kind: str) -> dict[str, Any]:
    tracks = [t for t in _json(analyzed / "overlays.json")["tracks"] if t["kind"] == kind]
    assert len(tracks) == 1, tracks
    track: dict[str, Any] = tracks[0]
    return track


def test_caption_becomes_one_text_track(analyzed: Path) -> None:
    track = _track(analyzed, "text")
    assert track["id"] == "text-001"
    assert (track["start"], track["end"]) == (CAPTION_START, CAPTION_END)
    x, y, w, h = CAPTION_BOX
    assert track["bbox"] == {"x": x, "y": y, "width": w, "height": h}
    assert track["sample_count"] == len(CAPTION_SAMPLES)


def test_static_logo_becomes_a_logo_track(analyzed: Path) -> None:
    track = _track(analyzed, "logo")
    assert (track["start"], track["end"]) == (0.0, float(DURATION))
    box = track["bbox"]
    lx, ly, lw, lh = LOGO_BOX
    assert abs(box["x"] - lx) <= 4 and abs(box["y"] - ly) <= 4
    assert abs(box["width"] - lw) <= 8 and abs(box["height"] - lh) <= 8
    assert track["stable_fraction"] >= 0.8


def test_masks_are_png_sequences_matching_the_tracks(analyzed: Path) -> None:
    for kind in ("text", "logo"):
        track = _track(analyzed, kind)
        assert len(track["masks"]) == track["sample_count"]
        first = track["masks"][0]
        assert first["file"] == f"masks/{track['id']}/{round(first['time'] * SAMPLE_FPS):06d}.png"
        mask = cv2.imread(str(analyzed / first["file"]), cv2.IMREAD_UNCHANGED)
        assert mask is not None
        assert mask.shape == (HEIGHT, WIDTH)
        assert mask.dtype == np.uint8
        box = track["bbox"]
        inside = mask[box["y"] : box["y"] + box["height"], box["x"] : box["x"] + box["width"]]
        assert int(np.min(inside)) == 255
        assert mask.sum() < 255 * WIDTH * HEIGHT * 0.2


def test_debug_preview_is_rendered(analyzed: Path) -> None:
    preview = analyzed / "debug_preview.mp4"
    assert preview.stat().st_size > 0
    capture = cv2.VideoCapture(str(preview))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert frames == pytest.approx(DURATION * 25, abs=2)


@pytest.mark.usefixtures("scripted")
def test_video_is_never_modified(overlay_video: Path, tmp_path: Path) -> None:
    before = hashlib.sha256(overlay_video.read_bytes()).hexdigest()
    step_dir = _run(overlay_video, tmp_path / "out")
    video = step_dir.parent / "extract" / "video_only.mp4"
    assert hashlib.sha256(video.read_bytes()).hexdigest() == before
    manifest = _json(step_dir.parents[1] / "manifest.json")
    assert manifest["steps"]["analyze"]["data"]["video_unchanged"] is True
    assert "debug_preview" not in manifest["steps"]["analyze"]["outputs"]


@pytest.mark.usefixtures("scripted")
def test_config_change_reruns_only_analyze(overlay_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(overlay_video, out)
    config = tmp_path / "strict.toml"
    config.write_text("[cuts]\nthreshold = 40.0\n", encoding="utf-8")
    report = run_pipeline(
        overlay_video,
        out,
        steps=_steps(overlay_video),
        options={analyze.CONFIG_OPTION: str(config)},
        console=Console(quiet=True),
    )
    assert report.outcomes() == {"extract": "cached", "analyze": "ran"}
    shots = _json(report.job_root / "steps/analyze/shots.json")
    assert shots["detector"]["threshold"] == 40.0


def test_cli_rejects_a_bad_analyze_config(overlay_video: Path, tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text("[cuts]\nthreshhold = 1\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(overlay_video),
            "--out",
            str(tmp_path / "out"),
            "--analyze-config",
            str(config),
        ],
    )
    assert result.exit_code == 2  # rejected before any step runs
    assert "threshhold" in result.output
    assert not (tmp_path / "out").exists()


@pytest.mark.real_models
def test_real_paddleocr_finds_the_caption(overlay_video: Path, tmp_path: Path) -> None:
    step_dir = _run(overlay_video, tmp_path / "out")
    tracks = _json(step_dir / "overlays.json")["tracks"]
    x, y, w, _h = CAPTION_BOX
    caption = [
        t
        for t in tracks
        if t["kind"] == "text"
        and abs(t["bbox"]["x"] - x) <= 10
        and abs(t["bbox"]["y"] - y) <= 10
        and abs(t["bbox"]["width"] - w) <= 20
    ]
    assert len(caption) == 1, tracks
    assert (caption[0]["start"], caption[0]["end"]) == (CAPTION_START, CAPTION_END)
