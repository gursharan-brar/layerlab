"""The extract step and `layerlab run`, end to end, on a synthetic ffmpeg-generated video."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import (
    SYNTHETIC_DURATION_SECONDS,
    SYNTHETIC_FPS,
    SYNTHETIC_SIZE,
    ffprobe_streams,
)
from layerlab.cli import app
from layerlab.hashing import sha256_file
from layerlab.job import JOB_ID_LENGTH
from layerlab.steps import STEP_NAMES

runner = CliRunner()


def _run(*args: str) -> Any:
    return runner.invoke(app, ["run", *args])


def _job_root(out: Path, video: Path) -> Path:
    return out / sha256_file(video)[:JOB_ID_LENGTH]


def _manifest(job_root: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((job_root / "manifest.json").read_text("utf-8"))
    return data


@pytest.fixture(scope="module")
def completed_job(synthetic_video: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("out")
    result = _run(str(synthetic_video), "--out", str(out))
    assert result.exit_code == 0, result.output
    return _job_root(out, synthetic_video)


def test_run_writes_manifest_with_every_step(completed_job: Path) -> None:
    manifest = _manifest(completed_job)
    assert list(manifest["steps"]) == list(STEP_NAMES)
    assert all(step["status"] == "succeeded" for step in manifest["steps"].values())


def test_extract_records_media_properties(completed_job: Path) -> None:
    media = _manifest(completed_job)["steps"]["extract"]["data"]
    assert media["fps"] == SYNTHETIC_FPS
    assert media["fps_rational"] == f"{SYNTHETIC_FPS}/1"
    assert (media["width"], media["height"]) == SYNTHETIC_SIZE
    assert media["duration_seconds"] == pytest.approx(SYNTHETIC_DURATION_SECONDS, abs=0.1)
    assert media["video_codec"] == "mpeg4"


def test_extract_writes_48k_stereo_wav(completed_job: Path) -> None:
    outputs = _manifest(completed_job)["steps"]["extract"]["outputs"]
    assert outputs["audio"] == "steps/extract/audio.wav"
    (stream,) = ffprobe_streams(completed_job / outputs["audio"])
    assert stream["codec_name"] == "pcm_s16le"
    assert int(stream["sample_rate"]) == 48_000
    assert stream["channels"] == 2


def test_extract_writes_stream_copied_video_without_audio(completed_job: Path) -> None:
    outputs = _manifest(completed_job)["steps"]["extract"]["outputs"]
    assert outputs["video"] == "steps/extract/video_only.mp4"
    streams = ffprobe_streams(completed_job / outputs["video"])
    assert [stream["codec_type"] for stream in streams] == ["video"]
    # Stream copy keeps the source codec instead of re-encoding.
    assert streams[0]["codec_name"] == "mpeg4"
    assert (streams[0]["width"], streams[0]["height"]) == SYNTHETIC_SIZE


def test_rerun_resumes_from_cache(synthetic_video: Path, tmp_path: Path) -> None:
    assert _run(str(synthetic_video), "--out", str(tmp_path)).exit_code == 0
    job_root = _job_root(tmp_path, synthetic_video)
    first = _manifest(job_root)["steps"]["extract"]["started_at"]

    result = _run(str(synthetic_video), "--out", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert "cached" in result.output
    assert _manifest(job_root)["steps"]["extract"]["started_at"] == first


def test_force_reruns_extract(synthetic_video: Path, tmp_path: Path) -> None:
    assert _run(str(synthetic_video), "--out", str(tmp_path)).exit_code == 0
    result = _run(str(synthetic_video), "--out", str(tmp_path), "--force")
    assert result.exit_code == 0, result.output
    assert "cached" not in result.output


def test_skip_steps(synthetic_video: Path, tmp_path: Path) -> None:
    result = _run(
        str(synthetic_video), "--out", str(tmp_path), "--skip", "audio", "--skip", "export"
    )
    assert result.exit_code == 0, result.output
    steps = _manifest(_job_root(tmp_path, synthetic_video))["steps"]
    assert steps["audio"]["status"] == "skipped"
    assert steps["export"]["status"] == "skipped"
    assert steps["analyze"]["status"] == "succeeded"


def test_unknown_skip_is_a_usage_error(synthetic_video: Path, tmp_path: Path) -> None:
    result = _run(str(synthetic_video), "--out", str(tmp_path), "--skip", "nope")
    assert result.exit_code == 2
    assert not any(tmp_path.iterdir())


def test_video_without_audio_fails_clearly(silent_video: Path, tmp_path: Path) -> None:
    result = _run(str(silent_video), "--out", str(tmp_path))
    assert result.exit_code == 1
    assert "no audio stream" in result.output
    assert "Fix the problem and run again" in result.output
    extract = _manifest(_job_root(tmp_path, silent_video))["steps"]["extract"]
    assert extract["status"] == "failed"
    assert "no audio stream" in extract["error"]


def test_missing_input_is_a_usage_error(tmp_path: Path) -> None:
    result = _run(str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "out"))
    assert result.exit_code == 2
