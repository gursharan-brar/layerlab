"""The audio step through the pipeline, on a synthetic mix with known stems.

A fake extract step writes the synthetic mix as `audio.wav`, so no ffmpeg is
needed. The separation backends are the fakes installed by conftest.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from layerlab.job import Job
from layerlab.pipeline import StepFailed, run_pipeline
from layerlab.step import StepResult, StepSpec
from layerlab.steps import audio
from synthetic_audio import BURST_TIMES, make_mix, read_wav, snr_db, write_wav

MIX = make_mix(6.0)
AUDIO = StepSpec(audio.NAME, audio.VERSION, audio.inputs, audio.run, audio.config)


def _fake_extract(job: Job) -> StepResult:
    write_wav(job.work_dir / "audio.wav", MIX.mix)
    return StepResult(outputs={"audio": "audio.wav"})


EXTRACT = StepSpec("extract", 1, lambda job: [job.source_path], _fake_extract)
STEPS = (EXTRACT, AUDIO)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"stand-in for a video")
    return path


def _run(source: Path, out: Path, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
    report = run_pipeline(source, out, steps=STEPS, console=Console(quiet=True), **kwargs)
    manifest = json.loads((report.job_root / "manifest.json").read_text("utf-8"))
    return report.job_root, manifest


def test_default_backend_writes_three_stems_close_to_ground_truth(
    source: Path, tmp_path: Path
) -> None:
    job_root, manifest = _run(source, tmp_path / "out")
    record = manifest["steps"]["audio"]
    assert record["status"] == "succeeded"
    assert record["outputs"] == {
        "dialogue": "steps/audio/dialogue.wav",
        "music": "steps/audio/music.wav",
        "effects": "steps/audio/effects.wav",
        "sfx_events": "steps/audio/sfx_events.json",
    }
    for stem in ("dialogue", "music", "effects"):
        estimate, rate = read_wav(job_root / record["outputs"][stem])
        assert rate == MIX.rate
        assert estimate.shape == MIX.stems[stem].shape
        assert snr_db(MIX.stems[stem], estimate) > 20, stem


def test_mix_check_is_recorded(source: Path, tmp_path: Path) -> None:
    _, manifest = _run(source, tmp_path / "out")
    data = manifest["steps"]["audio"]["data"]
    check = data["mix_check"]
    # The fake backend's stems sum to the mix exactly; only float32 rounding remains.
    assert check["residual_db"] is None or check["residual_db"] < -100
    assert check["within_tolerance"] is True
    assert check["summed_stems"] == ["dialogue", "music", "effects"]
    assert data["backend"] == "tiger"
    assert data["chunks"] == 5
    assert data["stems"]["effects"] == {"available": True}


def test_sfx_events_match_the_known_bursts(source: Path, tmp_path: Path) -> None:
    job_root, manifest = _run(source, tmp_path / "out")
    document = json.loads((job_root / "steps/audio/sfx_events.json").read_text("utf-8"))
    times = [event["time"] for event in document["events"]]
    assert len(times) == len(BURST_TIMES)
    for expected in BURST_TIMES:
        assert any(abs(time - expected) <= 0.03 for time in times)
    assert manifest["steps"]["audio"]["data"]["sfx_event_count"] == len(BURST_TIMES)


def test_demucs_backend_marks_effects_unavailable(source: Path, tmp_path: Path) -> None:
    job_root, manifest = _run(source, tmp_path / "out", options={audio.OPTION: "demucs"})
    record = manifest["steps"]["audio"]
    assert set(record["outputs"]) == {"dialogue", "music", "sfx_events"}
    assert not (job_root / "steps/audio/effects.wav").exists()
    assert record["data"]["stems"]["effects"]["available"] is False
    assert record["data"]["mix_check"]["summed_stems"] == ["dialogue", "music"]
    assert record["data"]["mix_check"]["within_tolerance"] is True
    events = json.loads((job_root / "steps/audio/sfx_events.json").read_text("utf-8"))
    assert events["available"] is False
    assert events["events"] == []


def test_changing_backend_reruns_only_the_audio_step(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(source, out)
    report = run_pipeline(
        source, out, steps=STEPS, options={audio.OPTION: "demucs"}, console=Console(quiet=True)
    )
    assert report.outcomes() == {"extract": "cached", "audio": "ran"}
    again = run_pipeline(
        source, out, steps=STEPS, options={audio.OPTION: "demucs"}, console=Console(quiet=True)
    )
    assert again.outcomes() == {"extract": "cached", "audio": "cached"}


def test_out_of_tolerance_residual_is_flagged(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from layerlab.backends import catalog
    from synthetic_audio import BandSplitSeparator

    monkeypatch.setitem(
        catalog.BACKENDS,
        "tiger",
        catalog.BackendInfo("tiger", (), lambda _d, _s: BandSplitSeparator(gain=0.5), {}),
    )
    _, manifest = _run(source, tmp_path / "out")
    check = manifest["steps"]["audio"]["data"]["mix_check"]
    assert check["residual_db"] == pytest.approx(-6.02, abs=0.01)
    assert check["within_tolerance"] is False


def test_unknown_backend_fails_the_step(source: Path, tmp_path: Path) -> None:
    with pytest.raises(StepFailed, match="unknown audio backend 'nope'"):
        _run(source, tmp_path / "out", options={audio.OPTION: "nope"})
