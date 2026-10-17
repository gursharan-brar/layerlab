"""SFX event detection on a synthetic effects stem with bursts at known times."""

import json
from pathlib import Path

import numpy as np
import pytest

from layerlab.audio.sfx import detect_events, events_document
from synthetic_audio import BURST_AMPLITUDE, BURST_SECONDS, BURST_TIMES, RATE, make_mix, write_wav

TOLERANCE_SECONDS = 0.03


@pytest.fixture(scope="module")
def effects_events(tmp_path_factory: pytest.TempPathFactory) -> list:  # type: ignore[type-arg]
    path = tmp_path_factory.mktemp("sfx") / "effects.wav"
    write_wav(path, make_mix(6.0).stems["effects"])
    events, rate = detect_events(path)
    assert rate == RATE
    return events


def test_every_burst_is_detected_near_its_onset(effects_events: list) -> None:  # type: ignore[type-arg]
    times = [event.time for event in effects_events]
    for expected in BURST_TIMES:
        assert any(abs(time - expected) <= TOLERANCE_SECONDS for time in times), (expected, times)
    assert len(effects_events) == len(BURST_TIMES)


def test_event_fields(effects_events: list) -> None:  # type: ignore[type-arg]
    for event in effects_events:
        assert 0.005 <= event.duration <= BURST_SECONDS * 2
        # The left channel peaks at BURST_AMPLITUDE and the right at 0.7x; detection uses the mean.
        assert event.peak_dbfs <= 20 * np.log10(BURST_AMPLITUDE) + 0.1
        assert event.peak_dbfs > 20 * np.log10(BURST_AMPLITUDE * 0.5)
        assert 0 < event.confidence <= 1
    assert max(event.confidence for event in effects_events) == 1.0


def test_silence_has_no_events(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "silence.wav", np.zeros((2, RATE * 2), dtype=np.float32))
    assert detect_events(path)[0] == []


def test_quiet_noise_floor_is_ignored(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    hiss = (1e-4 * rng.standard_normal((2, RATE * 2))).astype(np.float32)
    assert detect_events(write_wav(tmp_path / "hiss.wav", hiss))[0] == []


def test_events_document_is_json(effects_events: list) -> None:  # type: ignore[type-arg]
    document = events_document(effects_events, RATE, "effects.wav")
    parsed = json.loads(json.dumps(document))
    assert parsed["source"] == "effects.wav"
    assert set(parsed["events"][0]) == {"time", "duration", "peak_dbfs", "confidence"}


def test_events_across_block_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from layerlab.audio import sfx

    # Force several small blocks so bursts straddle block edges.
    monkeypatch.setattr(sfx, "BLOCK_SECONDS", 0.75)
    path = write_wav(tmp_path / "effects.wav", make_mix(6.0).stems["effects"])
    times = [event.time for event in detect_events(path)[0]]
    assert len(times) == len(BURST_TIMES)
    for expected in BURST_TIMES:
        assert any(abs(time - expected) <= TOLERANCE_SECONDS for time in times)
