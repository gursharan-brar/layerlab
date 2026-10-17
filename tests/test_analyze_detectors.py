"""The analyze detectors on their own: config, text tracking, logo stability. No ffmpeg."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from layerlab.analyze.config import load_config
from layerlab.analyze.logos import StabilityAccumulator
from layerlab.analyze.text_overlays import TextTracker, iou
from layerlab.backends.text_detection import TextBox
from layerlab.errors import LayerlabError

DEFAULTS = load_config()


# Config


def test_defaults_load() -> None:
    assert DEFAULTS.sampling.fps == 2.0
    assert DEFAULTS.cuts.threshold == 27.0
    assert DEFAULTS.logos.stable_fraction == 0.8
    assert DEFAULTS.text.min_samples == 2


def test_override_changes_only_given_keys(tmp_path: Path) -> None:
    path = tmp_path / "my.toml"
    path.write_text("[cuts]\nthreshold = 32\n[logos]\nstable_fraction = 0.9\n", encoding="utf-8")
    config = load_config(path)
    assert config.cuts.threshold == 32.0  # int accepted for a float setting
    assert config.logos.stable_fraction == 0.9
    assert config.cuts.min_shot_seconds == DEFAULTS.cuts.min_shot_seconds
    assert config.text == DEFAULTS.text


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[cutz]\nthreshold = 1\n", "unknown section"),
        ("[cuts]\nthreshhold = 1\n", "unknown key"),
        ('[cuts]\nthreshold = "high"\n', "threshold must be float"),
        ("[text]\nmin_samples = 2.5\n", "min_samples must be int"),
        ("[cuts]\nluma_only = 1\n", "luma_only must be bool"),
        ("[cuts\n", "not valid TOML"),
    ],
)
def test_bad_overrides_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LayerlabError, match=message):
        load_config(path)


# Text tracking


def _box(x: int, y: int, w: int = 200, h: int = 30, score: float = 0.9) -> TextBox:
    return TextBox(x, y, w, h, score)


def test_iou() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_persistent_box_becomes_one_track() -> None:
    tracker = TextTracker(DEFAULTS.text)
    for sample in range(2, 9):
        tracker.update(sample, [_box(40 + sample % 2, 290)])  # jitters by a pixel
    (track,) = tracker.tracks()
    assert (track.first, track.last) == (2, 8)
    assert track.bbox == (40, 290, 201, 30)


def test_single_sample_flashes_are_dropped() -> None:
    tracker = TextTracker(DEFAULTS.text)
    tracker.update(0, [_box(10, 10)])
    tracker.update(5, [_box(300, 200)])
    assert tracker.tracks() == []


def test_short_gap_is_bridged_and_filled() -> None:
    tracker = TextTracker(DEFAULTS.text)  # max_gap_samples = 1
    tracker.update(0, [_box(40, 290)])
    tracker.update(2, [_box(42, 290)])
    (track,) = tracker.tracks()
    assert track.filled() == {0: (40, 290, 200, 30), 1: (40, 290, 200, 30), 2: (42, 290, 200, 30)}


def test_long_gap_splits_the_track() -> None:
    tracker = TextTracker(DEFAULTS.text)
    for sample in (0, 1, 5, 6):
        tracker.update(sample, [_box(40, 290)])
    assert [(t.first, t.last) for t in tracker.tracks()] == [(0, 1), (5, 6)]


def test_separate_regions_are_separate_tracks() -> None:
    tracker = TextTracker(DEFAULTS.text)
    for sample in range(3):
        tracker.update(sample, [_box(40, 290), _box(40, 20)])
    assert sorted(t.bbox[1] for t in tracker.tracks()) == [20, 290]


def test_small_boxes_are_ignored_and_coordinates_scaled() -> None:
    tracker = TextTracker(DEFAULTS.text, scale=(2.0, 2.0))
    for sample in range(2):
        tracker.update(sample, [_box(10, 10, 50, 4), _box(20, 100, 100, 20)])
    (track,) = tracker.tracks()  # the 4 px box is 8 px scaled, under min_box_height
    assert track.bbox == (40, 200, 200, 40)


# Logos

SIZE = (160, 90)
LOGO = (120, 8, 30, 16)  # x, y, w, h


def _logo_patch() -> np.ndarray:
    patch = np.full((LOGO[3], LOGO[2]), 240, dtype=np.uint8)
    patch[4:12, 6:24] = 20
    patch[6:10, 12:18] = 140
    return patch


def _frames(count: int, logo_samples: set[int], *, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    frames = []
    for sample in range(count):
        frame = rng.integers(0, 256, (SIZE[1], SIZE[0]), dtype=np.uint8)
        if sample in logo_samples:
            x, y, w, h = LOGO
            frame[y : y + h, x : x + w] = _logo_patch()
        frames.append(frame)
    return frames


def _regions(frames: list[np.ndarray], **overrides: Any) -> list:  # type: ignore[type-arg]
    accumulator = StabilityAccumulator(SIZE, replace(DEFAULTS.logos, **overrides))
    for frame in frames:
        accumulator.update(frame)
    return accumulator.regions()


def test_always_visible_logo_is_found() -> None:
    (region,) = _regions(_frames(20, set(range(20))))
    assert region.bbox == LOGO
    assert region.stable_fraction == 1.0
    assert region.temporal_std == pytest.approx(0.0, abs=0.01)


def test_logo_visible_in_85_percent_of_samples_is_found() -> None:
    (region,) = _regions(_frames(20, set(range(3, 20))))
    assert region.bbox == LOGO
    assert 0.8 <= region.stable_fraction < 0.9
    assert region.temporal_std > 10


def test_logo_visible_half_the_time_is_not_static() -> None:
    assert _regions(_frames(20, set(range(10)))) == []


def test_flat_letterbox_bar_is_not_a_logo() -> None:
    frames = _frames(20, set())
    for frame in frames:
        frame[80:, :] = 0  # flat black bar along the bottom, 11% of the frame
    assert _regions(frames, max_area_fraction=0.2) == []


def test_too_few_samples_finds_nothing() -> None:
    assert _regions(_frames(3, {0, 1, 2})) == []


def test_presence_tracks_when_the_logo_is_on_screen() -> None:
    frames = _frames(20, set(range(3, 20)))
    accumulator = StabilityAccumulator(SIZE, DEFAULTS.logos)
    for frame in frames:
        accumulator.update(frame)
    (region,) = accumulator.regions()
    mode = accumulator.mode
    present = [i for i, frame in enumerate(frames) if accumulator.presence(region, frame, mode)]
    assert present == list(range(3, 20))
