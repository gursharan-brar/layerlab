"""Parsing ffprobe output into the media properties stored in the manifest."""

from pathlib import Path

import pytest

from layerlab.errors import LayerlabError
from layerlab.steps import extract

VIDEO = {
    "index": 0,
    "codec_type": "video",
    "codec_name": "h264",
    "width": 1920,
    "height": 1080,
    "avg_frame_rate": "30000/1001",
    "r_frame_rate": "30000/1001",
}
AUDIO = {
    "index": 1,
    "codec_type": "audio",
    "codec_name": "aac",
    "sample_rate": "44100",
    "channels": 1,
}


def test_describe_media() -> None:
    media = extract.describe_media({"format": {"duration": "12.345"}}, VIDEO, AUDIO)
    assert media.fps == pytest.approx(29.97, abs=0.001)
    assert media.fps_rational == "30000/1001"
    assert (media.width, media.height) == (1920, 1080)
    assert media.duration_seconds == pytest.approx(12.345)
    assert media.source_audio_sample_rate == 44_100
    assert (media.audio_sample_rate, media.audio_channels) == (48_000, 2)


def test_frame_rate_falls_back_to_r_frame_rate() -> None:
    video = {**VIDEO, "avg_frame_rate": "0/0", "r_frame_rate": "25/1"}
    assert extract.describe_media({}, video, AUDIO).fps == 25


def test_missing_values_are_none() -> None:
    video = {"index": 0, "codec_type": "video"}
    media = extract.describe_media({"format": {"duration": "N/A"}}, video, {})
    assert media.fps is None
    assert media.width is None
    assert media.duration_seconds is None


def test_cover_art_is_not_treated_as_video() -> None:
    cover = {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 1}}
    with pytest.raises(LayerlabError, match="no video stream"):
        extract._select_streams({"streams": [cover, AUDIO]}, Path("song.m4a"))


def test_missing_audio_is_reported() -> None:
    with pytest.raises(LayerlabError, match="no audio stream"):
        extract._select_streams({"streams": [VIDEO]}, Path("clip.mp4"))
