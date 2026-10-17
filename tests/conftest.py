"""Shared fixtures. Test media is generated with ffmpeg; no binary media is committed."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

SYNTHETIC_DURATION_SECONDS = 2
SYNTHETIC_FPS = 25
SYNTHETIC_SIZE = (320, 240)

# Candidate fonts for drawtext, so the overlay does not depend on fontconfig.
_FONT_CANDIDATES = {
    "win32": ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"],
    "darwin": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    "linux": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ],
}


REAL_MODELS_ENV = "LAYERLAB_REAL_MODEL_TESTS"
_CACHE_ENV = "LAYERLAB_CACHE_DIR"
_USER_CACHE = os.environ.get(_CACHE_ENV)


def _real_text_detector_loader() -> Any:
    from layerlab.steps import analyze

    return analyze.load_text_detector


_REAL_LOAD_TEXT_DETECTOR = _real_text_detector_loader()


class ScriptedTextDetector:
    """Fake text detector: returns preset boxes for each call, in call order (sample index)."""

    name = "scripted"

    def __init__(self, script: dict[int, list[Any]]) -> None:
        self.script = script
        self.calls = 0

    def detect(self, frame: Any) -> list[Any]:
        boxes = self.script.get(self.calls, [])
        self.calls += 1
        return list(boxes)

    def describe(self) -> dict[str, Any]:
        return {"model": "scripted"}


def _fake_tiger(_device: str, _status: Any) -> Any:
    from synthetic_audio import BandSplitSeparator

    return BandSplitSeparator(name="fake-tiger")


def _fake_demucs(_device: str, _status: Any) -> Any:
    from synthetic_audio import BandSplitSeparator

    return BandSplitSeparator(
        name="fake-demucs",
        stems=("dialogue", "music"),
        unavailable={"effects": "fake backend has no effects stem"},
    )


@pytest.fixture(scope="session", autouse=True)
def _hermetic_models(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep every test, and every fixture of any scope, off real models.

    Session-scoped so it is in place before module-scoped fixtures run the
    pipeline. Tests get an empty temp model cache and fake separation backends
    that need no weights. Tests marked `real_models` opt out (see below).
    """
    from layerlab.backends import catalog
    from layerlab.steps import analyze

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(_CACHE_ENV, str(tmp_path_factory.mktemp("model-cache")))
        patch.setattr(analyze, "load_text_detector", lambda *_args: ScriptedTextDetector({}))
        patch.setitem(
            catalog.BACKENDS, "tiger", catalog.BackendInfo("tiger", (), _fake_tiger, {"fake": 1})
        )
        patch.setitem(
            catalog.BACKENDS,
            "demucs",
            catalog.BackendInfo("demucs", (), _fake_demucs, {"fake": 1}),
        )
        yield


@pytest.fixture(autouse=True)
def _real_models_opt_in(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """`real_models` tests use real backends and cache; they need LAYERLAB_REAL_MODEL_TESTS=1."""
    if request.node.get_closest_marker("real_models") is None:
        return
    if not os.environ.get(REAL_MODELS_ENV):
        pytest.skip(f"set {REAL_MODELS_ENV}=1 to run tests that download and run real models")
    from layerlab.backends import catalog, htdemucs, tiger_dnr

    real = {
        "tiger": catalog.BackendInfo(
            "tiger", tiger_dnr.MODELS, tiger_dnr.TigerDnR.load, tiger_dnr.SETTINGS
        ),
        "demucs": catalog.BackendInfo(
            "demucs", htdemucs.MODELS, htdemucs.HTDemucsFT.load, htdemucs.SETTINGS
        ),
    }
    for name, info in real.items():
        monkeypatch.setitem(catalog.BACKENDS, name, info)
    from layerlab.steps import analyze

    monkeypatch.setattr(analyze, "load_text_detector", _REAL_LOAD_TEXT_DETECTOR)
    if _USER_CACHE:
        monkeypatch.setenv(_CACHE_ENV, _USER_CACHE)
    else:
        monkeypatch.delenv(_CACHE_ENV, raising=False)


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        message = "ffmpeg and ffprobe must be on PATH"
        # CI sets this so a missing ffmpeg fails the run instead of silently skipping.
        if os.environ.get("LAYERLAB_REQUIRE_FFMPEG"):
            pytest.fail(message)
        pytest.skip(message)
    return ffmpeg


def _drawtext_filter() -> str:
    options = [
        "text='layerlab'",
        "fontsize=32",
        "fontcolor=white",
        "box=1",
        "boxcolor=black@0.6",
        "x=(w-text_w)/2",
        "y=h-60",
    ]
    for candidate in _FONT_CANDIDATES.get(sys.platform, []):
        if Path(candidate).is_file():
            # Colons separate filter options, so the drive letter's colon is escaped.
            escaped = Path(candidate).as_posix().replace(":", "\\:")
            options.insert(0, f"fontfile='{escaped}'")
            break
    return "drawtext=" + ":".join(options)


def _generate(path: Path, *, with_audio: bool) -> Path:
    ffmpeg = _require_ffmpeg()
    width, height = SYNTHETIC_SIZE
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"smptebars=size={width}x{height}:rate={SYNTHETIC_FPS}:duration={SYNTHETIC_DURATION_SECONDS}",
    ]
    if with_audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=44100:duration={SYNTHETIC_DURATION_SECONDS}",
        ]
    command += ["-vf", _drawtext_filter(), "-c:v", "mpeg4", "-q:v", "5"]
    command += ["-c:a", "aac", "-b:a", "96k", "-shortest"] if with_audio else ["-an"]
    command.append(str(path))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"could not generate test video: {result.stderr.strip()}")
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two seconds of colour bars with a drawtext overlay and a 440 Hz mono tone."""
    return _generate(tmp_path_factory.mktemp("media") / "synthetic.mp4", with_audio=True)


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Colour bars with a drawtext overlay and no audio stream."""
    return _generate(tmp_path_factory.mktemp("media") / "silent.mp4", with_audio=False)


def ffprobe_streams(path: Path) -> list[dict[str, Any]]:
    """Streams of a media file as reported by ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    streams: list[dict[str, Any]] = json.loads(result.stdout)["streams"]
    return streams


@pytest.fixture(scope="session")
def overlay_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Known cuts, caption and logo: see tests/synthetic_video.py."""
    from synthetic_video import make_overlay_video

    return make_overlay_video(_require_ffmpeg(), tmp_path_factory.mktemp("media") / "overlay.mp4")
