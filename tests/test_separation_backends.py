"""The TIGER-DnR and Demucs backends.

Most tests use a fake onnxruntime session, so they need no weights. Tests
marked `real_models` download and run the real models; they only run with
LAYERLAB_REAL_MODEL_TESTS=1.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rich.console import Console

from layerlab.backends import htdemucs, tiger_dnr
from layerlab.backends.catalog import BACKENDS
from layerlab.errors import LayerlabError
from layerlab.job import Job
from layerlab.pipeline import run_pipeline
from layerlab.step import StepResult, StepSpec
from layerlab.steps import audio
from synthetic_audio import make_mix, write_wav


class _FakeInput:
    name = "mix_spec"


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: returns fixed-ratio 'stems'."""

    def __init__(self, ratios: tuple[float, float, float] = (0.5, 0.3, 0.2)) -> None:
        self.ratios = ratios
        self.calls = 0

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, _outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        spec = feeds["mix_spec"]
        assert spec.shape == (1, 2, 1025, 1034)
        assert spec.dtype == np.float32
        return [np.concatenate([spec * ratio for ratio in self.ratios], axis=1)]


def test_tiger_stems_are_stereo_and_sum_to_the_chunk() -> None:
    separator = tiger_dnr.TigerDnR(FakeSession(), "0" * 64)
    chunk = make_mix(3.0).mix
    stems = separator.separate(chunk, 48_000)
    assert set(stems) == {"dialogue", "music", "effects"}
    for stem in stems.values():
        assert stem.shape == chunk.shape
        assert stem.dtype == np.float32
    total = stems["dialogue"] + stems["music"] + stems["effects"]
    residual = np.sum((total - chunk) ** 2) / np.sum(chunk**2)
    assert residual < 1e-4  # resampling 48 kHz -> 44.1 kHz -> 48 kHz only


def test_tiger_masks_follow_the_model_power() -> None:
    # Output stems at 0.5/0.3/0.2 of the mix: power masks are 25:9:4.
    separator = tiger_dnr.TigerDnR(FakeSession(), "0" * 64)
    stems = separator.separate(make_mix(2.0).mix, 48_000)
    energy = {name: float(np.sum(stem**2)) for name, stem in stems.items()}
    total = sum(np.sqrt(value) for value in energy.values())
    shares = {name: np.sqrt(value) / total for name, value in energy.items()}
    assert shares["dialogue"] == pytest.approx(25 / 38, abs=0.01)
    assert shares["music"] == pytest.approx(9 / 38, abs=0.01)
    assert shares["effects"] == pytest.approx(4 / 38, abs=0.01)


def test_tiger_runs_the_model_once_per_chunk_whatever_the_channel_count() -> None:
    session = FakeSession()
    tiger_dnr.TigerDnR(session, "0" * 64).separate(make_mix(1.0).mix, 48_000)
    assert session.calls == 1


def test_tiger_rejects_chunks_longer_than_the_graph() -> None:
    separator = tiger_dnr.TigerDnR(FakeSession(), "0" * 64)
    with pytest.raises(LayerlabError, match="at most"):
        separator.separate(np.zeros((2, 48_000 * 13), dtype=np.float32), 48_000)


def test_demucs_stem_mapping() -> None:
    sources = {
        name: np.full((2, 4), value, dtype=np.float32)
        for name, value in {"drums": 1.0, "bass": 2.0, "other": 3.0, "vocals": 4.0}.items()
    }
    stems = htdemucs.map_stems(sources)
    assert set(stems) == {"dialogue", "music"}
    assert np.all(stems["dialogue"] == 4.0)
    assert np.all(stems["music"] == 6.0)


def test_demucs_reports_effects_unavailable() -> None:
    assert "effects" not in htdemucs.HTDemucsFT.stems
    assert "effects" in htdemucs.HTDemucsFT.unavailable


def test_demucs_without_extra_explains_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_demucs(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("demucs") or name == "torch":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_demucs)
    with pytest.raises(LayerlabError, match=r"layerlab\[demucs\]"):
        htdemucs.HTDemucsFT.load("cpu", lambda _text: None)


def test_backend_models_are_in_the_registry() -> None:
    # conftest swaps BACKENDS for fakes; check the real backend modules instead.
    from layerlab.models import load_registry

    registry = load_registry()
    for name in (*tiger_dnr.MODELS, *htdemucs.MODELS):
        assert registry.get(name).task == "separation"
    assert set(BACKENDS) == {"tiger", "demucs"}


def test_tests_never_use_real_backends() -> None:
    assert BACKENDS["tiger"].settings == {"fake": 1}
    assert BACKENDS["demucs"].settings == {"fake": 1}


# Real models: opt-in, they download weights into the user cache.

MIX = make_mix(4.0)


def _fake_extract(job: Job) -> StepResult:
    write_wav(job.work_dir / "audio.wav", MIX.mix)
    return StepResult(outputs={"audio": "audio.wav"})


def _run_real(backend: str, tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"stand-in")
    steps = (
        StepSpec("extract", 1, lambda job: [job.source_path], _fake_extract),
        StepSpec(audio.NAME, audio.VERSION, audio.inputs, audio.run, audio.config),
    )
    report = run_pipeline(
        source,
        tmp_path / "out",
        steps=steps,
        device="cpu",
        options={audio.OPTION: backend},
        console=Console(quiet=True),
    )
    manifest: dict[str, Any] = json.loads((report.job_root / "manifest.json").read_text("utf-8"))
    record: dict[str, Any] = manifest["steps"]["audio"]
    return record


@pytest.mark.real_models
def test_real_tiger(tmp_path: Path) -> None:
    record = _run_real("tiger", tmp_path)
    assert record["status"] == "succeeded"
    assert record["data"]["model"] == "tiger-dnr-int8"
    assert record["data"]["mix_check"]["within_tolerance"] is True
    assert record["data"]["mix_check"]["residual_db"] < -40


@pytest.mark.real_models
@pytest.mark.skipif(importlib.util.find_spec("demucs") is None, reason="layerlab[demucs] missing")
def test_real_demucs(tmp_path: Path) -> None:
    record = _run_real("demucs", tmp_path)
    assert record["status"] == "succeeded"
    assert record["data"]["stems"]["effects"]["available"] is False
    assert record["data"]["mix_check"]["residual_db"] is not None
