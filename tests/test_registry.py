"""The model registry file and its validation."""

import re
from pathlib import Path

import pytest

from layerlab.errors import LayerlabError
from layerlab.models import load_registry
from layerlab.models.registry import REGISTRY_ENV, parse_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = {
    ".safetensors",
    ".onnx",
    ".gguf",
    ".ggml",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".th",
    ".tflite",
    ".h5",
}
_SKIP_DIRS = {".git", ".venv", "venv", ".mypy_cache", ".ruff_cache", ".pytest_cache"}

VALID = """
[[models]]
name = "tiny"
task = "dialogue"
description = "A test model"
url = "https://example.com/weights/tiny.onnx"
sha256 = "{sha}"
size = 10
license = "MIT"
commercial_use = true
homepage = "https://example.com"
""".replace("{sha}", "0" * 64)


def test_bundled_registry_is_valid() -> None:
    registry = load_registry()
    assert len(registry) >= 1
    for spec in registry:
        assert spec.url.startswith("https://"), spec.name
        assert spec.license, spec.name


def test_bundled_huggingface_urls_are_pinned_to_a_commit() -> None:
    for spec in load_registry():
        if "huggingface.co" in spec.url:
            assert re.search(r"/resolve/[0-9a-f]{40}/", spec.url), spec.name


def test_registry_lists_non_commercial_models_honestly() -> None:
    for spec in load_registry():
        if spec.license in {"NOASSERTION", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"}:
            assert not spec.commercial_use, spec.name


def test_env_var_overrides_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "registry.toml"
    path.write_text(VALID, encoding="utf-8")
    monkeypatch.setenv(REGISTRY_ENV, str(path))
    assert load_registry().names == ["tiny"]


def test_filename_comes_from_url() -> None:
    assert parse_registry(VALID).get("tiny").filename == "tiny.onnx"


def test_unknown_model_lists_known_ones() -> None:
    with pytest.raises(LayerlabError, match="Known models: tiny"):
        parse_registry(VALID).get("huge")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('sha256 = "' + "0" * 64, 'sha256 = "abc', "sha256 must be 64"),
        ("size = 10", "size = 0", "size must be a positive integer"),
        ("size = 10", 'size = "10"', "size must be a positive integer"),
        ("commercial_use = true", 'commercial_use = "yes"', "commercial_use must be true"),
        ('task = "dialogue"', 'task = "cooking"', "task must be one of"),
        ('name = "tiny"', 'name = "Tiny Model"', "name 'Tiny Model'"),
        ("https://example.com/weights/tiny.onnx", "ftp://example.com/tiny.onnx", "url must be"),
        ('license = "MIT"\n', "", "missing license"),
    ],
)
def test_invalid_entries_are_rejected(old: str, new: str, message: str) -> None:
    assert old in VALID
    with pytest.raises(LayerlabError, match=message):
        parse_registry(VALID.replace(old, new))


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(LayerlabError, match="duplicate model name 'tiny'"):
        parse_registry(VALID + VALID)


def test_invalid_toml_is_reported() -> None:
    with pytest.raises(LayerlabError, match="not valid TOML"):
        parse_registry("[[models]\n")


def test_no_model_weights_in_the_repository() -> None:
    found = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.rglob("*")
        if path.suffix.lower() in WEIGHT_SUFFIXES
        and path.is_file()
        and not _SKIP_DIRS.intersection(path.relative_to(REPO_ROOT).parts)
    ]
    assert found == [], "model weights must be downloaded at runtime, not committed"
