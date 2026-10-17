"""Manifest schema loading and migration."""

import json
from pathlib import Path

import pytest

from layerlab.errors import LayerlabError
from layerlab.manifest import SCHEMA_VERSION, DeviceRecord, Manifest, SourceInfo

V1_MANIFEST = {
    "source": {"path": "/videos/clip.mp4", "sha256": "ab" * 32, "size_bytes": 10},
    "steps": {"extract": {"status": "succeeded", "version": 1, "input_hash": "x"}},
    "schema_version": 1,
    "layerlab_version": "0.1.0",
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}


def test_round_trip(tmp_path: Path) -> None:
    manifest = Manifest(
        source=SourceInfo("/videos/clip.mp4", "ab" * 32, 10),
        device=DeviceRecord("cuda", "RTX", "auto"),
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = Manifest.load(path)
    assert loaded.device == DeviceRecord("cuda", "RTX", "auto")
    assert loaded.schema_version == SCHEMA_VERSION


def test_version_1_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(V1_MANIFEST), encoding="utf-8")
    manifest = Manifest.load(path)
    assert manifest.device is None
    assert manifest.steps["extract"].status == "succeeded"
    manifest.save(path)
    assert json.loads(path.read_text("utf-8"))["schema_version"] == SCHEMA_VERSION


def test_future_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**V1_MANIFEST, "schema_version": 99}), encoding="utf-8")
    with pytest.raises(LayerlabError, match="schema version 99"):
        Manifest.load(path)


def test_corrupt_manifest_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LayerlabError, match="Delete the job folder"):
        Manifest.load(path)
