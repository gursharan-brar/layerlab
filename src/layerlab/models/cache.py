"""Where model weights live on disk, and fetching them on first use."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir

from layerlab.models.download import DownloadProgress, download
from layerlab.models.registry import ModelSpec, load_registry

CACHE_ENV = "LAYERLAB_CACHE_DIR"


def cache_dir() -> Path:
    """Model cache folder: `$LAYERLAB_CACHE_DIR/models`, else the platform user cache."""
    override = os.environ.get(CACHE_ENV)
    root = Path(override) if override else Path(user_cache_dir("layerlab", appauthor=False))
    return root / "models"


def model_path(spec: ModelSpec, root: Path | None = None) -> Path:
    return (root or cache_dir()) / spec.name / spec.filename


def is_cached(spec: ModelSpec, root: Path | None = None) -> bool:
    """True if the file is present with the expected size. Its hash was checked on download."""
    path = model_path(spec, root)
    return path.is_file() and path.stat().st_size == spec.size_bytes


def ensure_model(
    spec: ModelSpec,
    *,
    root: Path | None = None,
    on_progress: DownloadProgress | None = None,
) -> Path:
    """Path to the model's weights, downloading and verifying them first if needed."""
    path = model_path(spec, root)
    if not is_cached(spec, root):
        download(spec.url, path, sha256=spec.sha256, size=spec.size_bytes, on_progress=on_progress)
    return path


def get_model(name: str, *, on_progress: DownloadProgress | None = None) -> Path:
    """Look up `name` in the registry and return its weights, downloading on first use."""
    return ensure_model(load_registry().get(name), on_progress=on_progress)
