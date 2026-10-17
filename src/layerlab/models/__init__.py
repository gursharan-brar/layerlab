"""Model weights: the registry, the download cache, and fetching on first use.

Weights are never bundled with layerlab. Backends call `get_model(name)`,
which downloads the file into the user cache on first use and verifies it.
"""

from layerlab.models.cache import cache_dir, ensure_model, get_model, is_cached, model_path
from layerlab.models.download import DownloadError
from layerlab.models.registry import ModelSpec, Registry, load_registry

__all__ = [
    "DownloadError",
    "ModelSpec",
    "Registry",
    "cache_dir",
    "ensure_model",
    "get_model",
    "is_cached",
    "load_registry",
    "model_path",
]
