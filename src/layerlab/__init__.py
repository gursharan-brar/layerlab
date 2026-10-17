"""layerlab: split a video into dialogue, music, SFX, on-screen graphics and an edit timeline."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    __version__ = _package_version("layerlab")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
