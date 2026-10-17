"""The separation backends the audio step can use, selected with --audio-backend.

Backends are imported only when loaded, so heavy dependencies (onnxruntime,
torch) are not imported by `layerlab --version` or by unrelated steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from layerlab.backends import htdemucs, tiger_dnr
from layerlab.backends.separation import Separator, StatusCallback
from layerlab.errors import LayerlabError
from layerlab.models import load_registry


@dataclass(frozen=True)
class BackendInfo:
    """What the audio step needs to know about a backend before loading it."""

    name: str
    models: tuple[str, ...]
    load: Callable[[str, StatusCallback], Separator]
    settings: dict[str, Any] = field(default_factory=dict)

    def config(self) -> dict[str, Any]:
        """Settings that change the output, used in the audio step's input hash."""
        registry = load_registry()
        return {
            "backend": self.name,
            "models": {name: registry.get(name).sha256 for name in self.models},
            **self.settings,
        }


BACKENDS: dict[str, BackendInfo] = {
    "tiger": BackendInfo("tiger", tiger_dnr.MODELS, tiger_dnr.TigerDnR.load, tiger_dnr.SETTINGS),
    "demucs": BackendInfo("demucs", htdemucs.MODELS, htdemucs.HTDemucsFT.load, htdemucs.SETTINGS),
}
BACKEND_NAMES: tuple[str, ...] = tuple(BACKENDS)
DEFAULT_BACKEND = "tiger"


def get_backend(name: str) -> BackendInfo:
    try:
        return BACKENDS[name]
    except KeyError:
        raise LayerlabError(
            f"unknown audio backend '{name}'. Choose from: {', '.join(BACKEND_NAMES)}."
        ) from None
