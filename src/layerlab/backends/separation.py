"""The source separation interface.

A separator splits a (channels, samples) chunk into some of the stems
`dialogue`, `music` and `effects`. Backends that cannot produce a stem leave it
out of `stems` and say why in `unavailable`. The available backends are listed
in `layerlab.backends.catalog`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

STEM_NAMES: tuple[str, ...] = ("dialogue", "music", "effects")

FloatArray = NDArray[np.float32]
StatusCallback = Callable[[str], None]


class Separator(Protocol):
    name: str
    stems: tuple[str, ...]
    unavailable: dict[str, str]
    chunk_seconds: float
    overlap_seconds: float

    def separate(self, chunk: FloatArray, sample_rate: int) -> dict[str, FloatArray]:
        """Return each stem in `stems` with the same shape as `chunk`."""
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable details for the manifest (model, runtime, provider)."""
        ...
