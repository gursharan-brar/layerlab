"""Demucs htdemucs_ft separation (optional backend: `pip install "layerlab[demucs]"`).

Model: Hybrid Transformer Demucs, fine-tuned bag of four models, one per source
(Rouard, Massa, Defossez, ICASSP 2023). Code and weights: MIT,
facebookresearch/demucs.

Demucs separates music into drums, bass, other and vocals. layerlab maps vocals
to `dialogue` and the sum of the rest to `music`. It has no effects stem, so
`effects` is reported unavailable.

The checkpoints are PyTorch pickles. They are loaded only after layerlab's
model manager has verified their SHA-256 against the registry, which pins the
files Meta publishes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from layerlab.audio.dsp import FloatArray, fit_length, resample
from layerlab.backends.separation import StatusCallback
from layerlab.errors import LayerlabError
from layerlab.models import get_model, is_cached, load_registry

# One model per source, in htdemucs source order (drums, bass, other, vocals).
SOURCE_MODELS: dict[str, str] = {
    "drums": "htdemucs-ft-drums",
    "bass": "htdemucs-ft-bass",
    "other": "htdemucs-ft-other",
    "vocals": "htdemucs-ft-vocals",
}
MODELS: tuple[str, ...] = tuple(SOURCE_MODELS.values())

MODEL_RATE = 44_100
CHUNK_SECONDS = 30.0
OVERLAP_SECONDS = 2.0
SETTINGS: dict[str, Any] = {
    "chunk_seconds": CHUNK_SECONDS,
    "overlap_seconds": OVERLAP_SECONDS,
    "shifts": 1,
    "inner_overlap": 0.25,
}
_INSTALL_HINT = 'The demucs backend needs extra packages: pip install "layerlab[demucs]"'
_TORCH_DEVICES = {"cuda": "cuda", "rocm": "cuda", "mps": "mps", "cpu": "cpu"}


def map_stems(sources: dict[str, FloatArray]) -> dict[str, FloatArray]:
    """Demucs sources to layerlab stems: vocals are dialogue, everything else is music."""
    music = sources["drums"] + sources["bass"] + sources["other"]
    return {"dialogue": sources["vocals"], "music": music.astype(np.float32)}


class HTDemucsFT:
    name = "demucs"
    stems: tuple[str, ...] = ("dialogue", "music")
    unavailable: dict[str, str] = {  # noqa: RUF012 - read-only class constant
        "effects": "Demucs has no sound-effects stem; effects are mixed into music"
    }
    chunk_seconds = CHUNK_SECONDS
    overlap_seconds = OVERLAP_SECONDS

    def __init__(self, model: Any, device: str, torch: Any, apply_model: Any) -> None:
        self._model = model
        self._device = device
        self._torch = torch
        self._apply_model = apply_model

    @classmethod
    def load(cls, device: str, on_status: StatusCallback) -> HTDemucsFT:
        try:
            import torch
            from demucs.apply import BagOfModels, apply_model
            from demucs.states import load_model
        except ImportError as exc:
            raise LayerlabError(f"{_INSTALL_HINT} ({exc})") from exc

        registry = load_registry()
        models = []
        for name in MODELS:
            if not is_cached(registry.get(name)):
                on_status(f"downloading {name}")
            path = get_model(name)
            # Safe to unpickle: get_model verified the file's SHA-256 against the registry.
            package = torch.load(str(path), map_location="cpu", weights_only=False)
            models.append(load_model(package))
        weights = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        bag = BagOfModels(models, weights=weights)
        if list(bag.sources) != list(SOURCE_MODELS):
            raise LayerlabError(f"unexpected Demucs sources {bag.sources}")
        torch_device = _TORCH_DEVICES.get(device, "cpu")
        if torch_device == "cuda" and not torch.cuda.is_available():
            torch_device = "cpu"
        if torch_device == "mps" and not torch.backends.mps.is_available():
            torch_device = "cpu"
        return cls(bag, torch_device, torch, apply_model)

    def describe(self) -> dict[str, Any]:
        registry = load_registry()
        return {
            "model": "htdemucs_ft",
            "model_sha256": {name: registry.get(name).sha256 for name in MODELS},
            "runtime": f"torch {self._torch.__version__}",
            "provider": self._device,
            "stem_mapping": {"dialogue": "vocals", "music": "drums + bass + other"},
        }

    def separate(self, chunk: FloatArray, sample_rate: int) -> dict[str, FloatArray]:
        channels, length = chunk.shape
        audio = resample(chunk, sample_rate, MODEL_RATE)
        if channels == 1:
            audio = np.repeat(audio, 2, axis=0)
        elif channels > 2:
            audio = audio[:2]
        torch = self._torch
        mix = torch.from_numpy(np.ascontiguousarray(audio))
        reference = mix.mean(0)
        mean, std = reference.mean(), reference.std()
        scale = std if float(std) > 1e-8 else torch.tensor(1.0)
        with torch.no_grad():
            separated = self._apply_model(
                self._model,
                ((mix - mean) / scale)[None],
                shifts=SETTINGS["shifts"],
                split=True,
                overlap=SETTINGS["inner_overlap"],
                device=self._device,
            )[0]
        separated = separated * scale + mean / len(SOURCE_MODELS)
        sources: dict[str, FloatArray] = {}
        for index, source in enumerate(SOURCE_MODELS):
            stem = separated[index].cpu().numpy().astype(np.float32)
            if channels == 1:
                stem = stem.mean(axis=0, keepdims=True)
            back = resample(stem, MODEL_RATE, sample_rate)
            out = np.zeros((channels, length), dtype=np.float32)
            out[: back.shape[0]] = fit_length(back, length)
            sources[source] = out
        return map_stems(sources)
