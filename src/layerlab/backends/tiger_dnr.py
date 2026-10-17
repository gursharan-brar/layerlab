"""TIGER-DnR dialogue/music/effects separation with onnxruntime (default backend).

Model: TIGER for Divide and Remaster (Xu, Li, Chen, Hu, ICLR 2025), weights from
JusperLee/TIGER-DnR, as the int8-weight ONNX export audiojs/tiger-dnr
(`tiger.int8.onnx`). Licence: Apache-2.0 as stated on both model cards.

The graph takes one mono 12 s segment at 44.1 kHz as an STFT (n_fft 2048, hop
512, periodic Hann, centred, reflect-padded) shaped [1, 2, 1025, 1034] (real,
imaginary) and returns [1, 6, 1025, 1034]: dialogue, music, effects, each real
and imaginary.

To keep CPU cost to one model run per chunk, the model runs on the mid channel
(L + R) / 2. Its three stem spectrograms become soft masks (each stem's power
over the total), which are applied to the STFT of every input channel. The
stems therefore stay stereo and add back up to the mix up to resampling error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from layerlab.audio.dsp import FloatArray, fit_length, istft, resample, stft
from layerlab.backends.separation import STEM_NAMES, StatusCallback
from layerlab.errors import LayerlabError
from layerlab.models import get_model, is_cached, load_registry

MODEL_NAME = "tiger-dnr-int8"
MODELS: tuple[str, ...] = (MODEL_NAME,)

MODEL_RATE = 44_100
SEGMENT_SAMPLES = 529_200  # 12 s at 44.1 kHz, fixed by the graph
N_FFT = 2048
HOP = 512
CHUNK_SECONDS = SEGMENT_SAMPLES / MODEL_RATE
OVERLAP_SECONDS = 2.0
SETTINGS: dict[str, Any] = {
    "chunk_seconds": CHUNK_SECONDS,
    "overlap_seconds": OVERLAP_SECONDS,
    "stereo": "mid-channel inference, masks applied per channel",
}

_EPS = 1e-10
_PROVIDERS = {
    "cuda": ["CUDAExecutionProvider"],
    "rocm": ["ROCMExecutionProvider", "MIGraphXExecutionProvider"],
    "mps": ["CoreMLExecutionProvider"],
    "cpu": [],
}


class TigerDnR:
    name = "tiger"
    stems: tuple[str, ...] = STEM_NAMES
    unavailable: dict[str, str] = {}  # noqa: RUF012 - read-only class constant
    chunk_seconds = CHUNK_SECONDS
    overlap_seconds = OVERLAP_SECONDS

    def __init__(self, session: Any, model_sha256: str) -> None:
        self._session = session
        self._input = session.get_inputs()[0].name
        self._sha256 = model_sha256

    @classmethod
    def load(cls, device: str, on_status: StatusCallback) -> TigerDnR:
        import onnxruntime as ort

        spec = load_registry().get(MODEL_NAME)
        if not is_cached(spec):
            on_status(f"downloading {MODEL_NAME}")
        path = get_model(MODEL_NAME)
        available = set(ort.get_available_providers())
        providers = [p for p in _PROVIDERS.get(device, []) if p in available]
        providers.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        try:
            session = ort.InferenceSession(str(path), options, providers=providers)
        except Exception as exc:
            raise LayerlabError(f"could not load {path} with onnxruntime: {exc}") from exc
        return cls(session, spec.sha256)

    def describe(self) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "model_sha256": self._sha256,
            "runtime": "onnxruntime",
            "provider": self._session.get_providers()[0],
        }

    def separate(self, chunk: FloatArray, sample_rate: int) -> dict[str, FloatArray]:
        channels, length = chunk.shape
        resampled = resample(chunk, sample_rate, MODEL_RATE)
        model_length = resampled.shape[1]
        if model_length > SEGMENT_SAMPLES:
            raise LayerlabError(
                f"TIGER-DnR takes at most {CHUNK_SECONDS} s per chunk; got {model_length} samples"
            )
        resampled = fit_length(resampled, SEGMENT_SAMPLES)

        masks = self._masks(resampled.mean(axis=0))
        stems: dict[str, FloatArray] = {
            stem: np.zeros((channels, length), dtype=np.float32) for stem in STEM_NAMES
        }
        for channel in range(channels):
            spectrum = stft(resampled[channel], N_FFT, HOP)
            for index, stem in enumerate(STEM_NAMES):
                audio = istft(masks[index] * spectrum, N_FFT, HOP, SEGMENT_SAMPLES)
                back = resample(audio[:model_length], MODEL_RATE, sample_rate)
                stems[stem][channel] = fit_length(back, length)
        return stems

    def _masks(self, mid: FloatArray) -> np.ndarray[Any, np.dtype[np.complex64]]:
        """Soft masks, shape (3, 1025, 1034), that sum to one in every bin."""
        spectrum = stft(mid, N_FFT, HOP)
        model_input = np.stack([spectrum.real, spectrum.imag])[None].astype(np.float32)
        output = self._session.run(None, {self._input: model_input})[0][0]
        power = np.stack(
            [output[2 * index] ** 2 + output[2 * index + 1] ** 2 for index in range(3)]
        ).astype(np.float64)
        total = power.sum(axis=0, keepdims=True)
        masks = np.where(total > _EPS, power / np.maximum(total, _EPS), 1.0 / 3.0)
        return masks.astype(np.complex64)
