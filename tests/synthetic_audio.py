"""Synthetic audio with known stems, and a fake separator that can recover them.

The three ground-truth stems live in separate frequency bands:

- dialogue: 220 Hz and 440 Hz tones with a 4 Hz syllable-like envelope (< 1 kHz),
- music: a steady 1.5 kHz + 2.5 kHz chord (1-4 kHz),
- effects: 6-12 kHz noise bursts with a sharp attack at known times (> 4 kHz).

`BandSplitSeparator` splits each chunk by those bands, so it recovers the stems
almost exactly and its stems sum to the mix exactly. That lets tests check the
chunking, crossfades, residual check and SFX detection against a known answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

RATE = 48_000
BURST_TIMES = (0.5, 1.7, 2.9, 4.2, 5.3)
BURST_SECONDS = 0.08
BURST_AMPLITUDE = 0.5
DIALOGUE_BAND = (0.0, 1000.0)
MUSIC_BAND = (1000.0, 4000.0)
EFFECTS_BAND = (4000.0, math.inf)


@dataclass(frozen=True)
class Mix:
    rate: int
    mix: FloatArray
    stems: dict[str, FloatArray]


def make_mix(seconds: float = 6.0, rate: int = RATE, seed: int = 0) -> Mix:
    """A stereo mix, shape (2, samples), and its three ground-truth stems."""
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    t = np.arange(n) / rate

    envelope = 0.5 * (1 - np.cos(2 * np.pi * 4 * t))
    dialogue = 0.25 * envelope * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t))
    music = 0.08 * (np.sin(2 * np.pi * 1500 * t) + np.sin(2 * np.pi * 2500 * t))
    effects = np.zeros(n)
    noise = _band_noise(n, rate, 6000.0, 12000.0, rng)
    burst = round(BURST_SECONDS * rate)
    decay = np.exp(-np.arange(burst) / (0.015 * rate))
    attack = np.minimum(np.arange(burst) / (0.001 * rate), 1.0)
    for time in BURST_TIMES:
        start = round(time * rate)
        if start >= n:
            continue
        stop = min(start + burst, n)
        effects[start:stop] += noise[start:stop] * (attack * decay)[: stop - start]
    effects *= BURST_AMPLITUDE / max(float(np.max(np.abs(effects))), 1e-9)

    stems = {
        "dialogue": np.stack([dialogue, dialogue]),
        "music": np.stack([music * 1.2, music * 0.8]),
        "effects": np.stack([effects, effects * 0.7]),
    }
    stems32 = {name: stem.astype(np.float32) for name, stem in stems.items()}
    mix = sum(stems32.values(), np.zeros((2, n), dtype=np.float32)).astype(np.float32)
    return Mix(rate, mix, stems32)


def write_wav(path: Path, audio: FloatArray, rate: int = RATE) -> Path:
    sf.write(str(path), audio.T, rate, subtype="FLOAT")
    return path


def read_wav(path: Path) -> tuple[FloatArray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T.copy(), int(rate)


def snr_db(reference: FloatArray, estimate: FloatArray) -> float:
    noise = float(np.sum((reference.astype(np.float64) - estimate) ** 2))
    signal = float(np.sum(reference.astype(np.float64) ** 2))
    return math.inf if noise == 0 else 10 * math.log10(signal / noise)


def _band_noise(
    n: int, rate: int, low: float, high: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / rate)
    spectrum[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, n)


@dataclass
class BandSplitSeparator:
    """Fake backend: splits each chunk by frequency band. Its stems sum to the input."""

    name: str = "fake-bands"
    stems: tuple[str, ...] = ("dialogue", "music", "effects")
    unavailable: dict[str, str] = field(default_factory=dict)
    chunk_seconds: float = 1.5
    overlap_seconds: float = 0.25
    gain: float = 1.0
    calls: int = 0

    def separate(self, chunk: FloatArray, sample_rate: int) -> dict[str, FloatArray]:
        self.calls += 1
        spectrum = np.fft.rfft(chunk.astype(np.float64), axis=1)
        freqs = np.fft.rfftfreq(chunk.shape[1], 1 / sample_rate)
        bands = {"dialogue": DIALOGUE_BAND, "music": MUSIC_BAND, "effects": EFFECTS_BAND}
        out: dict[str, FloatArray] = {}
        for stem in self.stems:
            low, high = bands[stem]
            mask = (freqs >= low) & (freqs < high)
            audio = np.fft.irfft(spectrum * mask, chunk.shape[1], axis=1) * self.gain
            out[stem] = audio.astype(np.float32)
        if "effects" not in self.stems and "music" in out:
            # Like Demucs: anything the backend has no stem for ends up in music.
            low = MUSIC_BAND[0]
            rest = np.fft.irfft(spectrum * (freqs >= low), chunk.shape[1], axis=1)
            out["music"] = (rest * self.gain).astype(np.float32)
        return out

    def describe(self) -> dict[str, Any]:
        return {"model": self.name}


class IdentitySeparator:
    """Fake backend that returns the input as a single stem."""

    name = "identity"
    stems: tuple[str, ...] = ("dialogue",)
    unavailable: dict[str, str] = {}  # noqa: RUF012
    chunk_seconds = 1.0
    overlap_seconds = 0.2

    def separate(self, chunk: FloatArray, sample_rate: int) -> dict[str, FloatArray]:
        return {"dialogue": chunk.copy()}

    def describe(self) -> dict[str, Any]:
        return {"model": self.name}
