"""STFT and resampling helpers in NumPy/SciPy.

`stft`/`istft` match `torch.stft`/`torch.istft` with a periodic Hann window,
`center=True`, reflect padding and no normalisation, which is what the
TIGER-DnR ONNX graph was exported against.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
ComplexArray = NDArray[np.complex64]


@lru_cache(maxsize=4)
def hann(n_fft: int) -> NDArray[np.float64]:
    """Periodic Hann window, as `torch.hann_window` returns by default."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)


def stft(x: FloatArray, n_fft: int, hop: int) -> ComplexArray:
    """Complex spectrogram, shape (n_fft // 2 + 1, frames)."""
    pad = n_fft // 2
    padded = np.pad(x.astype(np.float64), pad, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    return np.asarray(np.fft.rfft(frames * hann(n_fft), axis=1).T, dtype=np.complex64)


def istft(spec: ComplexArray, n_fft: int, hop: int, length: int) -> FloatArray:
    """Inverse of `stft`, trimmed to `length` samples."""
    window = hann(n_fft)
    frames = np.fft.irfft(spec.T.astype(np.complex128), n=n_fft, axis=1) * window
    count = frames.shape[0]
    total = n_fft + hop * (count - 1)
    signal = np.zeros(total)
    norm = np.zeros(total)
    squared = window**2
    for index in range(count):
        start = index * hop
        signal[start : start + n_fft] += frames[index]
        norm[start : start + n_fft] += squared
    nonzero = norm > 1e-11
    signal[nonzero] /= norm[nonzero]
    pad = n_fft // 2
    out = signal[pad : pad + length]
    if out.shape[0] < length:
        out = np.pad(out, (0, length - out.shape[0]))
    return out.astype(np.float32)


def resample(x: FloatArray, source_rate: int, target_rate: int) -> FloatArray:
    """Polyphase resampling along the last axis."""
    if source_rate == target_rate:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly  # imported here to keep CLI start-up fast

    ratio = Fraction(target_rate, source_rate)
    out = resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)
    return np.asarray(out, dtype=np.float32)


def fit_length(x: FloatArray, length: int) -> FloatArray:
    """Zero-pad or trim the last axis to exactly `length` samples."""
    current = x.shape[-1]
    if current == length:
        return x
    if current > length:
        return x[..., :length]
    widths = [(0, 0)] * (x.ndim - 1) + [(0, length - current)]
    return np.pad(x, widths)
