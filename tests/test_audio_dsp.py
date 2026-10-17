"""STFT and resampling helpers."""

import importlib.util

import numpy as np
import pytest

from layerlab.audio.dsp import fit_length, istft, resample, stft


def _signal(n: int = 20_000) -> np.ndarray:
    rng = np.random.default_rng(1)
    return (0.3 * rng.standard_normal(n)).astype(np.float32)


def test_stft_shape_matches_tiger_graph() -> None:
    spectrum = stft(np.zeros(529_200, dtype=np.float32), 2048, 512)
    assert spectrum.shape == (1025, 1034)


def test_istft_inverts_stft() -> None:
    x = _signal()
    y = istft(stft(x, 2048, 512), 2048, 512, len(x))
    assert np.max(np.abs(x - y)) < 1e-5


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_stft_matches_torch() -> None:
    import torch

    x = _signal()
    expected = torch.stft(
        torch.from_numpy(x),
        2048,
        512,
        window=torch.hann_window(2048),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    ).numpy()
    assert np.allclose(stft(x, 2048, 512), expected, atol=1e-4)


def test_resample_round_trip_preserves_band_limited_signal() -> None:
    t = np.arange(48_000) / 48_000
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)[None]
    there = resample(x, 48_000, 44_100)
    assert there.shape == (1, 44_100)
    back = resample(there, 44_100, 48_000)
    middle = slice(1000, -1000)
    # The polyphase filter's passband ripple is about 0.1%.
    assert np.max(np.abs(back[0, middle] - x[0, middle])) < 2e-3


def test_fit_length() -> None:
    x = np.ones((2, 5), dtype=np.float32)
    assert fit_length(x, 3).shape == (2, 3)
    padded = fit_length(x, 8)
    assert padded.shape == (2, 8)
    assert float(padded[:, 5:].sum()) == 0
