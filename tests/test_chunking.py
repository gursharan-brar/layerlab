"""Chunked separation with crossfades, checked against known ground-truth stems."""

from pathlib import Path

import numpy as np
import pytest

from layerlab.audio.chunking import separate_file
from synthetic_audio import (
    RATE,
    BandSplitSeparator,
    IdentitySeparator,
    make_mix,
    read_wav,
    snr_db,
    write_wav,
)


@pytest.mark.parametrize("seconds", [0.3, 1.0, 1.8, 2.6, 4.123])
def test_identity_separator_reconstructs_exactly(tmp_path: Path, seconds: float) -> None:
    # Lengths shorter than one chunk, exactly one hop, and ragged multi-chunk tails.
    audio = make_mix(seconds).mix
    source = write_wav(tmp_path / "mix.wav", audio)
    stats = separate_file(
        source,
        {"dialogue": tmp_path / "out.wav"},
        IdentitySeparator(),
        chunk_seconds=1.0,
        overlap_seconds=0.2,
    )
    out, rate = read_wav(tmp_path / "out.wav")
    assert rate == RATE
    assert out.shape == audio.shape
    assert np.max(np.abs(out - audio)) < 1e-6
    assert stats.frames == audio.shape[1]
    assert stats.residual_db is not None and stats.residual_db < -100


def test_band_split_recovers_known_stems(tmp_path: Path) -> None:
    mix = make_mix(6.0)
    source = write_wav(tmp_path / "mix.wav", mix.mix)
    separator = BandSplitSeparator()
    destinations = {stem: tmp_path / f"{stem}.wav" for stem in separator.stems}
    progress: list[float] = []
    stats = separate_file(
        source,
        destinations,
        separator,
        chunk_seconds=1.5,
        overlap_seconds=0.25,
        on_progress=progress.append,
    )
    # 6 s in 1.5 s chunks with a 0.25 s overlap: hop 1.25 s, so 5 chunks.
    assert stats.chunks == separator.calls == 5
    assert progress == sorted(progress) and progress[-1] == 1.0
    for stem, path in destinations.items():
        estimate, _ = read_wav(path)
        assert snr_db(mix.stems[stem], estimate) > 20, stem
    assert stats.residual_db is not None and stats.residual_db < -100


def test_gain_error_shows_up_in_the_residual(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "mix.wav", make_mix(3.0).mix)
    separator = BandSplitSeparator(gain=0.9)
    stats = separate_file(
        source,
        {stem: tmp_path / f"{stem}.wav" for stem in separator.stems},
        separator,
        chunk_seconds=1.5,
        overlap_seconds=0.25,
    )
    # Stems at 90% of the mix leave a residual of 10%: 20 * log10(0.1) = -20 dB.
    assert stats.residual_db == pytest.approx(-20.0, abs=0.01)
    assert stats.max_abs_residual > 0


def test_silence_has_no_residual_ratio(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "mix.wav", np.zeros((2, RATE), dtype=np.float32))
    stats = separate_file(
        source,
        {"dialogue": tmp_path / "out.wav"},
        IdentitySeparator(),
        chunk_seconds=1.0,
        overlap_seconds=0.2,
    )
    assert stats.residual_db is None


def test_destination_the_backend_cannot_produce_is_rejected(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "mix.wav", make_mix(1.0).mix)
    with pytest.raises(ValueError, match="effects"):
        separate_file(
            source,
            {"effects": tmp_path / "fx.wav"},
            IdentitySeparator(),
            chunk_seconds=1.0,
            overlap_seconds=0.2,
        )
