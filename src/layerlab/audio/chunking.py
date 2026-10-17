"""Stream a WAV file through a separator in overlapping, crossfaded chunks.

Only one chunk (plus the previous chunk's overlap tail) is held in memory at a
time, so memory use is set by the chunk length, not the file length. Adjacent
chunks overlap by `overlap_seconds`; in the overlap the previous chunk fades
out with cos^2 while the next fades in with sin^2, so the weights always sum
to one.

While streaming, the residual between the mix and the sum of the stems is
accumulated so the caller can check that the stems add back up to the mix.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


class ChunkSeparator(Protocol):
    """Anything that splits a (channels, samples) chunk into named stems."""

    stems: tuple[str, ...]

    def separate(self, chunk: FloatArray, sample_rate: int) -> dict[str, FloatArray]:
        """Return each stem with the same shape as `chunk`."""
        ...


@dataclass(frozen=True)
class SeparationStats:
    """How well the written stems add back up to the mix."""

    sample_rate: int
    channels: int
    frames: int
    chunks: int
    mix_energy: float
    residual_energy: float
    max_abs_residual: float

    @property
    def residual_db(self) -> float | None:
        """Residual energy relative to the mix, in dB. None for a silent mix."""
        if self.mix_energy <= 0:
            return None
        if self.residual_energy <= 0:
            return -math.inf
        return 10 * math.log10(self.residual_energy / self.mix_energy)


def separate_file(
    source: Path,
    destinations: Mapping[str, Path],
    separator: ChunkSeparator,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
    on_progress: Callable[[float], None] | None = None,
) -> SeparationStats:
    """Separate `source` into one 32-bit float WAV per stem in `destinations`."""
    missing = set(destinations) - set(separator.stems)
    if missing:
        raise ValueError(f"separator does not produce: {', '.join(sorted(missing))}")
    with sf.SoundFile(str(source)) as reader:
        rate, channels, total = reader.samplerate, reader.channels, reader.frames
        chunk = max(1, round(chunk_seconds * rate))
        overlap = min(round(overlap_seconds * rate), chunk // 2)
        hop = chunk - overlap
        fade_in = (np.sin(0.5 * np.pi * (np.arange(overlap) + 0.5) / max(overlap, 1)) ** 2).astype(
            np.float32
        )
        fade_out = 1.0 - fade_in

        writers = {
            stem: sf.SoundFile(str(path), "w", rate, channels, subtype="FLOAT", format="WAV")
            for stem, path in destinations.items()
        }
        mix_energy = residual_energy = max_residual = 0.0
        chunks = 0
        try:
            tail: dict[str, FloatArray] | None = None
            position = 0
            while position < total:
                reader.seek(position)
                block: FloatArray = reader.read(chunk, dtype="float32", always_2d=True).T
                length = block.shape[1]
                last = position + chunk >= total
                stems = separator.separate(block, rate)
                chunks += 1

                if tail is not None:
                    blend = min(overlap, length)
                    for stem in separator.stems:
                        stems[stem][:, :blend] = (
                            tail[stem][:, :blend] * fade_out[:blend]
                            + stems[stem][:, :blend] * fade_in[:blend]
                        )

                emit = length if last else hop
                total_stems = np.zeros((channels, emit), dtype=np.float64)
                for stem in separator.stems:
                    total_stems += stems[stem][:, :emit]
                for stem, writer in writers.items():
                    writer.write(stems[stem][:, :emit].T)

                residual = block[:, :emit].astype(np.float64) - total_stems
                mix_energy += float(np.sum(block[:, :emit].astype(np.float64) ** 2))
                residual_energy += float(np.sum(residual**2))
                if residual.size:
                    max_residual = max(max_residual, float(np.max(np.abs(residual))))

                tail = None if last else {s: stems[s][:, hop:length] for s in separator.stems}
                if on_progress is not None:
                    on_progress(min(position + emit, total) / total)
                if last:
                    break
                position += hop
        finally:
            for writer in writers.values():
                writer.close()

    return SeparationStats(
        sample_rate=rate,
        channels=channels,
        frames=total,
        chunks=chunks,
        mix_energy=mix_energy,
        residual_energy=residual_energy,
        max_abs_residual=max_residual,
    )
