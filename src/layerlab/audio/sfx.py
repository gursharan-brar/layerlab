"""Sound-effect event detection on the effects stem with librosa onset detection.

The file is read in blocks (with a little context on either side so block
edges do not create false onsets) and reduced to frame-level features: onset
strength, RMS and the sample peak of each hop. Onsets are picked over the whole
file's onset-strength envelope with `librosa.onset.onset_detect`, then each is
moved to the first hop whose peak is within 20 dB of the attack's peak, which
places it within one hop (about 11 ms at 48 kHz) of the true attack.

For each event:

- `time`: onset in seconds,
- `duration`: until RMS falls 20 dB below the event's loudest frame, or the
  next onset, whichever comes first (capped at 5 s),
- `peak_dbfs`: sample peak within the event, in dBFS,
- `confidence`: onset strength relative to the strongest onset in the file (0..1).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

HOP = 512
FRAME = 2048
BLOCK_SECONDS = 120.0
CONTEXT_FRAMES = 8
SEARCH_FRAMES = 4
ATTACK_DB = 20.0
DECAY_DB = 20.0
SILENCE_DBFS = -60.0
MAX_EVENT_SECONDS = 5.0

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SfxEvent:
    time: float
    duration: float
    peak_dbfs: float
    confidence: float


def detect_events(
    path: Path, on_progress: Callable[[float], None] | None = None
) -> tuple[list[SfxEvent], int]:
    """Detect onsets in a WAV file. Returns the events and the file's sample rate."""
    import librosa  # imported here: it is slow to import

    strength, rms, peak, rate = _frame_features(path, on_progress)
    if strength.size == 0 or float(np.max(strength)) <= 0:
        return [], rate
    picked = librosa.onset.onset_detect(
        onset_envelope=strength, sr=rate, hop_length=HOP, units="frames"
    )
    if picked.size == 0:
        return [], rate

    starts: list[int] = []
    strengths: list[float] = []
    for frame in picked.tolist():
        start = _attack_start(peak, frame, starts[-1] + 1 if starts else 0)
        if starts and start <= starts[-1]:
            continue
        starts.append(start)
        strengths.append(float(strength[frame]))

    strongest = max(strengths)
    max_frames = max(1, round(MAX_EVENT_SECONDS * rate / HOP))
    floor = 10 ** (SILENCE_DBFS / 20)
    events: list[SfxEvent] = []
    for index, start in enumerate(starts):
        limit = starts[index + 1] if index + 1 < len(starts) else len(rms)
        limit = min(limit, start + max_frames, len(rms))
        if limit <= start:
            continue
        window = rms[start:limit]
        loudest = int(np.argmax(window))
        below = np.nonzero(window[loudest:] < window[loudest] * 10 ** (-DECAY_DB / 20))[0]
        end = start + loudest + (int(below[0]) if below.size else len(window) - loudest)
        event_peak = float(np.max(peak[start:end]))
        if event_peak < floor:
            continue
        events.append(
            SfxEvent(
                time=round(start * HOP / rate, 4),
                duration=round(max(end - start, 1) * HOP / rate, 4),
                peak_dbfs=round(20 * math.log10(event_peak), 2),
                confidence=round(min(strengths[index] / strongest, 1.0), 3),
            )
        )
    return events, rate


def events_document(events: list[SfxEvent], sample_rate: int, source: str) -> dict[str, Any]:
    """The JSON written to sfx_events.json."""
    return {
        "version": 1,
        "source": source,
        "sample_rate": sample_rate,
        "hop_length": HOP,
        "method": "librosa onset detection, refined to the attack within one hop",
        "events": [asdict(event) for event in events],
    }


def _attack_start(peak: FloatArray, frame: int, earliest: int) -> int:
    """First hop near `frame` whose peak is within ATTACK_DB of the attack's peak."""
    lo = max(earliest, frame - SEARCH_FRAMES)
    hi = min(len(peak), frame + SEARCH_FRAMES)
    if hi <= lo:
        return frame
    local = peak[lo:hi]
    above = np.nonzero(local >= float(np.max(local)) * 10 ** (-ATTACK_DB / 20))[0]
    return lo + int(above[0]) if above.size else frame


def _frame_features(
    path: Path, on_progress: Callable[[float], None] | None
) -> tuple[FloatArray, FloatArray, FloatArray, int]:
    import librosa

    with sf.SoundFile(str(path)) as reader:
        rate, total = reader.samplerate, reader.frames
        block = max(HOP, round(BLOCK_SECONDS * rate) // HOP * HOP)
        context = CONTEXT_FRAMES * HOP
        strengths: list[FloatArray] = []
        rmses: list[FloatArray] = []
        peaks: list[FloatArray] = []
        for start in range(0, total, block):
            lead = min(context, start)
            reader.seek(start - lead)
            mono = reader.read(lead + block + context, dtype="float32", always_2d=True).mean(axis=1)
            frames = math.ceil(min(block, total - start) / HOP)
            first = lead // HOP
            strength = librosa.onset.onset_strength(y=mono, sr=rate, hop_length=HOP)
            rms = librosa.feature.rms(y=mono, frame_length=FRAME, hop_length=HOP)[0]
            strengths.append(_frames(strength, first, frames))
            rmses.append(_frames(rms, first, frames))
            peaks.append(_hop_peaks(mono[lead:], frames))
            if on_progress is not None:
                on_progress(min(start + block, total) / total)
    if not strengths:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, rate
    return np.concatenate(strengths), np.concatenate(rmses), np.concatenate(peaks), rate


def _frames(feature: FloatArray, first: int, count: int) -> FloatArray:
    out = feature[first : first + count].astype(np.float32)
    if out.shape[0] < count:
        out = np.pad(out, (0, count - out.shape[0]))
    return out


def _hop_peaks(mono: FloatArray, count: int) -> FloatArray:
    """Sample peak of each hop: frame i covers samples [i * HOP, (i + 1) * HOP)."""
    samples = np.zeros(count * HOP, dtype=np.float32)
    usable = min(len(mono), count * HOP)
    samples[:usable] = np.abs(mono[:usable])
    return samples.reshape(count, HOP).max(axis=1)
