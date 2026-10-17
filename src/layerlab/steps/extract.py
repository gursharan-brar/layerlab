"""Extract step: split the input into a 48 kHz stereo WAV and a video-only MP4.

The video stream is stream-copied (no re-encode). fps, resolution and duration
are probed with ffprobe and recorded in the manifest.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from layerlab import ffmpeg
from layerlab.errors import LayerlabError
from layerlab.job import Job
from layerlab.step import StepResult

NAME = "extract"
VERSION = 1

AUDIO_FILE = "audio.wav"
VIDEO_FILE = "video_only.mp4"
SAMPLE_RATE = 48_000
CHANNELS = 2
AUDIO_CODEC = "pcm_s16le"


@dataclass(frozen=True)
class MediaInfo:
    """Properties of the source video recorded in the manifest."""

    fps: float | None
    fps_rational: str | None
    width: int | None
    height: int | None
    duration_seconds: float | None
    video_codec: str | None
    source_audio_codec: str | None
    source_audio_sample_rate: int | None
    source_audio_channels: int | None
    audio_sample_rate: int = SAMPLE_RATE
    audio_channels: int = CHANNELS


def inputs(job: Job) -> list[Path]:
    return [job.source_path]


def run(job: Job) -> StepResult:
    probed = ffmpeg.probe(job.source_path)
    video, audio = _select_streams(probed, job.source_path)
    media = describe_media(probed, video, audio)
    out = job.work_dir
    ffmpeg.run(
        [
            "-i",
            str(job.source_path),
            # Output 1: the video stream only, copied bit for bit.
            "-map",
            f"0:{video['index']}",
            "-c",
            "copy",
            "-an",
            "-sn",
            "-dn",
            str(out / VIDEO_FILE),
            # Output 2: the first audio stream as 48 kHz stereo PCM.
            "-map",
            f"0:{audio['index']}",
            "-vn",
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            AUDIO_CODEC,
            str(out / AUDIO_FILE),
        ],
        duration=media.duration_seconds,
        on_progress=job.report_progress,
    )
    return StepResult(outputs={"audio": AUDIO_FILE, "video": VIDEO_FILE}, data=asdict(media))


def describe_media(
    probed: dict[str, Any], video: dict[str, Any], audio: dict[str, Any]
) -> MediaInfo:
    """Build `MediaInfo` from ffprobe output and the chosen video and audio streams."""
    rate = _frame_rate(video)
    duration = _to_float(probed.get("format", {}).get("duration"))
    if duration is None:
        duration = _to_float(video.get("duration"))
    return MediaInfo(
        fps=round(float(rate), 6) if rate is not None else None,
        fps_rational=f"{rate.numerator}/{rate.denominator}" if rate is not None else None,
        width=_to_int(video.get("width")),
        height=_to_int(video.get("height")),
        duration_seconds=duration,
        video_codec=video.get("codec_name"),
        source_audio_codec=audio.get("codec_name"),
        source_audio_sample_rate=_to_int(audio.get("sample_rate")),
        source_audio_channels=_to_int(audio.get("channels")),
    )


def _select_streams(probed: dict[str, Any], source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    streams: list[dict[str, Any]] = probed.get("streams", [])
    video = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video"
            and not stream.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise LayerlabError(f"{source.name} has no video stream")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if audio is None:
        raise LayerlabError(
            f"{source.name} has no audio stream. layerlab needs one to separate "
            "dialogue, music and sound effects."
        )
    return video, audio


def _frame_rate(video: dict[str, Any]) -> Fraction | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = video.get(key)
        try:
            rate = Fraction(str(value))
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    return None


def _to_float(value: object) -> float | None:
    try:
        return float(str(value))
    except ValueError:
        return None


def _to_int(value: object) -> int | None:
    try:
        return int(str(value))
    except ValueError:
        return None
