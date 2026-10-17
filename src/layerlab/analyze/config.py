"""Thresholds for the analyze step's detectors, loaded from TOML.

`defaults.toml` next to this module holds every setting and its default. A user
file passed with `--analyze-config` may override any subset; unknown sections
or keys are rejected so a typo cannot be silently ignored.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, fields
from importlib.resources import files
from pathlib import Path
from typing import Any

from layerlab.errors import LayerlabError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass(frozen=True)
class SamplingConfig:
    fps: float
    max_side: int


@dataclass(frozen=True)
class CutsConfig:
    threshold: float
    min_shot_seconds: float
    luma_only: bool


@dataclass(frozen=True)
class TextConfig:
    limit_side_len: int
    pixel_threshold: float
    box_threshold: float
    unclip_ratio: float
    min_box_height: int
    iou_threshold: float
    max_gap_samples: int
    min_samples: int
    mask_padding: int


@dataclass(frozen=True)
class LogoConfig:
    analysis_width: int
    levels: int
    stable_fraction: float
    min_samples: int
    min_area_fraction: float
    max_area_fraction: float
    min_edge_density: float
    max_span_fraction: float
    presence_fraction: float
    mask_padding: int


@dataclass(frozen=True)
class PreviewConfig:
    max_side: int
    cut_marker_seconds: float


@dataclass(frozen=True)
class AnalyzeConfig:
    sampling: SamplingConfig
    cuts: CutsConfig
    text: TextConfig
    logos: LogoConfig
    preview: PreviewConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS: dict[str, type] = {
    "sampling": SamplingConfig,
    "cuts": CutsConfig,
    "text": TextConfig,
    "logos": LogoConfig,
    "preview": PreviewConfig,
}


def default_text() -> str:
    return files("layerlab.analyze").joinpath("defaults.toml").read_text(encoding="utf-8")


def load_config(override: Path | None = None) -> AnalyzeConfig:
    """Defaults, with `override`'s values on top."""
    merged = _parse(default_text(), "analyze defaults")
    if override is not None:
        try:
            text = override.read_text(encoding="utf-8")
        except OSError as exc:
            raise LayerlabError(f"could not read analyze config {override}: {exc}") from exc
        for section, values in _parse(text, str(override)).items():
            if section not in merged:
                raise LayerlabError(
                    f"{override}: unknown section [{section}]. Sections: {', '.join(_SECTIONS)}."
                )
            if not isinstance(values, dict):
                raise LayerlabError(f"{override}: [{section}] must be a table")
            unknown = sorted(set(values) - set(merged[section]))
            if unknown:
                raise LayerlabError(
                    f"{override}: unknown key(s) in [{section}]: {', '.join(unknown)}"
                )
            merged[section].update(values)
    return AnalyzeConfig(
        **{name: _build(cls, name, merged[name]) for name, cls in _SECTIONS.items()}
    )


def _parse(text: str, source: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise LayerlabError(f"{source} is not valid TOML: {exc}") from exc


def _build(cls: type, section: str, values: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        value = values.get(item.name)
        expected = {"float": float, "int": int, "bool": bool}[str(item.type)]
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise LayerlabError(
                f"analyze config [{section}] {item.name} must be {expected.__name__}, got {value!r}"
            )
        kwargs[item.name] = value
    return cls(**kwargs)
