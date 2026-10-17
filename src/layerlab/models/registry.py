"""The model registry: which weight files layerlab can download, and under what license."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from layerlab.errors import LayerlabError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REGISTRY_ENV = "LAYERLAB_MODEL_REGISTRY"
TASKS = frozenset({"separation", "dialogue", "music", "sfx", "graphics", "timeline", "inpaint"})

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STRING_FIELDS = ("name", "task", "description", "url", "sha256", "license", "homepage")


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable weight file."""

    name: str
    task: str
    description: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    commercial_use: bool
    homepage: str

    @property
    def filename(self) -> str:
        return Path(urlparse(self.url).path).name


class Registry:
    """Models by name, in registry file order."""

    def __init__(self, models: list[ModelSpec]) -> None:
        self._models = {model.name: model for model in models}

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._models.values())

    def __len__(self) -> int:
        return len(self._models)

    @property
    def names(self) -> list[str]:
        return list(self._models)

    def get(self, name: str) -> ModelSpec:
        try:
            return self._models[name]
        except KeyError:
            raise LayerlabError(
                f"unknown model '{name}'. Known models: {', '.join(self.names)}."
            ) from None


def load_registry(path: Path | None = None) -> Registry:
    """Load the registry from `path`, `$LAYERLAB_MODEL_REGISTRY`, or the bundled file."""
    if path is None and os.environ.get(REGISTRY_ENV):
        path = Path(os.environ[REGISTRY_ENV])
    if path is None:
        text = files("layerlab.models").joinpath("registry.toml").read_text(encoding="utf-8")
        source = "bundled registry"
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LayerlabError(f"could not read model registry {path}: {exc}") from exc
        source = str(path)
    return parse_registry(text, source)


def parse_registry(text: str, source: str = "registry") -> Registry:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise LayerlabError(f"{source} is not valid TOML: {exc}") from exc
    entries = raw.get("models", [])
    if not isinstance(entries, list):
        raise LayerlabError(f"{source}: 'models' must be an array of tables")
    models: list[ModelSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        model = _parse_entry(entry, f"{source}, model #{index + 1}")
        if model.name in seen:
            raise LayerlabError(f"{source}: duplicate model name '{model.name}'")
        seen.add(model.name)
        models.append(model)
    return Registry(models)


def _parse_entry(entry: Any, where: str) -> ModelSpec:
    if not isinstance(entry, dict):
        raise LayerlabError(f"{where}: expected a table")
    missing = [key for key in (*_STRING_FIELDS, "size", "commercial_use") if key not in entry]
    if missing:
        raise LayerlabError(f"{where}: missing {', '.join(missing)}")
    for key in _STRING_FIELDS:
        if not isinstance(entry[key], str) or not entry[key]:
            raise LayerlabError(f"{where}: '{key}' must be a non-empty string")
    name = entry["name"]
    if not _NAME.match(name):
        raise LayerlabError(f"{where}: name '{name}' must be lowercase letters, digits, . _ -")
    if entry["task"] not in TASKS:
        raise LayerlabError(f"{where}: task must be one of {', '.join(sorted(TASKS))}")
    if not _SHA256.match(entry["sha256"]):
        raise LayerlabError(f"{where}: sha256 must be 64 lowercase hex characters")
    size = entry["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise LayerlabError(f"{where}: size must be a positive integer (bytes)")
    if not isinstance(entry["commercial_use"], bool):
        raise LayerlabError(f"{where}: commercial_use must be true or false")
    url = urlparse(entry["url"])
    if url.scheme not in ("https", "http") or not url.netloc or not Path(url.path).name:
        raise LayerlabError(f"{where}: url must be an http(s) URL ending in a file name")
    return ModelSpec(
        name=name,
        task=entry["task"],
        description=entry["description"],
        url=entry["url"],
        sha256=entry["sha256"],
        size_bytes=size,
        license=entry["license"],
        commercial_use=entry["commercial_use"],
        homepage=entry["homepage"],
    )
