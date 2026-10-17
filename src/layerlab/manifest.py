"""The job manifest: the record of what each pipeline step did.

`manifest.json` is the source of truth for a job. Steps never read each other's
code; they find each other's outputs through the manifest.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from layerlab import __version__
from layerlab.errors import LayerlabError

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"

StepStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
_STATUSES = frozenset({"pending", "running", "succeeded", "failed", "skipped"})


def utc_now() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SourceInfo:
    """The input video the job was created from."""

    path: str
    sha256: str
    size_bytes: int


@dataclass
class StepRecord:
    """What one step did the last time it ran.

    `outputs` maps an output key to a POSIX path relative to the job folder.
    `input_hash` identifies the exact inputs the outputs were built from.
    """

    status: StepStatus
    version: int
    input_hash: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


@dataclass
class Manifest:
    """Job metadata plus one record per step."""

    source: SourceInfo
    steps: dict[str, StepRecord] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    layerlab_version: str = __version__
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Manifest:
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise LayerlabError(
                f"manifest schema version {version!r} is not supported "
                f"(this layerlab writes version {SCHEMA_VERSION})"
            )
        steps: dict[str, StepRecord] = {}
        for name, record in raw.get("steps", {}).items():
            if record.get("status") not in _STATUSES:
                raise ValueError(f"step {name!r} has unknown status {record.get('status')!r}")
            steps[name] = StepRecord(**record)
        return cls(
            source=SourceInfo(**raw["source"]),
            steps=steps,
            schema_version=SCHEMA_VERSION,
            layerlab_version=str(raw.get("layerlab_version", "")),
            created_at=str(raw["created_at"]),
            updated_at=str(raw["updated_at"]),
        )

    def save(self, path: Path) -> None:
        """Write atomically so a crash never leaves a half-written manifest."""
        self.layerlab_version = __version__
        self.updated_at = utc_now()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(raw)
        except LayerlabError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise LayerlabError(
                f"could not read {path}: {exc}. Delete the job folder to start over."
            ) from exc
