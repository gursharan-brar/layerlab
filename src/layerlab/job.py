"""A job: one input video, its job folder, and its manifest."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from layerlab.errors import LayerlabError
from layerlab.hashing import sha256_file
from layerlab.manifest import MANIFEST_NAME, Manifest, SourceInfo

JOB_ID_LENGTH = 16
STEPS_DIR = "steps"

ProgressCallback = Callable[[float], None]


def _ignore_progress(_fraction: float) -> None:
    pass


class Job:
    """Paths and state for one job folder.

    Layout::

        <out>/<sha256 of input, first 16 hex chars>/
            manifest.json
            steps/<step>/...      written only by that step
    """

    def __init__(self, root: Path, source_path: Path, manifest: Manifest) -> None:
        self.root = root
        self.source_path = source_path
        self.manifest = manifest
        self._work_dir: Path | None = None
        self._progress: ProgressCallback = _ignore_progress
        self._hash_cache: dict[tuple[str, int, int], str] = {}

    @classmethod
    def open(
        cls,
        source: Path,
        out_root: Path,
        *,
        on_hash_progress: ProgressCallback | None = None,
    ) -> Job:
        """Create the job folder for `source` under `out_root`, or reopen it."""
        source = source.resolve()
        sha = sha256_file(source, on_hash_progress)
        root = out_root.resolve() / sha[:JOB_ID_LENGTH]
        root.mkdir(parents=True, exist_ok=True)
        info = SourceInfo(path=str(source), sha256=sha, size_bytes=source.stat().st_size)
        manifest_path = root / MANIFEST_NAME
        if manifest_path.exists():
            manifest = Manifest.load(manifest_path)
            manifest.source = info
        else:
            manifest = Manifest(source=info)
        job = cls(root, source, manifest)
        job._hash_cache[job._cache_key(source)] = sha
        job.save()
        return job

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def save(self) -> None:
        self.manifest.save(self.manifest_path)

    def step_dir(self, step: str) -> Path:
        """Final output folder of a step."""
        return self.root / STEPS_DIR / step

    @property
    def work_dir(self) -> Path:
        """Folder the running step writes to. The runner promotes it on success."""
        if self._work_dir is None:
            raise RuntimeError("work_dir is only available while a step is running")
        return self._work_dir

    def output(self, step: str, key: str) -> Path:
        """Path of an output recorded by an earlier step."""
        record = self.manifest.steps.get(step)
        if record is None or record.status != "succeeded" or key not in record.outputs:
            raise LayerlabError(
                f"step '{step}' has not produced '{key}'. Run '{step}' first "
                "(it may have been skipped or have failed)."
            )
        path = self.root / record.outputs[key]
        if not path.exists():
            raise LayerlabError(
                f"output '{key}' of step '{step}' is missing at {path}. Re-run with --force."
            )
        return path

    def file_hash(self, path: Path) -> str:
        """SHA-256 of a file, cached for the life of this job object."""
        if not path.is_file():
            raise LayerlabError(f"input file not found: {path}")
        key = self._cache_key(path)
        if key not in self._hash_cache:
            self._hash_cache[key] = sha256_file(path)
        return self._hash_cache[key]

    def report_progress(self, fraction: float) -> None:
        """Called by steps to report progress as a 0..1 fraction."""
        self._progress(min(max(fraction, 0.0), 1.0))

    def begin_step(self, work_dir: Path, on_progress: ProgressCallback) -> None:
        self._work_dir = work_dir
        self._progress = on_progress

    def end_step(self) -> None:
        self._work_dir = None
        self._progress = _ignore_progress

    @staticmethod
    def _cache_key(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
