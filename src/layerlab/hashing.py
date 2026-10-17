"""Content hashing used for job folder names and step cache keys."""

import hashlib
from collections.abc import Callable
from pathlib import Path

_CHUNK_BYTES = 1 << 20


def sha256_file(path: Path, on_progress: Callable[[float], None] | None = None) -> str:
    """Return the hex SHA-256 of a file, reporting progress as a 0..1 fraction."""
    total = path.stat().st_size
    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
            done += len(chunk)
            if on_progress is not None and total:
                on_progress(done / total)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
