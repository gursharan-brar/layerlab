"""Resumable, checksum-verified downloads of model weights.

Data is streamed into `<file>.part`. If a `.part` file already exists the
download resumes with an HTTP Range request. The file is only renamed into
place after its size and SHA-256 match the registry.
"""

from __future__ import annotations

import http.client
import os
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from layerlab import __version__
from layerlab.errors import LayerlabError
from layerlab.hashing import sha256_file

_CHUNK_BYTES = 1 << 20
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

DownloadProgress = Callable[[int, int], None]
"""Called with (bytes_done, bytes_total)."""


class DownloadError(LayerlabError):
    """A model download failed or did not match its checksum."""


def partial_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


def download(
    url: str,
    dest: Path,
    *,
    sha256: str,
    size: int,
    on_progress: DownloadProgress | None = None,
    timeout: float = 30.0,
) -> None:
    """Download `url` to `dest`, resuming a previous partial download if there is one."""
    _check_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = partial_path(dest)
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        part.unlink()
        offset = 0
    if offset < size:
        _fetch(url, part, offset, size, on_progress, timeout)
    elif on_progress is not None:
        on_progress(size, size)

    received = part.stat().st_size
    if received != size:
        raise DownloadError(
            f"download of {dest.name} is incomplete ({received} of {size} bytes). "
            "Run the command again to resume."
        )
    actual = sha256_file(part)
    if actual != sha256:
        part.unlink()
        raise DownloadError(
            f"checksum mismatch for {dest.name}: expected {sha256}, got {actual}. "
            "The partial file was deleted; run the command again to retry."
        )
    os.replace(part, dest)


def _fetch(
    url: str,
    part: Path,
    offset: int,
    size: int,
    on_progress: DownloadProgress | None,
    timeout: float,
) -> None:
    headers = {"User-Agent": f"layerlab/{__version__}"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    host = urlparse(url).netloc
    try:
        response = urlopen(Request(url, headers=headers), timeout=timeout)
    except HTTPError as exc:
        if exc.code == 416 and offset:
            return  # The server says nothing is left to send; the checksum decides.
        raise DownloadError(f"download from {host} failed: HTTP {exc.code} {exc.reason}") from exc
    except (URLError, OSError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        raise DownloadError(
            f"could not reach {host}: {reason}. Check your connection and run again."
        ) from exc

    with response:
        if offset and response.status != 206:
            offset = 0  # The server ignored the Range header and is sending the whole file.
        done = offset
        try:
            with part.open("ab" if offset else "wb") as handle:
                while chunk := response.read(_CHUNK_BYTES):
                    done += len(chunk)
                    if done > size:
                        raise DownloadError(
                            f"{host} sent more than the expected {size} bytes; "
                            "the registry entry may be out of date"
                        )
                    handle.write(chunk)
                    if on_progress is not None:
                        on_progress(done, size)
        except (OSError, http.client.HTTPException) as exc:
            raise DownloadError(
                f"download interrupted after {done} of {size} bytes ({exc}). "
                "Run the command again to resume."
            ) from exc


def _check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS:
        return
    raise DownloadError(f"refusing to download from {url}: only https URLs are allowed")
