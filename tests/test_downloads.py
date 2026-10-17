"""Model downloads against a local HTTP server: verification, resume, caching, CLI."""

import hashlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from layerlab.cli import app
from layerlab.models import DownloadError, ensure_model, is_cached, model_path
from layerlab.models.cache import CACHE_ENV, cache_dir
from layerlab.models.download import download, partial_path
from layerlab.models.registry import REGISTRY_ENV, ModelSpec

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB, deterministic
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


@dataclass
class ServerState:
    payload: bytes = PAYLOAD
    honour_range: bool = True
    truncate_after: int | None = None
    ranges: list[str | None] = field(default_factory=list)


class _Server(ThreadingHTTPServer):
    state: ServerState


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def do_GET(self) -> None:
        state = self.server.state
        requested = self.headers.get("Range")
        state.ranges.append(requested)
        start = 0
        if requested and state.honour_range:
            start = int(requested.removeprefix("bytes=").split("-")[0])
            if start >= len(state.payload):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(state.payload) - 1}/{len(state.payload)}"
            )
        else:
            self.send_response(200)
        body = state.payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if state.truncate_after is not None:
            body = body[: state.truncate_after]
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@dataclass
class LocalServer:
    url: str
    state: ServerState


@pytest.fixture
def server() -> Iterator[LocalServer]:
    httpd = _Server(("127.0.0.1", 0), _Handler)
    httpd.state = ServerState()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield LocalServer(f"http://127.0.0.1:{httpd.server_address[1]}/weights.onnx", httpd.state)
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    return cache_dir()


def _spec(url: str, *, sha256: str = PAYLOAD_SHA256, commercial_use: bool = True) -> ModelSpec:
    return ModelSpec(
        name="test-model",
        task="dialogue",
        description="test",
        url=url,
        sha256=sha256,
        size_bytes=len(PAYLOAD),
        license="MIT" if commercial_use else "NOASSERTION",
        commercial_use=commercial_use,
        homepage="https://example.com",
    )


def test_cache_dir_honours_env(cache: Path, tmp_path: Path) -> None:
    assert cache == tmp_path / "cache" / "models"


def test_download_verifies_and_moves_into_place(server: LocalServer, tmp_path: Path) -> None:
    dest = tmp_path / "weights.onnx"
    seen: list[int] = []
    download(
        server.url,
        dest,
        sha256=PAYLOAD_SHA256,
        size=len(PAYLOAD),
        on_progress=lambda done, _total: seen.append(done),
    )
    assert dest.read_bytes() == PAYLOAD
    assert not partial_path(dest).exists()
    assert seen[-1] == len(PAYLOAD)


def test_partial_download_resumes_with_range(server: LocalServer, tmp_path: Path) -> None:
    dest = tmp_path / "weights.onnx"
    half = len(PAYLOAD) // 2
    partial_path(dest).write_bytes(PAYLOAD[:half])
    download(server.url, dest, sha256=PAYLOAD_SHA256, size=len(PAYLOAD))
    assert server.state.ranges == [f"bytes={half}-"]
    assert dest.read_bytes() == PAYLOAD


def test_server_ignoring_range_restarts_cleanly(server: LocalServer, tmp_path: Path) -> None:
    server.state.honour_range = False
    dest = tmp_path / "weights.onnx"
    partial_path(dest).write_bytes(PAYLOAD[:1000])
    download(server.url, dest, sha256=PAYLOAD_SHA256, size=len(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD


def test_complete_partial_file_is_verified_without_refetching(
    server: LocalServer, tmp_path: Path
) -> None:
    dest = tmp_path / "weights.onnx"
    partial_path(dest).write_bytes(PAYLOAD)
    download(server.url, dest, sha256=PAYLOAD_SHA256, size=len(PAYLOAD))
    assert server.state.ranges == []
    assert dest.read_bytes() == PAYLOAD


def test_interrupted_download_keeps_partial_then_resumes(
    server: LocalServer, tmp_path: Path
) -> None:
    dest = tmp_path / "weights.onnx"
    server.state.truncate_after = 300_000
    with pytest.raises(DownloadError, match=r"incomplete \(300000 of"):
        download(server.url, dest, sha256=PAYLOAD_SHA256, size=len(PAYLOAD))
    assert partial_path(dest).stat().st_size == 300_000
    assert not dest.exists()

    server.state.truncate_after = None
    download(server.url, dest, sha256=PAYLOAD_SHA256, size=len(PAYLOAD))
    assert server.state.ranges[-1] == "bytes=300000-"
    assert dest.read_bytes() == PAYLOAD


def test_checksum_mismatch_deletes_partial(server: LocalServer, tmp_path: Path) -> None:
    dest = tmp_path / "weights.onnx"
    with pytest.raises(DownloadError, match="checksum mismatch"):
        download(server.url, dest, sha256="0" * 64, size=len(PAYLOAD))
    assert not dest.exists()
    assert not partial_path(dest).exists()


def test_oversized_response_is_rejected(server: LocalServer, tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="more than the expected"):
        download(server.url, tmp_path / "w.onnx", sha256=PAYLOAD_SHA256, size=1000)


def test_unsatisfiable_range_reports_incomplete(server: LocalServer, tmp_path: Path) -> None:
    server.state.payload = b""
    partial_path(tmp_path / "w.onnx").write_bytes(b"x")
    with pytest.raises(DownloadError, match="incomplete"):
        download(server.url, tmp_path / "w.onnx", sha256=PAYLOAD_SHA256, size=10)


@pytest.mark.parametrize(
    "url", ["http://example.com/w.onnx", "file:///etc/passwd", "ftp://example.com/w.onnx"]
)
def test_only_https_or_localhost_is_allowed(url: str, tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="only https"):
        download(url, tmp_path / "w.onnx", sha256=PAYLOAD_SHA256, size=10)


def test_ensure_model_downloads_once(server: LocalServer, cache: Path) -> None:
    spec = _spec(server.url)
    assert not is_cached(spec)
    path = ensure_model(spec)
    assert path == cache / "test-model" / "weights.onnx"
    assert is_cached(spec)
    assert ensure_model(spec) == path
    assert len(server.state.ranges) == 1


def _write_registry(path: Path, url: str, *, commercial_use: bool = True) -> Path:
    path.write_text(
        f"""
[[models]]
name = "test-model"
task = "dialogue"
description = "A test model"
url = "{url}"
sha256 = "{PAYLOAD_SHA256}"
size = {len(PAYLOAD)}
license = "{"MIT" if commercial_use else "NOASSERTION"}"
commercial_use = {"true" if commercial_use else "false"}
homepage = "https://example.com"
""",
        encoding="utf-8",
    )
    return path


def test_cli_models_list(
    server: LocalServer, cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path / "registry.toml", server.url, commercial_use=False)
    monkeypatch.setenv(REGISTRY_ENV, str(registry))
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "test-model" in result.output
    assert "NOASSERTION" in result.output
    assert "Commercial" in result.output


def test_cli_models_list_shows_bundled_licenses() -> None:
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "silero-vad" in result.output
    assert "Apache-2.0" in result.output


def test_cli_models_fetch(
    server: LocalServer, cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path / "registry.toml", server.url, commercial_use=False)
    monkeypatch.setenv(REGISTRY_ENV, str(registry))
    runner = CliRunner()

    result = runner.invoke(app, ["models", "fetch", "test-model"])
    assert result.exit_code == 0, result.output
    assert "verified and saved" in result.output
    assert "not marked for commercial use" in result.output
    assert model_path(_spec(server.url)).read_bytes() == PAYLOAD

    again = runner.invoke(app, ["models", "fetch", "test-model"])
    assert again.exit_code == 0
    assert "already cached" in again.output
    assert len(server.state.ranges) == 1


def test_cli_models_fetch_unknown_name() -> None:
    result = CliRunner().invoke(app, ["models", "fetch", "no-such-model"])
    assert result.exit_code == 2
    assert "unknown model" in result.output


def test_cli_models_fetch_checksum_failure_exits_nonzero(
    server: LocalServer, cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path / "registry.toml", server.url)
    server.state.payload = PAYLOAD[::-1]
    monkeypatch.setenv(REGISTRY_ENV, str(registry))
    result = CliRunner().invoke(app, ["models", "fetch", "test-model"])
    assert result.exit_code == 1
    assert "checksum mismatch" in result.output
