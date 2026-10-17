"""Tests for the environment checks behind `layerlab doctor`."""

import shutil
import subprocess

import pytest

from layerlab import doctor
from layerlab.doctor import check_ffmpeg, check_gpu, check_python


def test_python_meets_minimum() -> None:
    assert check_python((3, 10, 0)).status == "ok"


def test_python_below_minimum_fails() -> None:
    check = check_python((3, 9, 18))
    assert check.status == "fail"
    assert "requires 3.10+" in check.detail


def test_ffmpeg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert check_ffmpeg().status == "fail"


def test_ffmpeg_present_reports_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="ffmpeg version 7.0 Copyright\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    check = check_ffmpeg()
    assert check.status == "ok"
    assert "ffmpeg version 7.0" in check.detail


def test_gpu_nvidia_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="NVIDIA RTX 4090, 24564 MiB\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    check = check_gpu()
    assert check.status == "ok"
    assert "RTX 4090" in check.detail


def test_gpu_absent_is_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(doctor, "_is_apple_silicon", lambda: False)
    assert check_gpu().status == "warn"
