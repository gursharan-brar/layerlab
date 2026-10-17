"""Tests for the layerlab command-line interface."""

import pytest
from typer.testing import CliRunner

from layerlab import __version__, cli
from layerlab.cli import app
from layerlab.doctor import Check

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"layerlab {__version__}"


def test_help_lists_doctor() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output


def test_doctor_exits_zero_when_all_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_checks",
        lambda: [Check("python", "ok", "3.12.0"), Check("gpu", "warn", "none")],
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "[ok  ] python: 3.12.0" in result.output


def test_doctor_exits_nonzero_when_a_required_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_checks",
        lambda: [Check("ffmpeg", "fail", "not found on PATH")],
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "required checks failed" in result.output
