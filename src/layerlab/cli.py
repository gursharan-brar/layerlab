"""Command-line entry point for layerlab."""

import typer

from layerlab import __version__
from layerlab.doctor import run_checks

app = typer.Typer(
    name="layerlab",
    help=(
        "Split a video into dialogue, music, SFX, on-screen graphics and an edit timeline. "
        "Runs locally."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"layerlab {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """layerlab command group."""


@app.command()
def doctor() -> None:
    """Check the local environment: Python version, ffmpeg on PATH, and GPU."""
    checks = run_checks()
    for check in checks:
        typer.echo(f"[{check.status:<4}] {check.name}: {check.detail}")
    if any(check.status == "fail" for check in checks):
        typer.echo("One or more required checks failed.")
        raise typer.Exit(code=1)
