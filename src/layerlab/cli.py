"""Command-line entry point for layerlab."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from layerlab import __version__
from layerlab.doctor import run_checks
from layerlab.errors import LayerlabError
from layerlab.pipeline import StepFailed, run_pipeline
from layerlab.steps import STEP_NAMES

app = typer.Typer(
    name="layerlab",
    help=(
        "Split a video into dialogue, music, SFX, on-screen graphics and an edit timeline. "
        "Runs locally."
    ),
    no_args_is_help=True,
    add_completion=False,
)

_SKIP_HELP = f"Skip a step. Repeat to skip several. Steps: {', '.join(STEP_NAMES)}."


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


@app.command()
def run(
    input_path: Path = typer.Argument(
        ...,
        metavar="INPUT",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Video file to process.",
    ),
    out: Path = typer.Option(
        Path("output"),
        "--out",
        "-o",
        file_okay=False,
        help="Folder that holds job folders. Each job is named from the input's hash.",
    ),
    skip: list[str] = typer.Option(
        [],
        "--skip",
        help=_SKIP_HELP,
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run every step, ignoring cached outputs."
    ),
) -> None:
    """Run the pipeline on INPUT. Re-running resumes, skipping steps whose inputs are unchanged."""
    unknown = sorted(set(skip) - set(STEP_NAMES))
    if unknown:
        raise typer.BadParameter(
            f"unknown step(s): {', '.join(unknown)}. Choose from: {', '.join(STEP_NAMES)}.",
            param_hint="--skip",
        )
    # soft_wrap keeps long paths on one line so they stay copyable.
    console = Console(soft_wrap=True)
    try:
        report = run_pipeline(input_path, out, skip=skip, force=force, console=console)
    except StepFailed as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        console.print(
            f"Outputs of earlier steps are kept. The error is recorded in "
            f"{escape(str(exc.manifest_path))}. Fix the problem and run again to resume."
        )
        raise typer.Exit(code=1) from exc
    except LayerlabError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    console.print(f"Job folder: {escape(str(report.job_root))}")
