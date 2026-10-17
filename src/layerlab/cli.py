"""Command-line entry point for layerlab."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from layerlab import __version__
from layerlab.backends.catalog import BACKEND_NAMES, DEFAULT_BACKEND
from layerlab.devices import DEVICE_CHOICES
from layerlab.doctor import run_checks
from layerlab.errors import LayerlabError
from layerlab.models import cache_dir, ensure_model, is_cached, load_registry, model_path
from layerlab.models.download import DownloadProgress
from layerlab.pipeline import StepFailed, run_pipeline
from layerlab.steps import STEP_NAMES
from layerlab.steps.audio import OPTION as AUDIO_BACKEND_OPTION

app = typer.Typer(
    name="layerlab",
    help=(
        "Split a video into dialogue, music, SFX, on-screen graphics and an edit timeline. "
        "Runs locally."
    ),
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(
    help="List and download model weights. Models also download on first use.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")

_SKIP_HELP = f"Skip a step. Repeat to skip several. Steps: {', '.join(STEP_NAMES)}."
_DEVICE_HELP = f"Compute device: {', '.join(DEVICE_CHOICES)}. auto picks the best available."
_AUDIO_BACKEND_HELP = (
    f"Source separation backend: {', '.join(BACKEND_NAMES)}. "
    # The backslash stops Rich reading [demucs] as markup in --help.
    'demucs needs pip install "layerlab\\[demucs]" and has no effects stem.'
)
_LICENSE_NOTE = "Licenses are as stated by each model's publisher. Check them yourself before use."


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
    device: str = typer.Option("auto", "--device", help=_DEVICE_HELP),
    audio_backend: str = typer.Option(DEFAULT_BACKEND, "--audio-backend", help=_AUDIO_BACKEND_HELP),
) -> None:
    """Run the pipeline on INPUT. Re-running resumes, skipping steps whose inputs are unchanged."""
    unknown = sorted(set(skip) - set(STEP_NAMES))
    if unknown:
        raise typer.BadParameter(
            f"unknown step(s): {', '.join(unknown)}. Choose from: {', '.join(STEP_NAMES)}.",
            param_hint="--skip",
        )
    if device.lower() not in DEVICE_CHOICES:
        raise typer.BadParameter(
            f"unknown device '{device}'. Choose from: {', '.join(DEVICE_CHOICES)}.",
            param_hint="--device",
        )
    if audio_backend not in BACKEND_NAMES:
        raise typer.BadParameter(
            f"unknown audio backend '{audio_backend}'. Choose from: {', '.join(BACKEND_NAMES)}.",
            param_hint="--audio-backend",
        )
    # soft_wrap keeps long paths on one line so they stay copyable.
    console = Console(soft_wrap=True)
    try:
        report = run_pipeline(
            input_path,
            out,
            skip=skip,
            force=force,
            device=device,
            options={AUDIO_BACKEND_OPTION: audio_backend},
            console=console,
        )
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


@models_app.command("list")
def models_list() -> None:
    """Show every model in the registry with its license and cache status."""
    console = Console(soft_wrap=True)
    try:
        registry = load_registry()
    except LayerlabError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    table = Table()
    table.add_column("Name", no_wrap=True)
    table.add_column("Task")
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("License", no_wrap=True)
    table.add_column("Commercial")
    table.add_column("Cached")
    for spec in registry:
        table.add_row(
            spec.name,
            spec.task,
            _human_size(spec.size_bytes),
            spec.license,
            "yes" if spec.commercial_use else "[red]no",
            "yes" if is_cached(spec) else "no",
        )
    console.print(table)
    console.print(_LICENSE_NOTE)
    console.print(f"Cache: {escape(str(cache_dir()))}")


@models_app.command("fetch")
def models_fetch(
    names: list[str] = typer.Argument(..., metavar="NAME...", help="Model name(s) to download."),
) -> None:
    """Download models now instead of on first use. Interrupted downloads resume."""
    console = Console(soft_wrap=True)
    try:
        registry = load_registry()
    except LayerlabError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    unknown = [name for name in names if name not in registry.names]
    if unknown:
        raise typer.BadParameter(
            f"unknown model(s): {', '.join(unknown)}. Known models: {', '.join(registry.names)}.",
            param_hint="NAME",
        )
    for spec in (registry.get(name) for name in names):
        if is_cached(spec):
            console.print(f"{spec.name}: already cached at {escape(str(model_path(spec)))}")
            continue
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
        )
        try:
            with progress:
                task = progress.add_task(spec.name, total=spec.size_bytes)
                path = ensure_model(spec, on_progress=_download_updater(progress, task))
        except LayerlabError as exc:
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1) from exc
        console.print(f"{spec.name}: verified and saved to {escape(str(path))}")
        if not spec.commercial_use:
            console.print(
                f"[yellow]Note:[/yellow] {spec.name} is licensed {spec.license}, which is not "
                f"marked for commercial use. See {spec.homepage}"
            )


def _download_updater(progress: Progress, task: TaskID) -> DownloadProgress:
    def update(done: int, _total: int) -> None:
        progress.update(task, completed=done)

    return update


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"  # pragma: no cover - loop always returns
