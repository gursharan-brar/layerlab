"""Runs the pipeline steps in order against a job folder.

For each step the runner:

1. hashes the step's declared inputs (plus its name and version),
2. skips the step if the manifest already records a success with that hash and
   its outputs still exist (unless `force` is set),
3. otherwise runs it in a temporary folder, then swaps that folder into place
   and records the outputs, or records the error and stops.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from layerlab.errors import LayerlabError
from layerlab.hashing import sha256_text
from layerlab.job import Job, ProgressCallback
from layerlab.manifest import StepRecord, utc_now
from layerlab.step import StepSpec
from layerlab.steps import STEPS

Outcome = Literal["ran", "cached", "skipped"]

_STATUS_LABELS: dict[Outcome, str] = {
    "ran": "[green]done",
    "cached": "[cyan]cached",
    "skipped": "[yellow]skipped",
}


@dataclass(frozen=True)
class StepReport:
    name: str
    outcome: Outcome


@dataclass
class RunReport:
    job_root: Path
    steps: list[StepReport] = field(default_factory=list)

    def outcomes(self) -> dict[str, Outcome]:
        return {report.name: report.outcome for report in self.steps}


class StepFailed(LayerlabError):
    """A step raised. Its error is recorded in the manifest."""

    def __init__(self, step: str, message: str, job: Job) -> None:
        super().__init__(f"step '{step}' failed: {message}")
        self.step = step
        self.job_root = job.root
        self.manifest_path = job.manifest_path


def run_pipeline(
    source: Path,
    out_root: Path,
    *,
    skip: Collection[str] = (),
    force: bool = False,
    steps: Sequence[StepSpec] = STEPS,
    console: Console | None = None,
) -> RunReport:
    """Run `steps` for `source`, creating or resuming its job folder under `out_root`."""
    names = [spec.name for spec in steps]
    unknown = sorted(set(skip) - set(names))
    if unknown:
        raise LayerlabError(
            f"unknown step(s) to skip: {', '.join(unknown)}. Steps are: {', '.join(names)}."
        )

    with _progress_display(console or Console()) as progress:
        hash_task = progress.add_task("hash input", total=1.0, status="")
        job = Job.open(source, out_root, on_hash_progress=_updater(progress, hash_task))
        progress.update(hash_task, completed=1.0, status="[green]done")

        tasks = {
            spec.name: progress.add_task(spec.name, total=1.0, status="pending") for spec in steps
        }
        for spec in steps:
            job.manifest.steps.setdefault(spec.name, StepRecord("pending", spec.version))
        job.save()

        report = RunReport(job_root=job.root)
        for spec in steps:
            task = tasks[spec.name]
            progress.update(task, status="running")
            try:
                outcome = _run_step(
                    job,
                    spec,
                    skip=spec.name in skip,
                    force=force,
                    on_progress=_updater(progress, task),
                )
            except BaseException:
                progress.update(task, status="[red]failed")
                raise
            label = _STATUS_LABELS[outcome]
            if outcome != "skipped":
                progress.update(task, completed=1.0)
                if job.manifest.steps[spec.name].data.get("placeholder"):
                    label += " (placeholder)"
            progress.update(task, status=label)
            report.steps.append(StepReport(spec.name, outcome))
    return report


def _run_step(
    job: Job, spec: StepSpec, *, skip: bool, force: bool, on_progress: ProgressCallback
) -> Outcome:
    previous = job.manifest.steps[spec.name]
    if skip:
        # A skipped step keeps an earlier success so later steps can still use its outputs.
        if previous.status != "succeeded":
            job.manifest.steps[spec.name] = StepRecord("skipped", spec.version)
            job.save()
        return "skipped"

    started_at = utc_now()
    started = time.perf_counter()
    try:
        input_hash = _input_hash(job, spec)
    except Exception as exc:
        _record_failure(job, spec, None, started_at, started, _describe(exc))
        raise StepFailed(spec.name, _describe(exc), job) from exc

    if not force and _is_current(job, previous, spec, input_hash):
        return "cached"

    work = job.step_dir(spec.name).with_name(f"{spec.name}.tmp")
    _remove(work)
    work.mkdir(parents=True)
    job.manifest.steps[spec.name] = StepRecord(
        "running", spec.version, input_hash=input_hash, started_at=started_at
    )
    job.save()

    job.begin_step(work, on_progress)
    try:
        result = spec.run(job)
        outputs = _promote(job, spec.name, work, result.outputs)
    except KeyboardInterrupt:
        _remove(work)
        _record_failure(job, spec, input_hash, started_at, started, "interrupted by user")
        raise
    except Exception as exc:
        _remove(work)
        _record_failure(job, spec, input_hash, started_at, started, _describe(exc))
        raise StepFailed(spec.name, _describe(exc), job) from exc
    finally:
        job.end_step()

    record = job.manifest.steps[spec.name]
    record.status = "succeeded"
    record.outputs = outputs
    record.data = dict(result.data)
    record.finished_at = utc_now()
    record.duration_seconds = round(time.perf_counter() - started, 3)
    job.save()
    return "ran"


def _input_hash(job: Job, spec: StepSpec) -> str:
    payload = {
        "step": spec.name,
        "version": spec.version,
        "inputs": [job.file_hash(path) for path in spec.inputs(job)],
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def _is_current(job: Job, record: StepRecord, spec: StepSpec, input_hash: str) -> bool:
    return (
        record.status == "succeeded"
        and record.version == spec.version
        and record.input_hash == input_hash
        and all((job.root / path).exists() for path in record.outputs.values())
    )


def _promote(job: Job, step: str, work: Path, outputs: dict[str, str]) -> dict[str, str]:
    """Move a finished work folder into place and return manifest-relative output paths."""
    missing = sorted(key for key, name in outputs.items() if not (work / name).exists())
    if missing:
        raise LayerlabError(f"step did not write its declared outputs: {', '.join(missing)}")
    final = job.step_dir(step)
    _remove(final)
    work.rename(final)
    return {key: (final / name).relative_to(job.root).as_posix() for key, name in outputs.items()}


def _record_failure(
    job: Job,
    spec: StepSpec,
    input_hash: str | None,
    started_at: str,
    started: float,
    message: str,
) -> None:
    job.manifest.steps[spec.name] = StepRecord(
        "failed",
        spec.version,
        input_hash=input_hash,
        started_at=started_at,
        finished_at=utc_now(),
        duration_seconds=round(time.perf_counter() - started, 3),
        error=message,
    )
    job.save()


def _describe(exc: BaseException) -> str:
    if isinstance(exc, LayerlabError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _remove(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _progress_display(console: Console) -> Progress:
    return Progress(
        TextColumn("{task.description:<10}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
        console=console,
    )


def _updater(progress: Progress, task: TaskID) -> ProgressCallback:
    def update(fraction: float) -> None:
        progress.update(task, completed=fraction)

    return update
