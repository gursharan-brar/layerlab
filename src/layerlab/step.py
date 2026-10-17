"""The contract between the pipeline runner and its steps.

A step module defines `NAME`, `VERSION`, `inputs(job)` and `run(job)`, and
optionally `config(job)`:

- `inputs(job)` lists the files the step reads.
- `config(job)` returns the JSON-serialisable settings that change the step's
  outputs (for example the chosen backend and its model hashes).
- The input hash is built from the step name, version, config and the hashes of
  its inputs. If it matches the one recorded in the manifest and the outputs
  still exist, the step is skipped.
- `run(job)` writes files into `job.work_dir` and returns a `StepResult`
  naming them. Bump `VERSION` whenever a change alters a step's outputs.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from layerlab.job import Job


@dataclass(frozen=True)
class StepResult:
    """What a step produced.

    `outputs` maps an output key to a file name inside `job.work_dir`.
    `data` is JSON-serialisable metadata stored in the step's manifest record.
    """

    outputs: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)


def no_config(_job: Job) -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class StepSpec:
    """A step as the runner sees it."""

    name: str
    version: int
    inputs: Callable[[Job], list[Path]]
    run: Callable[[Job], StepResult]
    config: Callable[[Job], dict[str, Any]] = no_config
