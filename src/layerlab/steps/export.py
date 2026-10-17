"""Export step: write the layers and timeline in formats editors can import.

Placeholder: no exporters are wired in yet. The step declares its inputs so the
pipeline contract (input hashing, caching, manifest) is exercised.
"""

from pathlib import Path

from layerlab.job import Job
from layerlab.step import StepResult

NAME = "export"
VERSION = 1


def inputs(job: Job) -> list[Path]:
    return [job.output("extract", "video"), job.output("extract", "audio")]


def run(job: Job) -> StepResult:
    return StepResult(data={"placeholder": True})
