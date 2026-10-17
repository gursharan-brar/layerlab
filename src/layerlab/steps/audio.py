"""Audio step: separate the soundtrack into dialogue, music and sound effects.

Placeholder: no separation backend is wired in yet. The step declares its input
so the pipeline contract (input hashing, caching, manifest) is exercised.
"""

from pathlib import Path

from layerlab.job import Job
from layerlab.step import StepResult

NAME = "audio"
VERSION = 1


def inputs(job: Job) -> list[Path]:
    return [job.output("extract", "audio")]


def run(job: Job) -> StepResult:
    return StepResult(data={"placeholder": True})
