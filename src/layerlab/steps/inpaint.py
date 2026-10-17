"""Inpaint step: remove on-screen graphics to recover a clean plate.

Placeholder: no inpainting backend is wired in yet. The step declares its input
so the pipeline contract (input hashing, caching, manifest) is exercised.
"""

from pathlib import Path

from layerlab.job import Job
from layerlab.step import StepResult

NAME = "inpaint"
VERSION = 1


def inputs(job: Job) -> list[Path]:
    return [job.output("extract", "video")]


def run(job: Job) -> StepResult:
    return StepResult(data={"placeholder": True})
