"""Analyze step: build the edit timeline (cuts, SFX hits, overlay appearances).

Placeholder: no detection backend is wired in yet. The step declares its inputs
so the pipeline contract (input hashing, caching, manifest) is exercised.
"""

from pathlib import Path

from layerlab.job import Job
from layerlab.step import StepResult

NAME = "analyze"
VERSION = 1


def inputs(job: Job) -> list[Path]:
    return [job.output("extract", "video"), job.output("extract", "audio")]


def run(job: Job) -> StepResult:
    return StepResult(data={"placeholder": True})
