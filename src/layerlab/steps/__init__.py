"""The pipeline steps, in the order they run."""

from layerlab.step import StepSpec
from layerlab.steps import analyze, audio, export, extract, inpaint

STEPS: tuple[StepSpec, ...] = (
    StepSpec(extract.NAME, extract.VERSION, extract.inputs, extract.run),
    StepSpec(audio.NAME, audio.VERSION, audio.inputs, audio.run, audio.config),
    StepSpec(analyze.NAME, analyze.VERSION, analyze.inputs, analyze.run),
    StepSpec(inpaint.NAME, inpaint.VERSION, inpaint.inputs, inpaint.run),
    StepSpec(export.NAME, export.VERSION, export.inputs, export.run),
)

STEP_NAMES: tuple[str, ...] = tuple(step.name for step in STEPS)

__all__ = ["STEPS", "STEP_NAMES"]
