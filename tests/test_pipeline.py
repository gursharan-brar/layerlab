"""Runner behaviour (ordering, caching, skipping, failures) using fake steps. No ffmpeg needed."""

import json
from pathlib import Path

import pytest
from rich.console import Console

from layerlab.errors import LayerlabError
from layerlab.hashing import sha256_file
from layerlab.job import JOB_ID_LENGTH, Job
from layerlab.pipeline import StepFailed, run_pipeline
from layerlab.step import StepResult, StepSpec


def _quiet() -> Console:
    return Console(quiet=True)


def _copy_source(job: Job) -> StepResult:
    (job.work_dir / "copy.bin").write_bytes(job.source_path.read_bytes())
    job.report_progress(1.0)
    return StepResult(outputs={"copy": "copy.bin"}, data={"size": job.source_path.stat().st_size})


def _reverse_copy(job: Job) -> StepResult:
    data = job.output("first", "copy").read_bytes()
    (job.work_dir / "reversed.bin").write_bytes(data[::-1])
    return StepResult(outputs={"reversed": "reversed.bin"})


def _explode(job: Job) -> StepResult:
    raise RuntimeError("boom")


FIRST = StepSpec("first", 1, lambda job: [job.source_path], _copy_source)
SECOND = StepSpec("second", 1, lambda job: [job.output("first", "copy")], _reverse_copy)
BROKEN = StepSpec("broken", 1, lambda job: [job.output("first", "copy")], _explode)
STEPS = (FIRST, SECOND)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "clip.bin"
    path.write_bytes(b"not really a video")
    return path


def _manifest(job_root: Path) -> dict[str, object]:
    data: dict[str, object] = json.loads((job_root / "manifest.json").read_text("utf-8"))
    return data


def test_job_folder_is_named_from_input_hash(source: Path, tmp_path: Path) -> None:
    report = run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    assert report.job_root == (tmp_path / "out").resolve() / sha256_file(source)[:JOB_ID_LENGTH]


def test_steps_run_in_order_and_record_outputs(source: Path, tmp_path: Path) -> None:
    report = run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    assert [step.name for step in report.steps] == ["first", "second"]
    assert report.outcomes() == {"first": "ran", "second": "ran"}

    manifest = _manifest(report.job_root)
    steps = manifest["steps"]
    assert isinstance(steps, dict)
    assert steps["first"]["status"] == "succeeded"
    assert steps["first"]["outputs"] == {"copy": "steps/first/copy.bin"}
    assert steps["first"]["data"] == {"size": source.stat().st_size}
    assert (report.job_root / "steps" / "second" / "reversed.bin").read_bytes() == (
        source.read_bytes()[::-1]
    )
    assert not list(report.job_root.glob("steps/*.tmp"))


def test_second_run_is_cached(source: Path, tmp_path: Path) -> None:
    first = run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    started = _manifest(first.job_root)["steps"]["first"]["started_at"]  # type: ignore[index]
    second = run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    assert second.outcomes() == {"first": "cached", "second": "cached"}
    assert _manifest(second.job_root)["steps"]["first"]["started_at"] == started  # type: ignore[index]


def test_force_reruns_everything(source: Path, tmp_path: Path) -> None:
    run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    report = run_pipeline(source, tmp_path / "out", steps=STEPS, force=True, console=_quiet())
    assert report.outcomes() == {"first": "ran", "second": "ran"}


def test_changed_input_reruns_only_downstream_steps(source: Path, tmp_path: Path) -> None:
    report = run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    (report.job_root / "steps" / "first" / "copy.bin").write_bytes(b"edited by hand")
    rerun = run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    assert rerun.outcomes() == {"first": "cached", "second": "ran"}
    assert (rerun.job_root / "steps" / "second" / "reversed.bin").read_bytes() == b"dnah yb detide"


def test_missing_output_reruns_step(source: Path, tmp_path: Path) -> None:
    report = run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    (report.job_root / "steps" / "second" / "reversed.bin").unlink()
    rerun = run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    assert rerun.outcomes() == {"first": "cached", "second": "ran"}


def test_version_bump_reruns_step(source: Path, tmp_path: Path) -> None:
    run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    bumped = StepSpec("second", 2, SECOND.inputs, SECOND.run)
    report = run_pipeline(source, tmp_path / "out", steps=(FIRST, bumped), console=_quiet())
    assert report.outcomes() == {"first": "cached", "second": "ran"}


def test_skip_records_skipped_on_a_new_job(source: Path, tmp_path: Path) -> None:
    report = run_pipeline(source, tmp_path / "out", steps=STEPS, skip={"second"}, console=_quiet())
    assert report.outcomes() == {"first": "ran", "second": "skipped"}
    assert _manifest(report.job_root)["steps"]["second"]["status"] == "skipped"  # type: ignore[index]


def test_skip_keeps_an_earlier_success(source: Path, tmp_path: Path) -> None:
    run_pipeline(source, tmp_path / "out", steps=STEPS, console=_quiet())
    report = run_pipeline(source, tmp_path / "out", steps=STEPS, skip={"first"}, console=_quiet())
    assert report.outcomes() == {"first": "skipped", "second": "cached"}
    assert _manifest(report.job_root)["steps"]["first"]["status"] == "succeeded"  # type: ignore[index]


def test_skipping_a_needed_step_fails_the_next_one(source: Path, tmp_path: Path) -> None:
    with pytest.raises(StepFailed) as failure:
        run_pipeline(source, tmp_path / "out", steps=STEPS, skip={"first"}, console=_quiet())
    assert failure.value.step == "second"
    assert "has not produced 'copy'" in str(failure.value)


def test_unknown_skip_is_rejected(source: Path, tmp_path: Path) -> None:
    with pytest.raises(LayerlabError, match="unknown step"):
        run_pipeline(source, tmp_path / "out", steps=STEPS, skip={"nope"}, console=_quiet())


def test_failure_records_error_and_keeps_earlier_outputs(source: Path, tmp_path: Path) -> None:
    with pytest.raises(StepFailed) as failure:
        run_pipeline(source, tmp_path / "out", steps=(FIRST, BROKEN), console=_quiet())
    job_root = failure.value.job_root
    assert "step 'broken' failed: RuntimeError: boom" in str(failure.value)

    steps = _manifest(job_root)["steps"]
    assert isinstance(steps, dict)
    assert steps["first"]["status"] == "succeeded"
    assert steps["broken"]["status"] == "failed"
    assert steps["broken"]["error"] == "RuntimeError: boom"
    assert (job_root / "steps" / "first" / "copy.bin").exists()
    assert not (job_root / "steps" / "broken.tmp").exists()


def test_failed_step_runs_again_on_resume(source: Path, tmp_path: Path) -> None:
    with pytest.raises(StepFailed):
        run_pipeline(source, tmp_path / "out", steps=(FIRST, BROKEN), console=_quiet())
    fixed = StepSpec("broken", 1, BROKEN.inputs, _reverse_copy)
    report = run_pipeline(source, tmp_path / "out", steps=(FIRST, fixed), console=_quiet())
    assert report.outcomes() == {"first": "cached", "broken": "ran"}


def test_step_that_omits_a_declared_output_fails(source: Path, tmp_path: Path) -> None:
    liar = StepSpec("liar", 1, lambda job: [], lambda job: StepResult(outputs={"x": "x.bin"}))
    with pytest.raises(StepFailed, match="did not write its declared outputs: x"):
        run_pipeline(source, tmp_path / "out", steps=(liar,), console=_quiet())
