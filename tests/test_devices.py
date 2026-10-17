"""Device detection and selection. Hardware and PyTorch are faked."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from layerlab import devices
from layerlab.devices import detect_devices, select_device
from layerlab.errors import LayerlabError


@pytest.fixture
def no_hardware(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No PyTorch, no vendor tools, not Apple Silicon, no /dev/kfd."""
    monkeypatch.setattr(devices, "_import_torch", lambda: None)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(devices, "_is_apple_silicon", lambda: False)
    monkeypatch.setattr(devices, "_ROCM_KERNEL_DEVICE", tmp_path / "no-kfd")


def _fake_torch(*, cuda: bool, hip: str | None = None, mps: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda _index: "Fake GPU"),
        version=SimpleNamespace(hip=hip),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.usefixtures("no_hardware")
def test_cpu_when_nothing_else() -> None:
    assert [device.name for device in detect_devices()] == ["cpu"]
    assert select_device().name == "cpu"


@pytest.mark.usefixtures("no_hardware")
def test_nvidia_smi_means_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None
    )

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="NVIDIA RTX 4090, 24564 MiB\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    choice = select_device()
    assert (choice.name, choice.detail, choice.requested) == (
        "cuda",
        "NVIDIA RTX 4090, 24564 MiB",
        "auto",
    )


@pytest.mark.usefixtures("no_hardware")
def test_rocm_tools_mean_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: name if name == "rocminfo" else None)
    assert select_device().name == "rocm"


@pytest.mark.usefixtures("no_hardware")
def test_apple_silicon_means_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_is_apple_silicon", lambda: True)
    assert select_device().name == "mps"


@pytest.mark.usefixtures("no_hardware")
def test_torch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_import_torch", lambda: _fake_torch(cuda=True))
    choice = select_device()
    assert choice.name == "cuda"
    assert "Fake GPU" in choice.detail


@pytest.mark.usefixtures("no_hardware")
def test_torch_hip_build_means_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_import_torch", lambda: _fake_torch(cuda=True, hip="6.0"))
    assert select_device().name == "rocm"


@pytest.mark.usefixtures("no_hardware")
def test_torch_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_import_torch", lambda: _fake_torch(cuda=False, mps=True))
    assert select_device().name == "mps"


@pytest.mark.usefixtures("no_hardware")
def test_cpu_only_torch_overrides_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    # A GPU is present, but PyTorch cannot use it, so the CPU is chosen.
    monkeypatch.setattr(devices, "_nvidia_gpus", lambda: ["NVIDIA RTX 4090"])
    monkeypatch.setattr(devices, "_import_torch", lambda: _fake_torch(cuda=False))
    assert select_device().name == "cpu"


@pytest.mark.usefixtures("no_hardware")
def test_explicit_device_overrides_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_nvidia_gpus", lambda: ["NVIDIA RTX 4090"])
    choice = select_device("CPU")
    assert (choice.name, choice.requested) == ("cpu", "cpu")


@pytest.mark.usefixtures("no_hardware")
def test_explicit_undetected_device_is_honoured_and_flagged() -> None:
    choice = select_device("cuda")
    assert choice.name == "cuda"
    assert "not detected" in choice.detail


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(LayerlabError, match="unknown device 'tpu'"):
        select_device("tpu")
