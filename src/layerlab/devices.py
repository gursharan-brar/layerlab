"""Compute device detection: CUDA, ROCm, Apple MPS, or CPU.

If PyTorch is installed its view wins, because it reflects what model backends
can actually use (a CPU-only PyTorch on a CUDA machine cannot use the GPU).
Without PyTorch, layerlab looks for vendor tools and hardware instead.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

from layerlab.errors import LayerlabError

DeviceName = Literal["cuda", "rocm", "mps", "cpu"]
DEVICE_NAMES: tuple[DeviceName, ...] = ("cuda", "rocm", "mps", "cpu")
DEVICE_CHOICES: tuple[str, ...] = ("auto", *DEVICE_NAMES)

_PROBE_TIMEOUT_SECONDS = 10
_ROCM_KERNEL_DEVICE = Path("/dev/kfd")


@dataclass(frozen=True)
class Device:
    name: DeviceName
    detail: str


@dataclass(frozen=True)
class DeviceChoice:
    """The device a run uses and how it was chosen."""

    name: DeviceName
    detail: str
    requested: str


def detect_devices() -> list[Device]:
    """Available devices, best first. Always ends with the CPU."""
    found = _torch_devices()
    if found is None:
        found = _hardware_devices()
    return [*found, Device("cpu", platform.processor() or platform.machine() or "CPU")]


def select_device(requested: str = "auto") -> DeviceChoice:
    """Pick the best device, or honour an explicit `requested` device."""
    requested = requested.lower()
    if requested not in DEVICE_CHOICES:
        raise LayerlabError(
            f"unknown device '{requested}'. Choose from: {', '.join(DEVICE_CHOICES)}."
        )
    devices = detect_devices()
    if requested == "auto":
        best = devices[0]
        return DeviceChoice(best.name, best.detail, requested)
    for device in devices:
        if device.name == requested:
            return DeviceChoice(device.name, device.detail, requested)
    # An explicit choice wins even if detection missed it; record that it was not detected.
    name: DeviceName = next(name for name in DEVICE_NAMES if name == requested)
    return DeviceChoice(name, "requested with --device; not detected on this machine", requested)


def _torch_devices() -> list[Device] | None:
    torch = _import_torch()
    if torch is None:
        return None
    found: list[Device] = []
    try:
        if torch.cuda.is_available():
            kind: DeviceName = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            found.append(Device(kind, f"{torch.cuda.get_device_name(0)} (PyTorch)"))
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            found.append(Device("mps", "Apple Metal (PyTorch)"))
    except Exception:  # A broken GPU driver must not stop detection; fall back to CPU.
        return found
    return found


def _import_torch() -> Any:
    if find_spec("torch") is None:
        return None
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def _hardware_devices() -> list[Device]:
    found: list[Device] = []
    nvidia = _nvidia_gpus()
    if nvidia:
        found.append(Device("cuda", "; ".join(nvidia)))
    if _has_rocm():
        found.append(Device("rocm", "AMD GPU (ROCm runtime found)"))
    if _is_apple_silicon():
        found.append(Device("mps", "Apple Silicon GPU (Metal)"))
    return found


def _nvidia_gpus() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _has_rocm() -> bool:
    if shutil.which("rocminfo") is not None or shutil.which("rocm-smi") is not None:
        return True
    return sys.platform == "linux" and _ROCM_KERNEL_DEVICE.exists()


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"
