"""The text detection interface and the PaddleOCR detection backend.

PaddleOCR's text detector is a DB (Differentiable Binarization) network: it
predicts, for every pixel, the probability that it belongs to text. This module
runs the PP-OCRv3 detection model exported to ONNX (`paddleocr-det-v3` in the
registry, Apache-2.0) on onnxruntime, so no PaddlePaddle install is needed. Only
detection runs; nothing is recognised.

Pre- and post-processing follow PaddleOCR's `DetResizeForTest`,
`NormalizeImage` and `DBPostProcess`, simplified to axis-aligned boxes:

1. resize so the longer side is at most `limit_side_len`, both sides multiples of 32,
2. scale to 0..1 and normalise with ImageNet mean/std (applied to BGR, as PaddleOCR does),
3. threshold the probability map at `pixel_threshold`, take connected regions,
4. keep regions whose mean probability is at least `box_threshold`,
5. grow each box by `unclip_ratio` (area * ratio / perimeter on every side).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from layerlab.errors import LayerlabError
from layerlab.models import get_model, is_cached, load_registry

MODEL_NAME = "paddleocr-det-v3"
MODELS: tuple[str, ...] = (MODEL_NAME,)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PROVIDERS = {
    "cuda": ["CUDAExecutionProvider"],
    "rocm": ["ROCMExecutionProvider", "MIGraphXExecutionProvider"],
    "mps": ["CoreMLExecutionProvider"],
    "cpu": [],
}

Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class TextBox:
    """An axis-aligned text box in frame pixels, with the detector's confidence."""

    x: int
    y: int
    width: int
    height: int
    score: float


class TextDetector(Protocol):
    name: str

    def detect(self, frame: Frame) -> list[TextBox]:
        """Text boxes in a BGR uint8 frame, in that frame's pixel coordinates."""
        ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DBSettings:
    limit_side_len: int = 960
    pixel_threshold: float = 0.3
    box_threshold: float = 0.6
    unclip_ratio: float = 1.5
    min_size: int = 3


class PaddleTextDetector:
    name = "paddleocr-det"

    def __init__(self, session: Any, settings: DBSettings, model_sha256: str) -> None:
        self._session = session
        self._input = session.get_inputs()[0].name
        self._settings = settings
        self._sha256 = model_sha256

    @classmethod
    def load(cls, device: str, settings: DBSettings, on_status: Any = None) -> PaddleTextDetector:
        import onnxruntime as ort

        spec = load_registry().get(MODEL_NAME)
        if on_status is not None and not is_cached(spec):
            on_status(f"downloading {MODEL_NAME}")
        path = get_model(MODEL_NAME)
        available = set(ort.get_available_providers())
        providers = [p for p in _PROVIDERS.get(device, []) if p in available]
        providers.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        try:
            session = ort.InferenceSession(str(path), options, providers=providers)
        except Exception as exc:
            raise LayerlabError(f"could not load {path} with onnxruntime: {exc}") from exc
        return cls(session, settings, spec.sha256)

    def describe(self) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "model_sha256": self._sha256,
            "runtime": "onnxruntime",
            "provider": self._session.get_providers()[0],
        }

    def detect(self, frame: Frame) -> list[TextBox]:
        import cv2

        height, width = frame.shape[:2]
        resized, scale_x, scale_y = _resize(frame, self._settings.limit_side_len)
        image = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        probability = self._session.run(None, {self._input: tensor})[0][0, 0]
        bitmap = (probability > self._settings.pixel_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[TextBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < self._settings.min_size:
                continue
            region = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(region, [contour - np.array([x, y])], -1, 1, thickness=-1)
            score = float(probability[y : y + h, x : x + w][region > 0].mean())
            if score < self._settings.box_threshold:
                continue
            grow = w * h * self._settings.unclip_ratio / (2 * (w + h))
            x0 = max(0, round((x - grow) * scale_x))
            y0 = max(0, round((y - grow) * scale_y))
            x1 = min(width, round((x + w + grow) * scale_x))
            y1 = min(height, round((y + h + grow) * scale_y))
            if x1 - x0 >= self._settings.min_size and y1 - y0 >= self._settings.min_size:
                boxes.append(TextBox(x0, y0, x1 - x0, y1 - y0, round(score, 4)))
        return boxes


def _resize(frame: Frame, limit: int) -> tuple[Frame, float, float]:
    """Resize like PaddleOCR's DetResizeForTest; return the image and the scale back."""
    import cv2

    height, width = frame.shape[:2]
    ratio = min(1.0, limit / max(height, width))
    new_h = max(32, round(height * ratio / 32) * 32)
    new_w = max(32, round(width * ratio / 32) * 32)
    resized = cv2.resize(frame, (new_w, new_h))
    return np.asarray(resized, dtype=np.uint8), width / new_w, height / new_h
