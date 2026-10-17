"""Group per-sample text boxes into overlay tracks.

Boxes come from a `TextDetector` run on frames sampled at `sampling.fps`. A box
joins an open track when it overlaps that track's latest box by at least
`iou_threshold` and the track was last seen no more than `max_gap_samples`
samples ago; otherwise it starts a new track. Tracks seen in fewer than
`min_samples` samples are dropped, so only text that persists in one region
becomes an overlay. Missed samples inside a track keep the previous box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from layerlab.analyze.config import TextConfig
from layerlab.backends.text_detection import TextBox

Box = tuple[int, int, int, int]  # x, y, width, height


@dataclass
class TextTrack:
    boxes: dict[int, Box] = field(default_factory=dict)
    scores: list[float] = field(default_factory=list)

    @property
    def first(self) -> int:
        return min(self.boxes)

    @property
    def last(self) -> int:
        return max(self.boxes)

    @property
    def bbox(self) -> Box:
        """Union of every box in the track."""
        x0 = min(box[0] for box in self.boxes.values())
        y0 = min(box[1] for box in self.boxes.values())
        x1 = max(box[0] + box[2] for box in self.boxes.values())
        y1 = max(box[1] + box[3] for box in self.boxes.values())
        return (x0, y0, x1 - x0, y1 - y0)

    @property
    def confidence(self) -> float:
        return round(sum(self.scores) / len(self.scores), 4)

    def filled(self) -> dict[int, Box]:
        """A box for every sample from first to last, holding the last box over gaps."""
        out: dict[int, Box] = {}
        current = self.boxes[self.first]
        for sample in range(self.first, self.last + 1):
            current = self.boxes.get(sample, current)
            out[sample] = current
        return out


def iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


class TextTracker:
    def __init__(self, settings: TextConfig, scale: tuple[float, float] = (1.0, 1.0)) -> None:
        """`scale` converts detector coordinates to video pixels (x, y)."""
        self._settings = settings
        self._scale = scale
        self._tracks: list[TextTrack] = []

    def update(self, sample: int, detections: list[TextBox]) -> None:
        sx, sy = self._scale
        boxes = [
            (
                (round(d.x * sx), round(d.y * sy), round(d.width * sx), round(d.height * sy)),
                d.score,
            )
            for d in detections
        ]
        boxes = [item for item in boxes if item[0][3] >= self._settings.min_box_height]
        taken: set[int] = set()
        for box, score in sorted(boxes, key=lambda item: -item[1]):
            best, best_iou = None, self._settings.iou_threshold
            for index, track in enumerate(self._tracks):
                if index in taken or sample - track.last > self._settings.max_gap_samples + 1:
                    continue
                overlap = iou(track.boxes[track.last], box)
                if overlap >= best_iou:
                    best, best_iou = index, overlap
            if best is None:
                self._tracks.append(TextTrack({sample: box}, [score]))
                taken.add(len(self._tracks) - 1)
            else:
                self._tracks[best].boxes[sample] = box
                self._tracks[best].scores.append(score)
                taken.add(best)

    def tracks(self) -> list[TextTrack]:
        """Tracks seen in at least `min_samples` samples, ordered by first appearance."""
        kept = [t for t in self._tracks if len(t.boxes) >= self._settings.min_samples]
        return sorted(kept, key=lambda track: (track.first, track.bbox))
