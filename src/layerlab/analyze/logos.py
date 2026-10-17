"""Static logo detection from per-pixel temporal stability.

Pass 1 (`StabilityAccumulator`) streams the sampled frames once at a small
analysis size. For every pixel it keeps a histogram of quantised luma
(`levels` bins) plus the running mean and variance. A pixel is near-constant in
a sample when its luma is within one level of that pixel's most common level,
and static when that holds in at least `stable_fraction` of samples.

Static pixels are cleaned up morphologically and split into connected regions.
Regions are kept if their area is within [`min_area_fraction`,
`max_area_fraction`] of the frame, they have enough edges
(`min_edge_density`) and they span less than `max_span_fraction` of the frame
in each direction: flat regions and letterbox or pillarbox bars are not logos.

Pass 2 (`presence`) re-reads the samples and marks a region present in a sample
when at least `presence_fraction` of its pixels sit at their usual level, which
gives each logo track its start and end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from layerlab.analyze.config import LogoConfig

Frame = NDArray[np.uint8]
Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class LogoRegion:
    mask: Mask  # at analysis size
    bbox: tuple[int, int, int, int]  # x, y, width, height at analysis size
    stable_fraction: float
    temporal_std: float
    edge_density: float


class StabilityAccumulator:
    def __init__(self, size: tuple[int, int], settings: LogoConfig) -> None:
        width, height = size
        self._settings = settings
        self._shape = (height, width)
        self._pixels = width * height
        self._shift = 8 - int(np.log2(settings.levels))
        self._hist = np.zeros(settings.levels * self._pixels, dtype=np.uint32)
        self._offsets = np.arange(self._pixels, dtype=np.int64)
        self._mean = np.zeros(self._shape, dtype=np.float64)
        self._m2 = np.zeros(self._shape, dtype=np.float64)
        self.samples = 0

    def quantise(self, gray: Frame) -> NDArray[np.int64]:
        return (gray >> self._shift).astype(np.int64)

    def update(self, gray: Frame) -> None:
        level = self.quantise(gray).ravel()
        self._hist += np.bincount(
            level * self._pixels + self._offsets, minlength=self._hist.size
        ).astype(np.uint32)
        self.samples += 1
        delta = gray - self._mean
        self._mean += delta / self.samples
        self._m2 += delta * (gray - self._mean)

    @property
    def mode(self) -> NDArray[np.int64]:
        """Most common luma level of every pixel."""
        hist = self._hist.reshape(self._settings.levels, *self._shape)
        result: NDArray[np.int64] = hist.argmax(axis=0)
        return result

    def stable_fraction(self) -> NDArray[np.float64]:
        """Fraction of samples in which each pixel was within one level of its mode."""
        hist = self._hist.reshape(self._settings.levels, *self._shape).astype(np.float64)
        padded = np.pad(hist, ((1, 1), (0, 0), (0, 0)))
        mode = self.mode
        rows, cols = np.indices(self._shape)
        near = (
            padded[mode, rows, cols] + padded[mode + 1, rows, cols] + padded[mode + 2, rows, cols]
        )
        return near / max(self.samples, 1)

    def regions(self) -> list[LogoRegion]:
        import cv2

        s = self._settings
        if self.samples < s.min_samples:
            return []
        fraction = self.stable_fraction()
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(
            (fraction >= s.stable_fraction).astype(np.uint8), cv2.MORPH_OPEN, kernel
        )
        static = np.asarray(cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel), dtype=np.uint8)
        mean = np.clip(self._mean, 0, 255).astype(np.uint8)
        edges = cv2.Canny(mean, 50, 150) > 0
        std = np.sqrt(self._m2 / max(self.samples - 1, 1))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(static, connectivity=8)
        regions: list[LogoRegion] = []
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            share = area / self._pixels
            if not s.min_area_fraction <= share <= s.max_area_fraction:
                continue
            height, width = self._shape
            if w >= s.max_span_fraction * width or h >= s.max_span_fraction * height:
                continue
            mask = labels == label
            # Edges just outside the region count too: a logo's outline is its border.
            grown = cv2.dilate(mask.astype(np.uint8), kernel) > 0
            density = float(edges[grown].mean())
            if density < s.min_edge_density:
                continue
            regions.append(
                LogoRegion(
                    mask=mask,
                    bbox=(x, y, w, h),
                    stable_fraction=round(float(fraction[mask].mean()), 4),
                    temporal_std=round(float(std[mask].mean()), 3),
                    edge_density=round(density, 4),
                )
            )
        return regions

    def presence(self, region: LogoRegion, gray: Frame, mode: NDArray[np.int64]) -> bool:
        """Whether the region shows its usual content in this sample."""
        level = self.quantise(gray)[region.mask]
        usual = mode[region.mask]
        matching = np.abs(level - usual) <= 1
        return float(matching.mean()) >= self._settings.presence_fraction
