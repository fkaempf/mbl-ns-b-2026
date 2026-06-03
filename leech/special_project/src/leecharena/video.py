"""Thin reader over cv2.VideoCapture with random frame access.

Frames are returned as RGB uint8 arrays (OpenCV decodes BGR). Random access uses
CAP_PROP_POS_FRAMES; with some H.264 builds seeking is approximate, but for sparse
hand-annotation that is acceptable — the exact frame index reached is what gets
recorded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VideoReader:
    def __init__(self, path: str | Path):
        import cv2

        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"could not open video: {self.path}")
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frame(self, index: int) -> np.ndarray:
        import cv2

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = self._cap.read()
        if not ok:
            raise IndexError(f"could not read frame {index} from {self.path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def time_s(self, index: int) -> float:
        return float(index) / self.fps if self.fps else float("nan")

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
