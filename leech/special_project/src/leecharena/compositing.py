"""Collapse a clip's frames into one composite image for dish placement.

A circular dish is static across the recording while leeches move, so a per-pixel
average (or median) over time keeps the dish rim sharp and washes the animals out
into the background. That composite is a far better target for circle detection or
hand-placement than any single frame.

``mean`` streams the video and accumulates a running sum (memory-light, exact over
all frames). ``median`` must hold the sampled frames in memory but removes moving
objects completely rather than leaving faint trails; pair it with ``sample`` on long
clips. OpenCV is imported lazily so the package imports without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _sample_indices(n_frames: int, sample: int | None) -> list[int] | None:
    """Frame indices to use: None means "every frame, read sequentially"; otherwise
    ``sample`` indices spread evenly across the clip."""
    if sample is None or sample >= n_frames:
        return None
    return [int(round(i)) for i in np.linspace(0, n_frames - 1, sample)]


def _iter_frames(cap, indices):
    """Yield RGB uint8 frames. indices None -> sequential read of the whole clip."""
    import cv2

    if indices is None:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if ok:
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def composite(video_path: str | Path, method: str = "mean", sample: int | None = None) -> np.ndarray:
    """Composite a clip into one RGB uint8 image.

    method: "mean" (per-pixel average) or "median" (per-pixel median).
    sample: number of frames to use, spread evenly across the clip; None uses every
            frame. Use it to bound work/memory on long clips.
    """
    import cv2

    if method not in ("mean", "median"):
        raise ValueError(f"unknown method {method!r}; use 'mean' or 'median'")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_indices(n_frames, sample)

    try:
        if method == "median":
            frames = list(_iter_frames(cap, indices))
            if not frames:
                raise RuntimeError(f"no frames read from {video_path}")
            stack = np.stack(frames)
            return np.round(np.median(stack, axis=0)).clip(0, 255).astype(np.uint8)

        acc = None
        count = 0
        for rgb in _iter_frames(cap, indices):
            f = rgb.astype(np.float64)
            acc = f if acc is None else acc + f
            count += 1
        if count == 0:
            raise RuntimeError(f"no frames read from {video_path}")
        return np.round(acc / count).clip(0, 255).astype(np.uint8)
    finally:
        cap.release()
