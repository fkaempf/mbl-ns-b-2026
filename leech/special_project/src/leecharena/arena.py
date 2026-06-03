"""Arena circle: pure geometry helpers plus OpenCV detection.

The pure helpers convert between a circle (cx, cy, r) in pixel x,y and the 4-corner
bounding box napari uses for an editable ellipse Shape. detect_circle wraps
cv2.HoughCircles and is the only part needing OpenCV.
"""

from __future__ import annotations

import numpy as np


def circle_to_corners(cx: float, cy: float, r: float) -> np.ndarray:
    """Circle -> (4, 2) ellipse-bbox corners in napari (row, col) order."""
    x0, x1 = cx - r, cx + r
    y0, y1 = cy - r, cy + r
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def corners_to_circle(corners: np.ndarray) -> tuple[float, float, float]:
    """(N, 2) ellipse-bbox corners in (row, col) -> (cx, cy, r) in pixel x,y.

    Radius is the mean of the bbox half-width and half-height, so a user who drags
    the ellipse into a slight oval still yields a sensible circle.
    """
    corners = np.asarray(corners, dtype=float)
    rows, cols = corners[:, 0], corners[:, 1]
    cx = (cols.min() + cols.max()) / 2.0
    cy = (rows.min() + rows.max()) / 2.0
    r = ((cols.max() - cols.min()) + (rows.max() - rows.min())) / 4.0
    return float(cx), float(cy), float(r)


def detect_circle(
    gray: np.ndarray,
    dp: float = 1.2,
    min_dist: float = 1000.0,
    param1: float = 100.0,
    param2: float = 30.0,
    min_radius: int = 0,
    max_radius: int = 0,
) -> tuple[float, float, float] | None:
    """Detect the most prominent circle in a grayscale image.

    Returns (cx, cy, r) in pixel x,y, or None if nothing is found. min_dist is large
    by default because a single-dish clip should contain exactly one arena, so we want
    the single strongest circle. HoughCircles returns circles ranked by accumulator
    strength; we take the first.
    """
    import cv2

    img = gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if circles is None:
        return None
    cx, cy, r = circles[0][0]
    return float(cx), float(cy), float(r)
