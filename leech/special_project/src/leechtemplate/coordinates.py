"""Pixel <-> canonical coordinate transforms.

Pure functions only: no I/O, no napari, no global state. This is the piece the
template's correctness hinges on, so it is isolated here and unit-tested.

Canonical frame (see the design doc for the full convention):
  - Origin (0, 0) is the *anterior* midline calibration landmark.
  - y increases toward the *posterior* midline landmark; y == 1 at that landmark.
  - x is perpendicular to the A-P axis, signed so the animal's RIGHT is positive.
  - Scale is isotropic: 1 canonical unit == anterior->posterior midline distance.
    Isotropic scaling preserves the figure's aspect ratio, so real left/right
    asymmetry is retained (we never mirror one side onto the other).

All pixel coordinates in this module are (x, y) == (column, row) with y pointing
DOWN, the usual image convention. napari uses (row, col) order; convert at the
boundary with `xy_from_rowcol` / `rowcol_from_xy` rather than mixing conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """Three clicked landmarks (in pixel x,y) defining a figure's canonical frame.

    Attributes:
        anterior_midline: (x, y) pixel coords of the anterior point on the midline.
        posterior_midline: (x, y) pixel coords of the posterior point on the midline.
        right_ref: (x, y) pixel coords of any point on the animal's right side.
            Only used to fix the SIGN of x; its magnitude does not matter.
    """

    anterior_midline: tuple[float, float]
    posterior_midline: tuple[float, float]
    right_ref: tuple[float, float]

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return (origin, ap_unit, ml_unit, ap_length_px).

        ap_unit points anterior->posterior; ml_unit points toward the animal's
        right; both are unit vectors in pixel space. ap_length_px is the scale.
        """
        origin = np.asarray(self.anterior_midline, dtype=float)
        post = np.asarray(self.posterior_midline, dtype=float)
        right = np.asarray(self.right_ref, dtype=float)

        ap = post - origin
        ap_length = float(np.hypot(*ap))
        if ap_length == 0.0:
            raise ValueError(
                "anterior and posterior midline landmarks coincide; "
                "cannot define an A-P axis"
            )
        ap_unit = ap / ap_length

        # Rotate ap_unit by +90 deg to get a perpendicular, then flip it so it
        # points toward the right-side reference.
        perp = np.array([-ap_unit[1], ap_unit[0]])
        if np.dot(right - origin, perp) < 0:
            perp = -perp
        # Guard against a right_ref placed exactly on the midline.
        if np.dot(right - origin, perp) == 0:
            raise ValueError(
                "right-side reference lies on the midline; it cannot disambiguate "
                "the left/right sign"
            )
        return origin, ap_unit, perp, ap_length


def pixel_to_canonical(points_xy: np.ndarray, calib: Calibration) -> np.ndarray:
    """Map pixel (x, y) points to canonical (x, y).

    Args:
        points_xy: array of shape (N, 2) of pixel coordinates, columns (x, y).
        calib: the figure's calibration.

    Returns:
        Array of shape (N, 2), columns (x_canonical, y_canonical).
    """
    pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
    origin, ap_unit, ml_unit, ap_length = calib.basis()
    rel = pts - origin
    x = rel @ ml_unit / ap_length
    y = rel @ ap_unit / ap_length
    return np.column_stack([x, y])


def canonical_to_pixel(points_canon: np.ndarray, calib: Calibration) -> np.ndarray:
    """Inverse of `pixel_to_canonical` (for redisplaying a saved template)."""
    pts = np.atleast_2d(np.asarray(points_canon, dtype=float))
    origin, ap_unit, ml_unit, ap_length = calib.basis()
    return origin + ap_length * (pts[:, [0]] * ml_unit + pts[:, [1]] * ap_unit)


def infer_side(x_canonical: float, midline_tol: float) -> str:
    """Classify a cell as 'L', 'R', or 'M' from its canonical x.

    Cells within `midline_tol` of x == 0 are midline ('M'). The annotation tool
    uses this only as a default; an explicit user choice always wins.
    """
    if abs(x_canonical) <= midline_tol:
        return "M"
    return "R" if x_canonical > 0 else "L"


def xy_from_rowcol(points_rowcol: np.ndarray) -> np.ndarray:
    """Convert napari (row, col) points to (x, y) == (col, row)."""
    pts = np.atleast_2d(np.asarray(points_rowcol, dtype=float))
    return pts[:, [1, 0]]


def rowcol_from_xy(points_xy: np.ndarray) -> np.ndarray:
    """Convert (x, y) == (col, row) points to napari (row, col)."""
    pts = np.atleast_2d(np.asarray(points_xy, dtype=float))
    return pts[:, [1, 0]]
