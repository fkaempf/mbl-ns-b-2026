"""Derive orientation metrics from raw annotations.

All angles are in radians, standard math convention (counter-clockwise, 0 along +x),
which means image y (which points down) is negated when forming direction vectors.
Rows with missing points yield NaN for the affected metrics rather than raising.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .store import load

METRIC_COLUMNS = [
    "heading_rad", "body_length_px", "bend_px",
    "food_dir_rad", "orient_rel_food_rad",
    "radial_pos", "arena_angle_rad",
]


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi] (so exactly 180 degrees comes out as +pi)."""
    w = (angle + np.pi) % (2 * np.pi) - np.pi
    return float(np.pi if w == -np.pi else w)


def direction(from_xy, to_xy) -> float:
    """Angle of the vector from -> to, math convention (image y negated)."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return float(np.arctan2(-dy, dx))


def perpendicular_distance(point, line_a, line_b) -> float:
    """Distance of `point` from the infinite line through line_a and line_b."""
    a = np.asarray(line_a, float)
    b = np.asarray(line_b, float)
    p = np.asarray(point, float)
    ab = b - a
    L = np.hypot(*ab)
    if L == 0:
        return float(np.hypot(*(p - a)))
    ap = p - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]  # 2-D scalar cross product
    return float(abs(cross) / L)


def _row_metrics(row: pd.Series) -> dict:
    ant = (row["ant_x"], row["ant_y"])
    post = (row["post_x"], row["post_y"])
    mid = (row["mid_x"], row["mid_y"])
    food = (row["food_x"], row["food_y"])
    center = (row["arena_cx"], row["arena_cy"])
    r = row["arena_r"]

    out = {c: np.nan for c in METRIC_COLUMNS}

    has_ap = not (pd.isna(ant[0]) or pd.isna(post[0]))
    has_mid = not pd.isna(mid[0])
    has_food = not pd.isna(food[0])
    has_arena = not (pd.isna(center[0]) or pd.isna(r))

    if has_ap:
        out["heading_rad"] = direction(post, ant)
        out["body_length_px"] = float(np.hypot(ant[0] - post[0], ant[1] - post[1]))
        if has_mid:
            out["bend_px"] = perpendicular_distance(mid, ant, post)

    ref = mid if has_mid else (post if has_ap else None)
    if has_food and ref is not None:
        out["food_dir_rad"] = direction(ref, food)
        if not np.isnan(out["heading_rad"]):
            out["orient_rel_food_rad"] = wrap_to_pi(out["heading_rad"] - out["food_dir_rad"])

    if has_arena and ref is not None and r:
        d = np.hypot(ref[0] - center[0], ref[1] - center[1])
        out["radial_pos"] = float(d / r)
        out["arena_angle_rad"] = direction(center, ref)

    return out


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with the METRIC_COLUMNS appended (NaN where points are missing)."""
    if df.empty:
        return df.assign(**{c: pd.Series(dtype=float) for c in METRIC_COLUMNS})
    metrics = pd.DataFrame([_row_metrics(r) for _, r in df.iterrows()], index=df.index)
    return pd.concat([df, metrics], axis=1)


def compute_from_store(store_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    df = compute_metrics(load(store_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
