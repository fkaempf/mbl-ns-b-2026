import numpy as np
import pandas as pd

from leecharena.metrics import (
    compute_metrics,
    direction,
    perpendicular_distance,
    wrap_to_pi,
)
from leecharena.store import COLUMNS


def _row(**kw):
    base = {c: np.nan for c in COLUMNS}
    base.update(video="c.mp4", frame=0, time_s=0.0, leech_idx=0)
    base.update(kw)
    return base


def test_wrap_to_pi():
    assert wrap_to_pi(np.pi) == np.pi
    assert wrap_to_pi(-np.pi) == np.pi
    assert wrap_to_pi(2 * np.pi) == 0


def test_direction_uses_math_convention():
    # In image coords y is down; a point directly "above" (smaller y) is +90 deg.
    assert direction((0, 0), (1, 0)) == 0.0                      # east
    assert direction((0, 0), (0, -1)) == np.pi / 2               # up -> north
    assert direction((0, 0), (0, 1)) == -np.pi / 2               # down -> south


def test_perpendicular_distance():
    assert perpendicular_distance((0, 5), (-10, 0), (10, 0)) == 5.0
    assert perpendicular_distance((3, 0), (0, 0), (10, 0)) == 0.0


def test_heading_and_orientation_relative_to_food():
    # Posterior at (0,0), anterior at (10,0) -> heading east (0 rad).
    # Food straight ahead of the middle -> facing food (rel == 0).
    row = _row(
        ant_x=10, ant_y=0, post_x=0, post_y=0, mid_x=5, mid_y=0,
        food_x=50, food_y=0, arena_cx=5, arena_cy=0, arena_r=10,
    )
    out = compute_metrics(pd.DataFrame([row])).iloc[0]
    assert out["heading_rad"] == 0.0
    assert out["body_length_px"] == 10.0
    assert out["food_dir_rad"] == 0.0
    assert out["orient_rel_food_rad"] == 0.0
    assert out["radial_pos"] == 0.0  # middle sits on the arena center


def test_orientation_relative_food_when_facing_away():
    # Heading east but food directly behind (west) -> rel == pi.
    row = _row(
        ant_x=10, ant_y=0, post_x=0, post_y=0, mid_x=5, mid_y=0,
        food_x=-50, food_y=0, arena_cx=5, arena_cy=0, arena_r=10,
    )
    out = compute_metrics(pd.DataFrame([row])).iloc[0]
    assert out["orient_rel_food_rad"] == np.pi


def test_radial_position_at_wall():
    # Middle one radius away from center -> radial_pos == 1.
    row = _row(
        ant_x=10, ant_y=0, post_x=0, post_y=0, mid_x=10, mid_y=0,
        arena_cx=0, arena_cy=0, arena_r=10,
    )
    out = compute_metrics(pd.DataFrame([row])).iloc[0]
    assert out["radial_pos"] == 1.0


def test_missing_points_yield_nan():
    out = compute_metrics(pd.DataFrame([_row(leech_idx=-1)])).iloc[0]
    assert np.isnan(out["heading_rad"])
    assert np.isnan(out["orient_rel_food_rad"])
