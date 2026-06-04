"""Tests for the food-orientation analysis script's angle logic."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
import pytest

import food_orientation as fo


def _row(ant, mid, post, food=(np.nan, np.nan)):
    return pd.Series({
        "ant_x": ant[0], "ant_y": ant[1], "mid_x": mid[0], "mid_y": mid[1],
        "post_x": post[0], "post_y": post[1], "food_x": food[0], "food_y": food[1],
    })


def test_straight_body_aimed_at_food():
    # post-mid-anterior colinear along +x, food further along +x
    m = fo._row_angles(_row(ant=(2, 0), mid=(1, 0), post=(0, 0), food=(3, 0)))
    assert m["food_align_deg"] == pytest.approx(0.0, abs=1e-6)
    assert m["body_align_deg"] == pytest.approx(0.0, abs=1e-6)


def test_straight_but_facing_away_from_food():
    # head points -x, food is +x -> straight body but ~180 deg off food
    m = fo._row_angles(_row(ant=(0, 0), mid=(1, 0), post=(2, 0), food=(3, 0)))
    assert m["food_align_deg"] == pytest.approx(180.0, abs=1e-6)
    assert m["body_align_deg"] == pytest.approx(0.0, abs=1e-6)


def test_curled_body_is_not_straight():
    # 90-degree bend at mid; no food
    m = fo._row_angles(_row(ant=(1, 1), mid=(1, 0), post=(0, 0)))
    assert m["body_align_deg"] == pytest.approx(90.0, abs=1e-6)
    assert np.isnan(m["food_align_deg"])     # food absent -> undefined


def test_missing_food_yields_nan_alignment():
    m = fo._row_angles(_row(ant=(2, 0), mid=(1, 0), post=(0, 0)))
    assert np.isnan(m["food_align_deg"])
    assert m["body_align_deg"] == pytest.approx(0.0, abs=1e-6)


def test_compute_filters_placeholders_and_flags_straight():
    df = pd.DataFrame([
        {"leech_idx": -1, "video": "v", "time_s": 0.0,
         "ant_x": np.nan, "ant_y": np.nan, "mid_x": np.nan, "mid_y": np.nan,
         "post_x": np.nan, "post_y": np.nan, "food_x": np.nan, "food_y": np.nan},
        {"leech_idx": 0, "video": "v", "time_s": 1.0,
         "ant_x": 2, "ant_y": 0, "mid_x": 1, "mid_y": 0,
         "post_x": 0, "post_y": 0, "food_x": 3, "food_y": 0},
        {"leech_idx": 1, "video": "v", "time_s": 1.0,
         "ant_x": 1, "ant_y": 1, "mid_x": 1, "mid_y": 0,
         "post_x": 0, "post_y": 0, "food_x": np.nan, "food_y": np.nan},
    ])
    out = fo.compute(df, straight_deg=20.0)
    assert len(out) == 2                       # placeholder dropped
    straight = out.set_index("leech_idx")["is_straight"]
    assert bool(straight.loc[0]) is True       # colinear
    assert bool(straight.loc[1]) is False      # 90-deg bend
