#!/usr/bin/env python
"""Analyze how well leeches are oriented toward food over time, from annotations.csv.

Two SEPARATE measures per leech per frame (not combined into one flag):

  food_align_deg  — orientation toward food. Angle between the anterior-mid heading
                    line (direction mid -> anterior) and the direction from the mid
                    node to the food (mid -> food). 0 deg = aimed straight at the food.
                    NaN when the frame has no food annotated.

  body_align_deg  — body straightness ("are the 3 nodes aligned"). The turn angle at
  + is_straight     the mid node, between (posterior -> mid) and (mid -> anterior).
                    0 deg = perfectly straight. is_straight = body_align_deg < --straight-deg
                    (default 20), i.e. the leech is in a straightened, crawling posture.

Angles reuse leecharena.metrics.direction / wrap_to_pi (radians, math convention,
image-y negated), so they match the rest of the project.

Run:  .venv/bin/python scripts/food_orientation.py --config arena_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from leecharena.config import Config           # noqa: E402
from leecharena.metrics import direction, wrap_to_pi  # noqa: E402
from leecharena.store import load as load_store  # noqa: E402


def _pt(row, name):
    """(x, y) for node `name`, or None if either coord is missing."""
    x, y = row[f"{name}_x"], row[f"{name}_y"]
    if pd.isna(x) or pd.isna(y):
        return None
    return (float(x), float(y))


def _row_angles(row) -> dict:
    """Per-row food alignment and body straightness (degrees), NaN where undefined."""
    ant, mid, post = _pt(row, "ant"), _pt(row, "mid"), _pt(row, "post")
    fx, fy = row["food_x"], row["food_y"]
    food = None if pd.isna(fx) or pd.isna(fy) else (float(fx), float(fy))

    out = {"heading_am_rad": np.nan, "food_dir_rad": np.nan,
           "food_align_deg": np.nan, "body_align_deg": np.nan}
    if ant is not None and mid is not None:
        heading = direction(mid, ant)                    # mid -> anterior (heading line)
        out["heading_am_rad"] = heading
        if food is not None:
            food_dir = direction(mid, food)              # mid -> food
            out["food_dir_rad"] = food_dir
            out["food_align_deg"] = abs(np.degrees(wrap_to_pi(heading - food_dir)))
        if post is not None:
            seg_in = direction(post, mid)                # posterior -> mid
            seg_out = direction(mid, ant)                # mid -> anterior
            out["body_align_deg"] = abs(np.degrees(wrap_to_pi(seg_out - seg_in)))
    return out


def compute(df: pd.DataFrame, straight_deg: float = 20.0) -> pd.DataFrame:
    """Append food_align_deg, body_align_deg, is_straight (and the raw heading/food
    direction radians) to the real leech rows (leech_idx >= 0)."""
    df = df[df["leech_idx"] >= 0].copy()
    angles = pd.DataFrame([_row_angles(r) for _, r in df.iterrows()], index=df.index)
    df = pd.concat([df, angles], axis=1)
    df["is_straight"] = df["body_align_deg"] < straight_deg   # NaN -> False
    return df


def _summary(df: pd.DataFrame, straight_deg: float) -> None:
    n = len(df)
    food = df["food_align_deg"].notna()
    straight_def = df["body_align_deg"].notna()
    print(f"\nLeech rows (leech_idx>=0): {n}")
    print(f"Rows with food annotated: {int(food.sum())}  "
          f"({0 if n == 0 else 100 * food.mean():.0f}%)")
    if food.any():
        fa = df.loc[food, "food_align_deg"]
        print(f"food_align_deg toward food: mean {fa.mean():.1f}, median {fa.median():.1f}, "
              f"min {fa.min():.1f}, max {fa.max():.1f}")
    if straight_def.any():
        print(f"body straight (<{straight_deg:.0f} deg): "
              f"{int(df['is_straight'].sum())}/{int(straight_def.sum())} "
              f"({100 * df.loc[straight_def, 'is_straight'].mean():.0f}%)")
    print("\nper video:")
    for vid, g in df.groupby("video"):
        gf = g["food_align_deg"].dropna()
        print(f"  {vid}: {len(g)} rows, {len(gf)} with food, "
              f"mean food_align {gf.mean():.1f} deg, "
              f"{100 * g['is_straight'].mean():.0f}% straight")


def _plots(df: pd.DataFrame, out_png: Path, straight_deg: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    food = df[df["food_align_deg"].notna()]

    ax = axes[0]
    for vid, g in food.groupby("video"):
        ax.scatter(g["time_s"], g["food_align_deg"], s=14, alpha=0.7, label=str(vid))
    ax.axhline(straight_deg, color="grey", ls="--", lw=1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("food_align_deg (0 = at food)")
    ax.set_title("Orientation toward food over time")
    if not food.empty:
        ax.legend(fontsize=7)

    ax = axes[1]
    if not food.empty:
        ax.hist(food["food_align_deg"], bins=18, range=(0, 180), color="tab:blue")
    ax.axvline(straight_deg, color="grey", ls="--", lw=1)
    ax.set_xlabel("food_align_deg"); ax.set_ylabel("count")
    ax.set_title("Distribution of food alignment")

    ax = axes[2]
    frac = df.groupby("video")["is_straight"].mean()
    if not frac.empty:
        ax.bar(range(len(frac)), frac.values, color="tab:green")
        ax.set_xticks(range(len(frac)))
        ax.set_xticklabels([v[:14] for v in frac.index], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel(f"fraction straight (<{straight_deg:.0f} deg)")
    ax.set_title("Body-straight fraction per video")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"\nwrote {out_png}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="arena_config.yaml")
    parser.add_argument("--straight-deg", type=float, default=20.0,
                        help="max 3-node turn angle to count the body as straight")
    parser.add_argument("--no-show", action="store_true", help="don't open the plot window")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    df = compute(load_store(cfg.annotations_path), straight_deg=args.straight_deg)
    if df.empty:
        print("No leech annotations yet.")
        return

    out_csv = cfg.annotations_path.with_name("food_orientation.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    _summary(df, args.straight_deg)
    _plots(df, cfg.annotations_path.with_name("food_orientation.png"), args.straight_deg)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
