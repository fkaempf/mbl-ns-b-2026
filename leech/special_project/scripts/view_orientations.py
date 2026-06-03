"""Quick look at annotated orientations: headings inside the arena and relative to food.

Usage:
    python scripts/view_orientations.py --config arena_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leecharena.config import Config  # noqa: E402
from leecharena.metrics import compute_from_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="arena_config.yaml")
    args = parser.parse_args()
    cfg = Config.load(args.config)

    metrics_path = cfg.annotations_path.with_name("metrics.csv")
    df = compute_from_store(cfg.annotations_path, metrics_path)
    df = df[df["leech_idx"] >= 0]
    if df.empty:
        print("No leech annotations yet.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: leech positions in the unit arena, arrows = heading.
    ax = axes[0]
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.6"))
    sub = df.dropna(subset=["radial_pos", "arena_angle_rad", "heading_rad"])
    px = sub["radial_pos"] * np.cos(sub["arena_angle_rad"])
    py = sub["radial_pos"] * np.sin(sub["arena_angle_rad"])
    ax.quiver(px, py, np.cos(sub["heading_rad"]), np.sin(sub["heading_rad"]),
              angles="xy", scale=20, width=0.004, color="tab:red")
    ax.set_title("leech position & heading in arena")
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    # Right: distribution of orientation relative to food (0 = facing food).
    ax = axes[1]
    rel = df["orient_rel_food_rad"].dropna()
    if len(rel):
        ax.hist(np.degrees(rel), bins=24, range=(-180, 180), color="tab:blue")
    ax.axvline(0, color="k", ls="--", label="facing food")
    ax.set_xlabel("orientation relative to food (deg)")
    ax.set_ylabel("count")
    ax.set_title(f"n={len(rel)} leech-frames")
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
