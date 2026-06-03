"""Eyeball a saved template in the canonical frame: midline, L/R/M, per aspect.

Usage:
    python scripts/view_template.py            # uses config.yaml, both aspects
    python scripts/view_template.py --aspect ventral
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leechtemplate.config import Config  # noqa: E402
from leechtemplate.template_io import load_template  # noqa: E402

SIDE_COLOR = {"L": "tab:blue", "R": "tab:red", "M": "tab:green"}


def plot_aspect(ax, df, aspect: str) -> None:
    ax.axvline(0.0, color="0.6", lw=1, ls="--", label="midline")
    for side, color in SIDE_COLOR.items():
        sub = df[df["side"] == side]
        if len(sub):
            ax.scatter(sub["x"], sub["y"], c=color, label=side, s=40)
            for _, r in sub.iterrows():
                ax.annotate(r["name"], (r["x"], r["y"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"{aspect} (n={len(df)})")
    ax.set_xlabel("medial-lateral  (left < 0 < right)")
    ax.set_ylabel("anterior-posterior  (posterior +)")
    ax.invert_yaxis()  # posterior down, matching figure orientation
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--aspect", choices=["ventral", "dorsal"], default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    aspects = [args.aspect] if args.aspect else config.aspects
    available = [(a, config.template_csv(a)) for a in aspects if config.template_csv(a).exists()]
    if not available:
        print("No template CSVs found yet for:", aspects)
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 6), squeeze=False)
    for ax, (aspect, path) in zip(axes[0], available):
        plot_aspect(ax, load_template(path), aspect)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
