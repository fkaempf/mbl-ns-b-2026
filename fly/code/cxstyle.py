"""Repo-wide presentation style, tuned for PowerPoint / the pheromone paper.

- vector-first: figures are saved as SVG (see utils.save_fig), text stays editable
  (``svg.fonttype='none'``) and the heavy imaging/scatter layers are rasterised in place
  so the file stays small (``rasterize_heavy``);
- transparent background + white foreground, so every panel drops onto a dark slide;
- the pink / yellow / white scheme from the FC2 render (cx_fc2 video): pink = the FC2
  bump / imaging signal, yellow = the heading reference (a fly), white = structure / text.

utils.save_fig calls apply_style() (once, on import) and rasterize_heavy() for every plot,
so the whole repo inherits this. Scripts just use PINK / YELLOW / WHITE / PINK_CMAP.
"""

import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.colors import LinearSegmentedColormap

PINK = "#DE7FC7"        # FC2 render pink (the bump / imaging signal)
YELLOW = "#F4C542"      # fly yellow (the heading reference)
WHITE = "#FFFFFF"       # structure / text
PINK_CMAP = LinearSegmentedColormap.from_list("cx_pink", ["#08010A", "#7A2E6B", PINK, "#FFE9F6"])
PREVIEW_BG = "#000000"  # black preview / review background (the slides are black)


def apply_style(fg=WHITE):
    """Set the rcParams once: editable-text SVG, transparent canvas, white foreground."""
    plt.rcParams.update({
        "svg.fonttype": "none",
        "savefig.transparent": True,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": fg, "axes.labelcolor": fg, "axes.titlecolor": fg,
        "axes.edgecolor": fg, "xtick.color": fg, "ytick.color": fg,
        "legend.framealpha": 0.0, "legend.edgecolor": fg,
    })


def rasterize_heavy(fig, min_points=2000):
    """Rasterise the heavy artists (images, pcolormesh, big scatter / line collections)
    so a vector SVG stays small while text and axes remain vector."""
    for ax in fig.get_axes():
        for im in ax.get_images():
            im.set_rasterized(True)
        for line in ax.get_lines():                       # dense plot() traces
            try:
                if len(line.get_xdata()) >= min_points:
                    line.set_rasterized(True)
            except Exception:
                pass
        for coll in ax.collections:
            if isinstance(coll, QuadMesh):
                coll.set_rasterized(True)
                continue
            n = 0
            for getter in ("get_offsets", "get_paths", "get_segments"):
                try:
                    n = max(n, len(getattr(coll, getter)()))
                except Exception:
                    pass
            if n >= min_points:
                coll.set_rasterized(True)
    return fig
