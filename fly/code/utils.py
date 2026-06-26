"""Shared helpers for the fly trajectory/heading analysis scripts."""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# FicTrac integrated_position is in radians of ball rotation;
# multiply by the ball radius (mm) to get lab-frame distance.
BALL_RADIUS_MM = 3.065

# Local data dir (<repo>/fly/data) and figure-output dir (<repo>/fly/plots).
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_CODE_DIR), "data")
PLOTS_DIR = os.path.join(os.path.dirname(_CODE_DIR), "plots")


def fulltrack_csv(folder):
    """Return the FicTrac fulltrack CSV for an experiment ``folder``.

    ``folder`` may be absolute or relative to :data:`DATA_DIR`; a direct path to a
    CSV file is also accepted and returned unchanged.
    """
    if not os.path.isabs(folder):
        folder = os.path.join(DATA_DIR, folder)
    if os.path.isfile(folder):
        return folder
    matches = sorted(glob.glob(os.path.join(folder, "*fictrac_node_fulltrack.csv")))
    if not matches:
        raise FileNotFoundError(f"no *_fictrac_node_fulltrack.csv in {folder}")
    return matches[0]


def load_trajectory(folder):
    """Load an experiment's fulltrack CSV and derive lab-frame position and time.

    Returns ``(data, x, y, time_s)``: the raw DataFrame, lab-frame x/y in mm, and
    elapsed time in seconds (starting at 0).
    """
    data = pd.read_csv(fulltrack_csv(folder))
    x = -np.asarray(data.integrated_position_lab_1) * BALL_RADIUS_MM
    y = np.asarray(data.integrated_position_lab_0) * BALL_RADIUS_MM
    time_unix_nano = np.asarray(data.timestamp)
    time_s = (time_unix_nano - time_unix_nano[0]) / 1e9
    return data, x, y, time_s


def experiment_label(folder):
    """Figure label: the folder's last two path parts (e.g. ``2026_06_24/rig1_experiment_06``)."""
    parts = os.path.normpath(folder).split(os.sep)
    return "/".join(parts[-2:]) if len(parts) >= 2 else folder


def experiment_id(folder):
    """Filename id: the experiment folder's basename (e.g. ``rig1_experiment_06``)."""
    return os.path.basename(os.path.normpath(folder))


def add_scalebar(ax):
    """Add a 10 cm scale bar to the lower-right of a trajectory axis."""
    ax.add_artist(AnchoredSizeBar(
        ax.transData, 100, '10 cm', loc='lower right',
        bbox_to_anchor=(0.95, 0.05), bbox_transform=ax.transAxes,
        pad=0.5, color='black', frameon=False, size_vertical=1,
    ))


def draw_trajectory(ax, x, y, *, mark_start=False, scalebar=True, title=None):
    """Draw an x/y trajectory: equal aspect, no axes, optional start marker + scale bar."""
    if mark_start:
        ax.scatter(x[0], y[0], c='#0FFF50', zorder=3)
    ax.plot(x, y, zorder=2)
    ax.axis('equal')
    ax.axis('off')
    if scalebar:
        add_scalebar(ax)
    if title:
        ax.set_title(title, fontsize=9)


def save_fig(fig, filename, title=None, subdir=None):
    """Save ``fig`` into :data:`PLOTS_DIR` (or a ``subdir`` of it), created if needed.

    ``title`` is added as a figure suptitle so the saved image names its source data.
    ``subdir`` groups a set of standalone panels into their own folder.
    """
    if title:
        fig.suptitle(title, fontsize=10)
    out = PLOTS_DIR if subdir is None else os.path.join(PLOTS_DIR, subdir)
    os.makedirs(out, exist_ok=True)
    # bbox_inches='tight' trims the surrounding whitespace so the saved image fits
    # its content (trajectories use equal aspect, which otherwise leaves wide margins).
    fig.savefig(os.path.join(out, filename), dpi=150, bbox_inches='tight')


def save_panel(subdir, filename, draw, *, figsize=(6.5, 5), title=None, subplot_kw=None):
    """Build a standalone single-axis figure, fill it via ``draw(ax)``, and save it.

    Each panel becomes its own image under ``PLOTS_DIR/subdir``. ``title`` (e.g. the
    experiment label) is the figure suptitle for provenance; ``draw`` sets the
    panel's own axis title. ``subplot_kw`` is passed to ``plt.subplots`` (e.g.
    ``{"projection": "polar"}``).
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw or {})
    draw(ax)
    save_fig(fig, filename, title=title, subdir=subdir)
    plt.close(fig)
