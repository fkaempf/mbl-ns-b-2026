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


def vrpos_csv(folder):
    """Return the Unity VR ``vrpos`` CSV for an experiment ``folder``.

    ``folder`` is resolved like :func:`fulltrack_csv` (absolute, relative to
    :data:`DATA_DIR`, or a direct path to the CSV).
    """
    if not os.path.isabs(folder):
        folder = os.path.join(DATA_DIR, folder)
    if os.path.isfile(folder):
        return folder
    matches = sorted(glob.glob(os.path.join(folder, "*unity_vr_driver_vrpos.csv")))
    if not matches:
        raise FileNotFoundError(f"no *_unity_vr_driver_vrpos.csv in {folder}")
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


def _moving_average(a, win):
    return a if win <= 1 else np.convolve(a, np.ones(win) / win, mode="same")


def _interp_circular(xq, xp, deg):
    """Interpolate a circular signal (degrees) onto ``xq`` via unit vectors, so it
    is correct across the +/-180 wrap (plain interpolation is not)."""
    rad = np.radians(deg)
    cos_i = np.interp(xq, xp, np.cos(rad))
    sin_i = np.interp(xq, xp, np.sin(rad))
    return np.degrees(np.arctan2(sin_i, cos_i))


def load_combined(folder, vr_heading="proper_rotation_z", min_speed_mm_s=5.0,
                  max_speed_mm_s=None, smooth_win=7):
    """Merge an experiment's fulltrack and vrpos logs onto one timeline.

    The two files share a unix clock, so the vrpos channels are interpolated onto
    the fulltrack timestamps (heading interpolated circularly). Walking speed is
    derived from the fulltrack lab-frame position; samples slower than
    ``min_speed_mm_s`` are dropped so the headings reflect real locomotion.

    Returns a DataFrame with ``t`` (s from start), ``fx, fy`` (mm) and ``fh`` (deg,
    wrapped) from fulltrack, ``vrx, vry, vrh`` interpolated from vrpos, and
    ``speed`` (mm/s). Pass ``min_speed_mm_s=0`` to keep every sample.
    """
    ft = pd.read_csv(fulltrack_csv(folder), usecols=[
        "timestamp", "integrated_position_lab_0", "integrated_position_lab_1",
        "integrated_heading_lab"])
    t_abs = ft.timestamp.to_numpy(float) / 1e9                  # unix seconds
    fx = -ft.integrated_position_lab_1.to_numpy() * BALL_RADIUS_MM
    fy = ft.integrated_position_lab_0.to_numpy() * BALL_RADIUS_MM
    fh = (np.degrees(ft.integrated_heading_lab.to_numpy()) + 180) % 360 - 180

    vr = pd.read_csv(vrpos_csv(folder), usecols=[
        "last_timestamp", "proper_position_x", "proper_position_y", vr_heading])
    vr = vr.sort_values("last_timestamp")
    vt = vr.last_timestamp.to_numpy()
    vrx = np.interp(t_abs, vt, vr.proper_position_x.to_numpy())
    vry = np.interp(t_abs, vt, vr.proper_position_y.to_numpy())
    vh = vr[vr_heading].to_numpy()
    if vr_heading == "last_heading":
        vh = np.degrees(vh)
    vrh = _interp_circular(t_abs, vt, (vh + 180) % 360 - 180)

    fs = 1.0 / np.median(np.diff(t_abs))
    speed = np.hypot(np.gradient(_moving_average(fx, smooth_win)) * fs,
                     np.gradient(_moving_average(fy, smooth_win)) * fs)

    df = pd.DataFrame(dict(t=t_abs - t_abs[0], t_unix=t_abs, fx=fx, fy=fy, fh=fh,
                           vrx=vrx, vry=vry, vrh=vrh, speed=speed))
    df = df[(t_abs >= vt[0]) & (t_abs <= vt[-1])]               # vrpos coverage only
    if min_speed_mm_s:
        df = df[df.speed >= min_speed_mm_s]
    if max_speed_mm_s:                                          # drop tracking-glitch jumps
        df = df[df.speed <= max_speed_mm_s]
    return df.reset_index(drop=True)


def experiment_label(folder):
    """Figure label: the folder's last two path parts (e.g. ``2026_06_24/rig1_experiment_06``)."""
    parts = os.path.normpath(folder).split(os.sep)
    return "/".join(parts[-2:]) if len(parts) >= 2 else folder


def experiment_id(folder):
    """Filename id: the experiment folder's basename (e.g. ``rig1_experiment_06``)."""
    return os.path.basename(os.path.normpath(folder))


def add_scalebar(ax, loc='lower right'):
    """Add a 10 cm scale bar to a trajectory axis (``loc`` = any legend-style corner)."""
    ax.add_artist(AnchoredSizeBar(
        ax.transData, 100, '10 cm', loc=loc,
        pad=0.6, color='black', frameon=False, size_vertical=1,
    ))


def emptiest_corner(ax, x, y, frac=0.33):
    """Pick the axis corner (legend ``loc`` string) with the fewest trace points, so
    a scale bar / legend placed there dodges the trajectory."""
    x, y = np.asarray(x), np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    rx, ry = frac * (x1 - x0), frac * (y1 - y0)
    corners = {'lower right': (x1, y0), 'lower left': (x0, y0),
               'upper right': (x1, y1), 'upper left': (x0, y1)}
    return min(corners, key=lambda loc:
               int(np.sum((np.abs(x - corners[loc][0]) < rx) &
                          (np.abs(y - corners[loc][1]) < ry))))


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


def save_fig(fig, filename, title=None, subdir=None, dpi=450):
    """Save ``fig`` into :data:`PLOTS_DIR` (or a ``subdir`` of it), created if needed.

    ``title`` is added as a figure suptitle so the saved image names its source data.
    ``subdir`` groups a set of standalone panels into their own folder. ``dpi`` raises
    the output resolution for detailed figures.
    """
    if title:
        fig.suptitle(title, fontsize=10)
    out = PLOTS_DIR if subdir is None else os.path.join(PLOTS_DIR, subdir)
    os.makedirs(out, exist_ok=True)
    # bbox_inches='tight' trims the surrounding whitespace so the saved image fits
    # its content (trajectories use equal aspect, which otherwise leaves wide margins).
    fig.savefig(os.path.join(out, filename), dpi=dpi, bbox_inches='tight')


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
