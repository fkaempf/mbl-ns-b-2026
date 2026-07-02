"""Unify the FC2 fan-shaped-body imaging with the VR + barrier task, per fly.

Each imaging fly's ``*_final.pkl`` already fuses behaviour (FicTrac ``fulltrack`` +
Unity ``vr``) with the 16 fan-shaped-body column ROIs (``c1_roi_1..16``) on a common
~100 Hz unix-time base. This layers on the task structure the behaviour pipeline uses,
so the imaging sits in exactly the same frame as ``load_combined`` + the barrier
helpers:

  * ``speed`` / ``mv_dir``  - VR walking speed (mm/s) and travel direction (deg),
  * ``wall_on`` + ``wall_cx/cy/rot/w`` + ``wall_dist`` + ``wall_bearing`` - the barrier
    (RectMaze) presentation active at each frame and where it sits relative to the fly,
  * ``laser_on`` - the aversive-laser (shock) intervals,
  * ``bump_phase`` / ``bump_amp`` - the FC2 population bump: PVA phase (deg, column
    space 0..360) and resultant amplitude over the 16 columns from dF/F.

The barrier/laser times share the pkl's unix-second clock (vrcmd timestamps are already
seconds; ``laser_intervals`` converts the ns light-sugar log), so events map straight on.

Writes one unified pickle per fly (``data/imaging0630/<fly>_unified.pkl``) plus a
concatenated ``data/imaging0630/imaging0630_unified.pkl`` with a ``fly`` column, and a
validation figure (``plots/imaging/fc2_overview.png``).

Run from this directory:  python imaging_unify.py
"""

import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as st

import bounces as bo
import barrier_traces as bt
from utils import DATA_DIR, save_fig, add_scalebar
from fc2_utils import get_pva, calculate_normalized_signal, remove_wrap_lines
from cxstyle import PINK, YELLOW, WHITE, PINK_CMAP

TRAJ_CMAP = LinearSegmentedColormap.from_list("cx_traj", ["#ffffff", PINK])   # trajectory over time
LASER_MX, LASER_MY = 0.05, 2.5   # laser-zone half-margins (config), for drawing the barrier


def add_wall(ax, cx, cy, rot, w, th):
    """Draw the barrier: a grey wall inside its red laser (hurt) zone at (cx,cy).

    The vrcmd rotation_z is in the (y-flipped) VR heading frame, so in the vr_x/vr_y
    position frame the wall's drawing angle is -rot. Verified empirically: the positions
    of laser-shocked frames lie along -rot (PCA, median 1.4 deg over 30 walls, 90% within
    15 deg), whereas +rot is ~34 deg off."""
    ang = -rot
    hw, hth = w + 2 * LASER_MX, th + 2 * LASER_MY
    ax.add_patch(Rectangle((cx - hw / 2, cy - hth / 2), hw, hth, angle=ang,
                           rotation_point="center", facecolor="red", alpha=0.3,
                           edgecolor="red", lw=0.6, zorder=2))
    ax.add_patch(Rectangle((cx - w / 2, cy - th / 2), w, th, angle=ang,
                           rotation_point="center", facecolor="0.5",
                           edgecolor="white", lw=0.8, zorder=3))

IMG_DIR = "imaging0630"
FLIES = ["Fly1", "Fly2_001", "Fly5_002", "Fly6_002"]
NCOL = 16
MOVE_CUTOFF = 5.0        # mm/s; "moving" frames for the bump-vs-heading check
SMOOTH_N = 121           # circular low-pass window (samples, ~1 s at ~100 Hz) for readability
DS_PLOT = 4              # plot every DS_PLOT-th sample to declutter the dense traces
R_MIN = 0.25             # only draw the bump where the PVA amplitude is at least this
PAD_S = 8.0              # seconds of context shown before wall-on / after wall-off per trial
ROI_COLS = [f"c1_roi_{i}" for i in range(1, NCOL + 1)]


def wrap180(d):
    return (d + 180) % 360 - 180


def eid_of(folder):
    f = glob.glob(os.path.join(DATA_DIR, folder, "*vrcmd.csv"))
    return os.path.basename(f[0]).split("_unity")[0] if f else folder


def sun_color_of(folder):
    cf = glob.glob(os.path.join(DATA_DIR, folder, "*config.yaml"))
    if not cf:
        return None
    m = re.search(r"sun_color:\s*\(([^)]+)\)", open(cf[0]).read())
    return tuple(float(v) for v in m.group(1).split(",")[:3]) if m else None


def circ_corr(a_deg, b_deg):
    """Jammalamadaka circular correlation between two angle series (deg)."""
    a, b = np.radians(a_deg), np.radians(b_deg)
    a0 = np.angle(np.exp(1j * a).mean())
    b0 = np.angle(np.exp(1j * b).mean())
    sa, sb = np.sin(a - a0), np.sin(b - b0)
    return float((sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum()))


def windowed_track(bump_deg, head_deg, mov, win=300, step=150):
    """FC2 encodes a *goal*: the bump tracks heading within a bout but at an offset that
    drifts and resets between bouts, so a single session-wide circular correlation washes
    out to ~0. This measures tracking the right way - the circular correlation within
    short (~win-frame) windows of moving frames - and returns the array of per-window
    values. Report its p90 / fraction-strong, not the global value."""
    out = []
    for i in range(0, len(bump_deg) - win, step):
        s = slice(i, i + win)
        m = mov[s] & np.isfinite(bump_deg[s])
        if m.sum() > 50:
            out.append(circ_corr(bump_deg[s][m], head_deg[s][m]))
    return np.array(out) if out else np.array([np.nan])


def unify(fly):
    folder = f"{IMG_DIR}/{fly}"
    # The fused imaging+behaviour DataFrame. Most flies name it "*_final.pkl", but some
    # (e.g. Fly1's fly1_06_30_2026.pkl) don't - so take the folder's pkl that is neither
    # an imaging intermediate (mean_pixel_values) nor our own _unified output.
    cands = [p for p in glob.glob(os.path.join(DATA_DIR, folder, "*.pkl"))
             if "mean_pixel_values" not in os.path.basename(p) and not p.endswith("_unified.pkl")]
    pkl = cands[0]
    # Trusted first-party data: this pkl is our own imaging pipeline's output (a plain
    # pandas DataFrame) copied from the lab volume, not an untrusted source.
    df = pickle.load(open(pkl, "rb")).reset_index(drop=True)
    t = df["time"].to_numpy(float)

    # --- derived VR speed / travel direction ---
    dt = np.gradient(t)
    dt[dt <= 0] = np.nan
    vx = np.gradient(df["vr_x"].to_numpy(float)) / dt
    vy = np.gradient(df["vr_y"].to_numpy(float)) / dt
    df["speed"] = np.hypot(vx, vy)
    # vr_heading uses a flipped angular convention (0 = +y, clockwise) vs the position
    # atan2 frame: verified because a wall spawned dead ahead has world-direction
    # atan2(dy,dx) == 90 - vr_heading (constant to ~2 deg). Put the travel direction in
    # that same convention so mv_dir is directly comparable to vr_heading.
    df["mv_dir"] = (90 - np.degrees(np.arctan2(vy, vx))) % 360

    # --- FC2 bump: maxmin-normalise each column, then lab-standard PVA on the matrix ---
    Fn = np.column_stack([calculate_normalized_signal(df[c], method="maxmin") for c in ROI_COLS])
    theta, r = get_pva(Fn)
    df["bump_phase"], df["bump_amp"] = theta, r

    # --- barrier (wall) presentations + geometry relative to the fly ---
    x, y, h = df["vr_x"].to_numpy(), df["vr_y"].to_numpy(), df["vr_heading"].to_numpy()
    for c in ["wall_on"]:
        df[c] = False
    for c in ["wall_cx", "wall_cy", "wall_rot", "wall_w", "wall_dist", "wall_bearing"]:
        df[c] = np.nan
    walls = bo.wall_trials(folder)
    for ton, toff, cx, cy, rot, w, th in walls:
        m = (t >= ton) & (t <= toff)
        if not m.any():
            continue
        df.loc[m, "wall_on"] = True
        df.loc[m, "wall_cx"], df.loc[m, "wall_cy"] = cx, cy
        df.loc[m, "wall_rot"], df.loc[m, "wall_w"] = rot, w
        df.loc[m, "wall_dist"] = np.hypot(cx - x[m], cy - y[m])
        # egocentric bearing of the wall vs the fly's facing: 0 = wall dead ahead,
        # +/-180 = behind. The +h-90 accounts for vr_heading's flipped convention (see mv_dir).
        df.loc[m, "wall_bearing"] = wrap180(np.degrees(np.arctan2(cy - y[m], cx - x[m])) + h[m] - 90)

    # --- aversive laser (shock) intervals ---
    df["laser_on"] = False
    intervals = bt.laser_intervals(folder)
    for a, b in intervals:
        df.loc[(t >= a) & (t <= b), "laser_on"] = True

    # --- metadata ---
    df.attrs.update(dict(fly=fly, eid=eid_of(folder), sun_color=sun_color_of(folder),
                         barrier_width=bo.experiment_barrier_width(folder),
                         n_walls=len(walls), n_shock_intervals=len(intervals)))
    out = os.path.join(DATA_DIR, IMG_DIR, f"{fly}_unified.pkl")
    df.to_pickle(out)

    mov = df["speed"].to_numpy() >= MOVE_CUTOFF
    bp, hd = df["bump_phase"].to_numpy(), df["vr_heading"].to_numpy()
    has_bump = bool(np.isfinite(bp).any())
    wc = windowed_track(bp, hd, mov)
    p90, strong = (np.nanpercentile(wc, 90), np.nanmean(wc > 0.5)) if has_bump else (np.nan, np.nan)
    print(f"  {fly}: {len(df)} frames, {df.attrs['n_walls']} walls, "
          f"{int(df['wall_on'].sum())} wall-on / {int(df['laser_on'].sum())} laser-on frames, "
          f"speed p50/p95={np.nanpercentile(df.speed,50):.1f}/{np.nanpercentile(df.speed,95):.1f} mm/s, "
          f"bump={'yes' if has_bump else 'NO ROI DATA'} windowed-circ-r p90={p90:+.2f} frac>0.5={strong:.2f} -> {out}")
    return df, (p90, strong)


def validate_fig(frames, filt=False):
    """All experiments, bump vs time: one row per fly, heading (fly) + offset-aligned
    bump over the full session, with wall on/off triangles and shocks. ``filt=True``
    saves the readability alternative (wider low-pass + decimation) as fc2_overview_filtered."""
    nn, ds = (SMOOTH_N, DS_PLOT) if filt else (25, 2)
    fig, axes = plt.subplots(len(frames), 1, figsize=(16, 2.6 * len(frames)),
                             squeeze=False, constrained_layout=True)
    for ri, (fly, df, stats) in enumerate(frames):
        ax = axes[ri, 0]
        t = df["time"].to_numpy(); t = t - t[0]
        has_bump = np.isfinite(df[ROI_COLS].to_numpy(float)).any()
        won = df["wall_on"].to_numpy()
        spawn = np.where(np.diff(won.astype(int)) == 1)[0] + 1     # wall-on rising edges
        if won[0]:
            spawn = np.r_[0, spawn]
        ons = np.where(np.diff(df["laser_on"].to_numpy().astype(int)) == 1)[0]
        hw, pa, r = aligned_bump(df, nn)
        idx = r > R_MIN
        ax.fill_between(t, -180, 180, where=won, color=WHITE, alpha=0.08, step="mid")
        plot_wrapped(ax, t[::ds], np.where(idx, hw, np.nan)[::ds], color=YELLOW, lw=1.2, alpha=0.9,
                     label="heading (fly)")
        if has_bump:
            plot_wrapped(ax, t[::ds], np.where(idx, pa, np.nan)[::ds], color=PINK, lw=1.5, alpha=0.95,
                         label="FC2 bump (r>0.25)")
        mark_walls(ax, t, won, -180)
        ax.plot(t[ons], np.full(len(ons), 180), "|", color=WHITE, ms=6, alpha=0.7, label="shock")
        ax.set_xlim(0, t[-1]); ax.set_ylim(-180, 180); ax.set_ylabel("deg")
        ax.set_title(f"{fly} ({df.attrs.get('eid', '')}, {len(spawn)} walls)", fontsize=9)
    axes[-1, 0].set_xlabel("time in session (s)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=len(lab), fontsize=8)
    tag = f"  [filtered ~{SMOOTH_N} samples]" if filt else ""
    suffix = "_filtered" if filt else ""
    fig.suptitle(f"FC2 bump vs time across experiments{tag}", fontsize=12)
    save_fig(fig, f"fc2_overview{suffix}.png", subdir="imaging")
    plt.close(fig)


def csmooth(deg, n=25):
    """Wrap-safe circular moving average of an angle series (deg): average as unit
    vectors so it doesn't glitch across the 0/360 seam."""
    c = np.convolve(np.cos(np.radians(deg)), np.ones(n) / n, mode="same")
    s = np.convolve(np.sin(np.radians(deg)), np.ones(n) / n, mode="same")
    return np.degrees(np.arctan2(s, c)) % 360


def plot_wrapped(ax, t, deg, wrapthr=180, **kw):
    """Plot an angle/position trace as a line, cutting it at wraps with the lab-standard
    remove_wrap_lines (a jump > wrapthr). wrapthr is 180 for degrees, 8 for a 16-column
    axis, etc."""
    ax.plot(t, remove_wrap_lines(np.asarray(deg, float).copy(), trsh=wrapthr), **kw)


def mark_walls(ax, t, won, y):
    """Wall events as triangles just under the data: filled triangle = wall ON (spawn),
    open triangle = wall OFF (destroy)."""
    w = won.astype(int)
    on = np.where(np.diff(w) == 1)[0] + 1
    if w[0]:
        on = np.r_[0, on]
    off = np.where(np.diff(w) == -1)[0] + 1
    ax.plot(t[on], np.full(len(on), y), "^", ms=9, color=WHITE, ls="none",
            clip_on=False, label="wall on")
    ax.plot(t[off], np.full(len(off), y), "^", ms=9, mfc="none", mec=WHITE, mew=1.4,
            ls="none", clip_on=False, label="wall off")


def norm_matrix(df):
    """maxmin-normalised column matrix (T x 16) for the FC2 raster."""
    return np.column_stack([calculate_normalized_signal(df[c], method="maxmin") for c in ROI_COLS])


def aligned_bump(df, n=25):
    """Bump phase offset-aligned to the fulltrack heading (circular-mean offset), both
    circularly smoothed (window ``n`` samples) and wrapped to [-180,180], plus the PVA
    amplitude r. Mirrors the lab plotting workflow (heading = fly, aligned bump = pink)."""
    phase = df["bump_phase"].to_numpy()
    r = df["bump_amp"].to_numpy()
    heading = (df["fulltrack_heading"].to_numpy() * 180 / np.pi) % 360
    offset = st.circmean((heading - phase) % 360, low=0, high=360, nan_policy="omit")
    pa = (csmooth(phase, n) + offset) % 360
    pa = (pa + 180) % 360 - 180
    hw = (csmooth(heading, n) + 180) % 360 - 180
    return hw, pa, r


def single_fig(fly, filt=False):
    """One experiment, bump vs time: FC2 column raster (maxmin-normalised) on top, and
    the heading (fly, yellow) + offset-aligned bump (pink, where r>R_MIN) below, with
    wall on/off triangles and shocks. ``filt=True`` saves a readability alternative:
    a wider (~1 s) circular low-pass on the traces plus plot-decimation, as
    ``fc2_<fly>_filtered``. Reads the already-built _unified.pkl."""
    # Trusted first-party data: this _unified.pkl was written by this same script.
    df = pd.read_pickle(os.path.join(DATA_DIR, IMG_DIR, f"{fly}_unified.pkl"))
    t = df["time"].to_numpy(); t = t - t[0]
    won = df["wall_on"].to_numpy()
    spawn = np.where(np.diff(won.astype(int)) == 1)[0] + 1
    if won[0]:
        spawn = np.r_[0, spawn]
    ons = np.where(np.diff(df["laser_on"].to_numpy().astype(int)) == 1)[0]
    has_bump = np.isfinite(df[ROI_COLS].to_numpy(float)).any()
    data_mat = norm_matrix(df).T                            # (16 x T) maxmin columns
    n, ds = (SMOOTH_N, DS_PLOT) if filt else (25, 2)
    hw, pa, r = aligned_bump(df, n)                         # heading + offset-aligned bump, [-180,180]
    idx = r > R_MIN                                         # show the bump only where it is strong

    fig = plt.figure(figsize=(26, 2.6), constrained_layout=True)   # really wide, low banner
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])
    a0 = fig.add_subplot(gs[0, 0])                          # bump raster vs time (top)
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)               # bump phase + heading vs time (bottom)

    if has_bump:
        a0.imshow(data_mat, aspect="auto", origin="lower", extent=[0, t[-1], 0.5, 16.5],
                  cmap=PINK_CMAP, vmin=0, vmax=0.9, interpolation="nearest", rasterized=True)
        colphase = csmooth(df["bump_phase"].to_numpy(), n) / 360 * NCOL + 0.5   # extracted phase
        plot_wrapped(a0, t[::ds], colphase[::ds], wrapthr=NCOL / 2, color="#37e0d0",
                     lw=0.9, alpha=0.8, label="extracted phase")
    mark_walls(a0, t, won, 0.5)
    a0.set_ylabel("FSB column (ROI 1-16)"); a0.set_ylim(0.5, 16.5)
    a0.set_title("FC2 columns (maxmin) - the bump", fontsize=10)

    a1.fill_between(t, -180, 180, where=won, color=WHITE, alpha=0.08, step="mid")
    hplot = np.where(idx, hw, np.nan)[::ds]
    pplot = np.where(idx, pa, np.nan)[::ds]
    plot_wrapped(a1, t[::ds], hplot, color=YELLOW, lw=1.4, alpha=0.9, label="heading (fly)")
    if has_bump:
        plot_wrapped(a1, t[::ds], pplot, color=PINK, lw=1.7, alpha=0.95, label="FC2 bump (r>0.25)")
    mark_walls(a1, t, won, -180)
    a1.plot(t[ons], np.full(len(ons), 180), "|", color=WHITE, ms=7, alpha=0.7, label="shock")
    a1.set_xlim(0, t[-1]); a1.set_ylim(-180, 180)
    a1.set_xlabel("time in session (s)"); a1.set_ylabel("heading + orientation (deg)")
    h0, l0 = a0.get_legend_handles_labels()
    h1, l1 = a1.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h0 + h1, l0 + l1):
        seen.setdefault(ll, hh)
    fig.legend(seen.values(), seen.keys(), loc="outside lower center", ncol=len(seen), fontsize=8)

    eid = df.attrs.get("eid", "")
    tag = f"  [filtered ~{SMOOTH_N} samples]" if filt else ""
    suffix = "_filtered" if filt else ""
    fig.suptitle(f"FC2 fan-shaped-body imaging + barrier task - {fly}  ({eid}, {len(spawn)} walls){tag}",
                 fontsize=13)
    save_fig(fig, f"fc2_{fly}{suffix}.png", subdir="imaging")
    plt.close(fig)
    print(f"  saved plots/imaging/fc2_{fly}{suffix}.png")


def trials_from_wall(won):
    """(on, off) index pairs, one per wall presentation, from the wall_on rising/falling
    edges (handling a session that starts or ends with a wall on)."""
    w = won.astype(int)
    e = np.diff(w)
    ons = list(np.where(e == 1)[0] + 1)
    offs = list(np.where(e == -1)[0] + 1)
    if w[0]:
        ons = [0] + ons
    if w[-1]:
        offs = offs + [len(w) - 1]
    return list(zip(ons, offs))


def trial_figs(fly, filt=False, ticks=False):
    """One file PER wall-on -> wall-off trial (like the bounce gallery). LEFT = the planar
    trajectory during the trial (fly path coloured white->pink over time, the barrier drawn
    as a grey wall in its red laser zone, shocks ringed, start dot). RIGHT = the
    corresponding neural activity: the FC2 bump raster (maxmin, extracted phase overlaid) on
    top and heading (fly) + bump orientation below. PAD_S s of context each side.

    ``ticks=True`` also marks the trajectory with a cross every 10 s (labelled), so you can
    read time / speed along the path, and saves into the trials_ticks subfolder."""
    sub = "imaging/trials_ticks" if ticks else "imaging/trials"
    # Trusted first-party data: this _unified.pkl was written by this same script.
    df = pd.read_pickle(os.path.join(DATA_DIR, IMG_DIR, f"{fly}_unified.pkl"))
    t = df["time"].to_numpy()
    xall, yall = df["vr_x"].to_numpy(), df["vr_y"].to_numpy()
    las = df["laser_on"].to_numpy()
    Wn = norm_matrix(df)
    n, ds = (SMOOTH_N, DS_PLOT) if filt else (25, 2)
    hw, pa, r = aligned_bump(df, n)
    colphase = csmooth(df["bump_phase"].to_numpy(), n) / 360 * NCOL + 0.5   # extracted phase -> column
    idx = r > R_MIN
    walls = bo.wall_trials(f"{IMG_DIR}/{fly}")
    suffix = "_filtered" if filt else ""
    eid = df.attrs.get("eid", "")
    for k, (ton, toff, cx, cy, rot, w, th) in enumerate(walls, 1):
        i0 = int(np.searchsorted(t, ton - PAD_S))
        i1 = int(min(np.searchsorted(t, toff + PAD_S), len(t) - 1))
        sl = slice(i0, i1)
        tr = t[sl] - ton                                     # 0 at wall on
        dur = toff - ton

        fig = plt.figure(figsize=(16, 4.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.7], height_ratios=[1.25, 1])
        axp = fig.add_subplot(gs[:, 0])                      # LEFT: planar trajectory
        axr = fig.add_subplot(gs[0, 1])                      # RIGHT top: bump raster
        axt = fig.add_subplot(gs[1, 1], sharex=axr)          # RIGHT bottom: heading + orientation

        # --- LEFT: planar trajectory coloured over time, with the barrier ---
        xs, ys = xall[sl], yall[sl]
        pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=TRAJ_CMAP, array=tr[:-1], lw=1.4, rasterized=True)
        axp.add_collection(lc)
        add_wall(axp, cx, cy, rot, w, th)
        shi = np.where(np.diff(las[sl].astype(int)) == 1)[0]
        if len(shi):
            axp.scatter(xs[shi], ys[shi], facecolor="none", edgecolor="red", s=45, lw=1.3, zorder=6)
        axp.scatter(xs[0], ys[0], color=WHITE, s=25, zorder=6)          # start
        if ticks:                                                        # a cross every 10 s
            tk = np.arange(np.ceil(tr[0] / 10) * 10, tr[-1] + 1e-6, 10)
            tkx, tky = np.interp(tk, tr, xs), np.interp(tk, tr, ys)
            axp.scatter(tkx, tky, marker="+", c="#37e0d0", s=55, lw=1.3, zorder=7)
            for s10, xx, yy in zip(tk, tkx, tky):
                axp.annotate(f"{s10:.0f}", (xx, yy), color="#37e0d0", fontsize=6,
                             xytext=(3, 3), textcoords="offset points", zorder=7)
        axp.set_aspect("equal"); axp.autoscale(); axp.axis("off")
        add_scalebar(axp)
        cb = fig.colorbar(lc, ax=axp, fraction=0.045, pad=0.02)
        cb.set_label("time from wall on (s)")
        axp.set_title("trajectory (planar)", fontsize=9)

        # --- RIGHT: neural activity ---
        axr.imshow(Wn[sl].T, aspect="auto", origin="lower", extent=[tr[0], tr[-1], 0.5, 16.5],
                   cmap=PINK_CMAP, vmin=0, vmax=0.9, rasterized=True)
        plot_wrapped(axr, tr[::ds], colphase[sl][::ds], wrapthr=NCOL / 2, color="#37e0d0",
                     lw=1.1, alpha=0.85, label="extracted phase")
        axr.set_ylabel("FSB column"); axr.set_ylim(0.5, 16.5)
        axr.set_title("FC2 bump (maxmin)", fontsize=9)
        m = idx[sl]
        plot_wrapped(axt, tr[::ds], np.where(m, hw[sl], np.nan)[::ds], color=YELLOW, lw=1.5, label="heading (fly)")
        plot_wrapped(axt, tr[::ds], np.where(m, pa[sl], np.nan)[::ds], color=PINK, lw=1.8, label="FC2 bump (r>0.25)")
        axt.set_ylim(-180, 180); axt.set_ylabel("heading + orient. (deg)")
        sh = tr[shi]
        # event markers under both right panels: wall on = filled white triangle, wall off
        # = open white triangle, shocks = red triangles (labelled once, on the lower panel)
        for ax, yy, lab in ((axr, 0.5, False), (axt, -180, True)):
            ax.axvline(0, color=WHITE, lw=0.8, alpha=0.5)
            ax.axvline(dur, color=WHITE, lw=0.8, alpha=0.5, ls=":")
            ax.plot([0], [yy], "^", ms=10, color=WHITE, ls="none", clip_on=False,
                    label="wall on" if lab else None)
            ax.plot([dur], [yy], "^", ms=10, mfc="none", mec=WHITE, mew=1.5, ls="none",
                    clip_on=False, label="wall off" if lab else None)
            if len(sh):
                ax.plot(sh, np.full(len(sh), yy), "^", ms=8, color="red", ls="none",
                        clip_on=False, label="shock" if lab else None)
            ax.set_xlim(tr[0], tr[-1])
        axt.set_xlabel("time from wall on (s)")

        fig.suptitle(f"FC2 wall trial - {fly} ({eid}) - trial {k} ({dur:.0f} s"
                     f"{', filtered' if filt else ''})", fontsize=11)
        h0, lb0 = axr.get_legend_handles_labels()
        h1, lb1 = axt.get_legend_handles_labels()
        fig.legend(h0 + h1, lb0 + lb1, loc="outside lower center", ncol=len(lb0 + lb1), fontsize=8)
        save_fig(fig, f"fc2_{fly}_trial_{k:03d}{suffix}.png", subdir=sub,
                 svg_subdir=sub + "/svg")
        plt.close(fig)
    print(f"  saved {len(walls)} trials -> plots/{sub}/fc2_{fly}_trial_*{suffix}")


def main():
    if len(sys.argv) > 1:                                    # single-experiment plot(s)
        for fly in (FLIES if sys.argv[1] == "all" else [sys.argv[1]]):
            single_fig(fly, filt=False)
            single_fig(fly, filt=True)                       # readability alternative
            trial_figs(fly, filt=True)                       # per wall-on/off trial
            trial_figs(fly, filt=True, ticks=True)           # same, with 10 s crosses on the path
        return
    frames, all_df = [], []
    for fly in FLIES:
        df, stats = unify(fly)
        frames.append((fly, df, stats))
        d = df.copy(); d.insert(0, "fly", fly)
        all_df.append(d)
    comb = pd.concat(all_df, ignore_index=True)
    comb.to_pickle(os.path.join(DATA_DIR, IMG_DIR, "imaging0630_unified.pkl"))
    print(f"combined: {len(comb)} frames across {len(FLIES)} flies -> "
          f"{os.path.join(DATA_DIR, IMG_DIR, 'imaging0630_unified.pkl')}")
    for f in (False, True):                                  # raw + filtered alternative
        validate_fig(frames, filt=f)
        for fly in FLIES:
            single_fig(fly, filt=f)
    for fly in FLIES:                                        # per wall-on/off trial breakdown
        trial_figs(fly, filt=True)
        trial_figs(fly, filt=True, ticks=True)               # same, with 10 s crosses on the path


if __name__ == "__main__":
    main()
