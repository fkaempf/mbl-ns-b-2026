"""Shared compute for the FC2 + behaviour analyses (FC2_ANALYSIS_PLAN.md, items 1-7).

Reads the ``<fly>_unified.pkl`` written by ``imaging_unify``. The bump is characterised
by a per-frame least-squares fit of the 16 fan-shaped-body columns to

    f(col) = A * cos(theta - phi) + C

(the first Fourier harmonic on evenly spaced columns): ``A`` = magnitude, ``phi`` =
phase, ``C`` = baseline shelf. The fit ``phi`` equals the existing PVA phase exactly
(subtracting each frame's column-min and normalising does not rotate the resultant on a
full circle), so it is a safe drop-in; ``A`` is the true cosine modulation depth, which
is the canonical "magnitude" for every downstream item. The column profile fitted is the
same maxmin-normalised matrix the PVA uses (``imaging_unify.norm_matrix``).

Also here: task-event edges (shock / wall on / wall off), event-triggered averaging
(scalar and circular), bump angular velocity, and within-fly windowed correlation of
heading vs phase. Correlations are computed within fly (the FC2 offset differs across
flies); the honest unit is the fly, not the sample.
"""

import os

import numpy as np
import pandas as pd

from utils import DATA_DIR
from imaging_unify import IMG_DIR, FLIES, NCOL, ROI_COLS, norm_matrix, circ_corr, csmooth
from cxstyle import PINK, YELLOW, WHITE

MOVE_CUTOFF = 5.0        # mm/s; frames slower than this have a meaningless travel direction
# The analysis figures use only the strong-bump flies (Fly1/Fly2 have too weak a bump to
# contribute); Fly5_002 is the cleanest. Set to ["Fly5_002"] for a Fly5-only view.
ANALYSIS_FLIES = ["Fly5_002", "Fly6_002"]
FLY_COLORS = [PINK, YELLOW, "#37e0d0", WHITE]        # per-fly colours, shared by the plot scripts


def wrap180(d):
    """Wrap an angle (deg) to [-180, 180)."""
    return (np.asarray(d, float) + 180) % 360 - 180


def load(fly):
    """Load a fly's unified pickle.

    Trusted first-party data: these pickles are written by ``imaging_unify`` in this repo
    (plain pandas DataFrames), not an untrusted source, so ``read_pickle`` is safe here.
    """
    return pd.read_pickle(os.path.join(DATA_DIR, IMG_DIR, f"{fly}_unified.pkl"))


# --- the bump fit -----------------------------------------------------------------

def fit_bump(Fn):
    """Per-frame least-squares fit of ``A*cos(theta-phi)+C`` to the 16 columns.

    ``Fn`` is a (T, 16) column matrix (use :func:`imaging_unify.norm_matrix`). Because the
    columns are evenly spaced this is the first Fourier harmonic in closed form. Returns
    ``A`` (magnitude), ``phi`` (phase, deg 0..360), ``C`` (baseline shelf), and per-frame
    fit ``r2``.
    """
    Fn = np.asarray(Fn, float)
    T, n = Fn.shape
    ang = np.arange(n) * 2 * np.pi / n + np.pi / n      # matches get_pva / PVA convention
    a = (2.0 / n) * (Fn @ np.cos(ang))
    b = (2.0 / n) * (Fn @ np.sin(ang))
    A = np.hypot(a, b)
    phi = np.degrees(np.arctan2(b, a)) % 360
    C = Fn.mean(axis=1)
    fit = A[:, None] * np.cos(ang[None, :] - np.radians(phi)[:, None]) + C[:, None]
    ss_res = np.sum((Fn - fit) ** 2, axis=1)
    ss_tot = np.sum((Fn - C[:, None]) ** 2, axis=1)
    r2 = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)
    return A, phi, C, r2


def add_fit(df):
    """Return a copy of ``df`` with ``fit_A, fit_phase, fit_C, fit_r2`` attached, plus the
    fly heading (``head`` deg 0..360, fulltrack) and the heading-minus-phase offset
    (``offset`` deg, wrapped to [-180,180], aligned by the moving-frame circular mean)."""
    A, phi, C, r2 = fit_bump(norm_matrix(df))
    df = df.copy()
    df["fit_A"], df["fit_phase"], df["fit_C"], df["fit_r2"] = A, phi, C, r2
    df["head"] = (df["fulltrack_heading"].to_numpy(float) * 180 / np.pi) % 360
    mov = df["speed"].to_numpy() >= MOVE_CUTOFF
    off_raw = (df["head"].to_numpy() - phi) % 360
    off0 = _circmean_deg(off_raw[mov & np.isfinite(off_raw)])
    df["offset"] = wrap180(off_raw - off0)
    return df


def _circmean_deg(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return np.degrees(np.angle(np.mean(np.exp(1j * np.radians(a))))) % 360


def bump_angular_velocity(phase_deg, t, smooth_n=7):
    """Wrap-aware |dphi/dt| in deg/s from a phase series (deg) sampled at times ``t`` (s).

    The raw per-frame phase is jittery (a one-frame wobble at ~120 Hz reads as thousands of
    deg/s), so the phase is first circularly smoothed over ``smooth_n`` samples (~0.06 s);
    a real bump jump over ~0.1-0.3 s survives. Pass ``smooth_n=1`` for the raw derivative.
    """
    phase_deg = np.asarray(phase_deg, float)
    t = np.asarray(t, float)
    ph = csmooth(phase_deg, smooth_n) if smooth_n and smooth_n > 1 else phase_deg
    dphi = wrap180(np.diff(ph))
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    v = np.abs(dphi / dt)
    return np.concatenate([[np.nan], v])          # same length as input


# --- task-event edges -------------------------------------------------------------

def _rising(mask):
    m = np.asarray(mask).astype(int)
    idx = list(np.where(np.diff(m) == 1)[0] + 1)
    if m[0]:
        idx = [0] + idx
    return np.array(idx, int)


def _falling(mask):
    m = np.asarray(mask).astype(int)
    idx = list(np.where(np.diff(m) == -1)[0] + 1)
    if m[-1]:
        idx = idx + [len(m) - 1]
    return np.array(idx, int)


def shock_onset_times(df):
    """Times (s) of every aversive-laser onset (rising edge of ``laser_on``)."""
    return df["time"].to_numpy()[_rising(df["laser_on"].to_numpy())]


def first_shock_time(df):
    """Time (s) of the first laser onset, or None if the fly was never shocked."""
    on = shock_onset_times(df)
    return float(on[0]) if len(on) else None


def wall_on_times(df):
    return df["time"].to_numpy()[_rising(df["wall_on"].to_numpy())]


def wall_off_times(df):
    return df["time"].to_numpy()[_falling(df["wall_on"].to_numpy())]


# --- event-triggered averaging ----------------------------------------------------

def event_aligned(t, sig, events, pre, post, dt=0.1):
    """Align a scalar signal to each event and resample onto a common grid.

    Returns ``(grid, M)`` where ``grid`` is ``[-pre, post]`` at step ``dt`` and ``M`` is
    (n_events x len(grid)), each row the signal interpolated around one event time. Use
    :func:`mean_sem` for the mean +/- SEM.
    """
    t = np.asarray(t, float)
    sig = np.asarray(sig, float)
    grid = np.arange(-pre, post + 1e-9, dt)
    rows = []
    for e in events:
        m = np.isfinite(sig)
        rows.append(np.interp(e + grid, t[m], sig[m], left=np.nan, right=np.nan))
    return grid, (np.vstack(rows) if rows else np.empty((0, len(grid))))


def event_aligned_phase_change(t, phase_deg, events, pre, post, dt=0.1, base=(-2.0, 0.0)):
    """Align a circular phase (deg) to each event as a *change* from its pre-event baseline.

    Absolute phase is arbitrary per event, so each row is rotated so the mean phase in the
    ``base`` window (s, relative to the event) is 0, then wrapped to [-180,180]. Returns
    ``(grid, M)`` of signed phase change in deg; average with :func:`mean_sem`.
    """
    t = np.asarray(t, float)
    ph = np.radians(np.asarray(phase_deg, float))
    grid = np.arange(-pre, post + 1e-9, dt)
    m = np.isfinite(phase_deg)
    rows = []
    for e in events:
        ci = np.interp(e + grid, t[m], np.cos(ph[m]), left=np.nan, right=np.nan)
        si = np.interp(e + grid, t[m], np.sin(ph[m]), left=np.nan, right=np.nan)
        ang = np.arctan2(si, ci)
        bsel = (grid >= base[0]) & (grid < base[1])
        b0 = np.angle(np.nanmean(np.exp(1j * ang[bsel]))) if np.isfinite(ang[bsel]).any() else np.nan
        rows.append(wrap180(np.degrees(ang - b0)))
    return grid, (np.vstack(rows) if rows else np.empty((0, len(grid))))


def binned(x, y, bins, min_n=20):
    """Mean of ``y`` within each ``x`` bin: returns bin-centre, mean, SEM arrays. Bins with
    fewer than ``min_n`` finite points are dropped (so sparse tails do not add noisy points)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.digitize(x, bins)
    cx, cy, ce = [], [], []
    for k in range(1, len(bins)):
        m = (idx == k) & np.isfinite(y) & np.isfinite(x)
        if m.sum() >= min_n:
            cx.append(float(np.mean(x[m])))
            cy.append(float(np.mean(y[m])))
            ce.append(float(np.std(y[m]) / np.sqrt(m.sum())))
    return np.array(cx), np.array(cy), np.array(ce)


def mean_sem(M):
    """Column-wise nan-mean and nan-SEM of an (n x g) event-aligned matrix."""
    M = np.asarray(M, float)
    n = np.sum(np.isfinite(M), axis=0)
    mean = np.nanmean(M, axis=0)
    sem = np.nanstd(M, axis=0) / np.sqrt(np.maximum(n, 1))
    return mean, sem


# --- within-fly windowed correlation ---------------------------------------------

def windowed(df, win_s=10.0, step_s=5.0, min_move_frac=0.3, min_n=50):
    """Slide a window over a fly's session and, per moving-enough window, compute the
    circular correlation of heading vs bump phase together with the window's mean bump
    magnitude and mean walking speed.

    ``df`` must already carry ``fit_A``/``fit_phase``/``head`` (call :func:`add_fit`).
    Returns a DataFrame with one row per kept window: ``circ_r`` (heading vs phase),
    ``mean_A`` (magnitude), ``mean_speed`` (mm/s, moving frames), ``t_mid`` (s).
    """
    t = df["time"].to_numpy(); t = t - t[0]
    head = df["head"].to_numpy()
    phase = df["fit_phase"].to_numpy()
    A = df["fit_A"].to_numpy()
    speed = df["speed"].to_numpy()
    mov = speed >= MOVE_CUTOFF
    fs = 1.0 / np.median(np.diff(t))
    win, step = int(win_s * fs), int(step_s * fs)
    out = []
    for i in range(0, len(t) - win, step):
        s = slice(i, i + win)
        m = mov[s] & np.isfinite(phase[s]) & np.isfinite(head[s])
        if m.sum() < min_n or m.mean() < min_move_frac:
            continue
        out.append(dict(
            t_mid=float(t[s][win // 2]),
            circ_r=circ_corr(head[s][m], phase[s][m]),
            mean_A=float(np.nanmean(A[s][m])),
            mean_speed=float(np.nanmean(speed[s][m])),
        ))
    return pd.DataFrame(out)
