"""Computation for the confinement scalloping analysis.

Pure, reusable functions (no plotting, no file I/O) that turn an x/y walking
trajectory inside a circular virtual enclosure into per-scallop geometry, timing,
handedness, speed-coupling and spectral metrics. The plotting/runner lives in
``confinement_scalloping.py``.

A *scallop* is one out-and-back excursion: the fly walks out to the wall
(a *contact*, a maximum of the radial distance r), turns, and retreats to a
radial minimum before the next contact. Each contact is the anchor of one scallop.
"""

import numpy as np
from scipy.signal import find_peaks, welch

# Nominal virtual enclosure: 100 mm diameter -> 50 mm radius (rig setting).
ENCLOSURE_RADIUS_MM = 50.0


# --- geometry of the arena -------------------------------------------------

def fit_circle(x, y):
    """Algebraic (Kåsa) least-squares circle fit. Returns (cx, cy, radius)."""
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    radius = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return cx, cy, radius


def arena_geometry(x, y, wall_quantile=0.90):
    """Estimate enclosure centre and radius from the confined path.

    The centre is fit from the outer (wall-touching) points only: start from the
    median centre, then circle-fit the points whose radius is in the top
    ``1 - wall_quantile`` fraction. Returns ``(cx, cy, radius_fit)``.
    """
    cx0, cy0 = np.median(x), np.median(y)
    r0 = np.hypot(x - cx0, y - cy0)
    edge = r0 >= np.quantile(r0, wall_quantile)
    return fit_circle(x[edge], y[edge])


def radial(x, y, cx, cy):
    """Radial distance of each sample from the arena centre."""
    return np.hypot(x - cx, y - cy)


def azimuth(x, y, cx, cy):
    """Angular position (rad) of each sample around the arena centre."""
    return np.arctan2(y - cy, x - cx)


# --- velocity / smoothing --------------------------------------------------

def smooth(a, win):
    """Centred moving-average smooth (odd ``win``); edges shrink the window."""
    if win <= 1:
        return np.asarray(a, float)
    k = np.ones(win) / win
    return np.convolve(np.asarray(a, float), k, mode="same")


def velocity(x, y, fs, smooth_win=7):
    """Smoothed velocity components and movement direction (rad).

    Returns ``(vx, vy, speed, direction)`` with speed in mm/s.
    """
    xs, ys = smooth(x, smooth_win), smooth(y, smooth_win)
    vx = np.gradient(xs) * fs
    vy = np.gradient(ys) * fs
    speed = np.hypot(vx, vy)
    direction = np.arctan2(vy, vx)
    return vx, vy, speed, direction


def angular_velocity(heading, fs, smooth_win=9):
    """Signed turning rate (rad/s) from an (accumulating) heading signal.

    Use ``integrated_heading_lab`` rather than the per-sample movement direction:
    the latter is dominated by noise when the fly is slow, giving spurious
    thousand-deg/s spikes. The heading is unwrapped and lightly smoothed first.
    """
    d = smooth(np.unwrap(heading), smooth_win)
    return np.gradient(d) * fs


# --- contacts and scallops -------------------------------------------------

def detect_contacts(r, radius, fs, height_frac=0.85, prominence_mm=3.0,
                    min_sep_s=0.4, smooth_s=0.1):
    """Indices of wall contacts: prominent radial maxima near the wall.

    ``r`` is smoothed over ``smooth_s`` first; a contact is a local maximum of the
    smoothed signal reaching ``height_frac * radius``, standing at least
    ``prominence_mm`` above its surroundings, with contacts ``min_sep_s`` apart.
    The prominence requirement is what rejects sub-millimetre wall-skimming wiggles
    while keeping every genuine approach-and-turn; the count is stable across a
    wide range of these thresholds (see the parameter sweep in the design notes).
    """
    rs = smooth(r, max(1, int(smooth_s * fs)) | 1)
    peaks, _ = find_peaks(rs, height=height_frac * radius,
                          distance=max(1, int(min_sep_s * fs)),
                          prominence=prominence_mm)
    return peaks


def detect_contacts_hysteresis(r, radius, fs, high_frac=0.90, low_frac=0.75,
                               smooth_s=0.1):
    """Alternative contact detector keyed to *deep* out-and-back excursions.

    The fly is "at the wall" once smoothed ``r`` rises above ``high_frac * radius``
    and stays there until ``r`` falls below ``low_frac * radius`` (a genuine
    retreat to the interior). Each wall bout yields one contact, at its radial
    maximum. This counts deep crossings and merges wall-following petals; contrast
    with :func:`detect_contacts`, which resolves the individual petals.
    """
    rs = smooth(r, max(1, int(smooth_s * fs)) | 1)
    hi, lo = high_frac * radius, low_frac * radius
    contacts, state, start = [], "low", None
    for i, v in enumerate(rs):
        if state == "low" and v >= hi:
            state, start = "high", i
        elif state == "high" and v < lo:
            contacts.append(start + int(np.argmax(rs[start:i])))
            state = "low"
    if state == "high" and start is not None:
        contacts.append(start + int(np.argmax(rs[start:])))
    return np.asarray(contacts, int)


def retreat_indices(r, contacts):
    """Deepest interior point (radial minimum) between consecutive contacts.

    Returns an array the same length as ``contacts``; entry *i* is the index of
    the minimum of ``r`` in the interval ``(contacts[i], contacts[i+1])`` and the
    last entry repeats the previous (no interval after the final contact).
    """
    out = []
    for i in range(len(contacts) - 1):
        a, b = contacts[i], contacts[i + 1]
        out.append(a + 1 + int(np.argmin(r[a + 1:b])) if b > a + 1 else a)
    out.append(out[-1] if out else (contacts[-1] if len(contacts) else 0))
    return np.asarray(out)


def scallop_metrics(x, y, r, cx, cy, radius, contacts, direction, speed, omega, fs,
                    vel_win_s=0.12, turn_win_s=0.25):
    """Per-scallop geometry, handedness and incidence/reflection.

    For each contact: retreat depth to the next interior minimum, the turn angle
    and handedness, the minimum radius of curvature through the turn
    (speed / peak angular velocity), and the angles of incidence/reflection vs the
    inward wall normal. Returns a dict of equal-length arrays (one per contact).
    """
    n = len(x)
    kv = max(2, int(vel_win_s * fs))
    kt = max(3, int(turn_win_s * fs))
    retreats = retreat_indices(r, contacts)

    out = {k: [] for k in (
        "idx", "t_frac", "azimuth", "retreat_depth", "turn_angle", "turn_sign",
        "peak_omega", "incidence", "reflection", "chord")}

    def heading_vec(i0, i1):
        i0, i1 = max(0, i0), min(n - 1, i1)
        return np.array([x[i1] - x[i0], y[i1] - y[i0]])

    prev_contact = None
    for p, rt in zip(contacts, retreats):
        v_in, v_out = heading_vec(p - kv, p), heading_vec(p, p + kv)
        a_in, a_out = np.arctan2(*v_in[::-1]), np.arctan2(*v_out[::-1])
        turn = (a_out - a_in + np.pi) % (2 * np.pi) - np.pi  # signed [-pi, pi]

        nrm = -np.array([x[p] - cx, y[p] - cy])
        nrm = nrm / (np.hypot(*nrm) + 1e-12)

        def ang_to_normal(v):
            v = v / (np.hypot(*v) + 1e-12)
            return np.arccos(np.clip(np.dot(v, nrm), -1, 1))

        # turn sharpness: peak angular velocity through the contact (signed)
        s = slice(max(0, p - kt), min(n, p + kt + 1))
        j = np.argmax(np.abs(omega[s]))
        peak_w = omega[s][j]

        chord = (np.hypot(x[p] - x[prev_contact], y[p] - y[prev_contact])
                 if prev_contact is not None else np.nan)

        out["idx"].append(p)
        out["t_frac"].append(p / n)
        out["azimuth"].append(np.arctan2(y[p] - cy, x[p] - cx))
        out["retreat_depth"].append(radius - r[rt])
        out["turn_angle"].append(turn)
        out["turn_sign"].append(np.sign(peak_w))
        out["peak_omega"].append(peak_w)
        out["incidence"].append(ang_to_normal(v_in))
        out["reflection"].append(ang_to_normal(v_out))
        out["chord"].append(chord)
        prev_contact = p

    return {k: np.asarray(v, float) for k, v in out.items()}


def mean_sem(A, axis=0):
    """Mean and standard error of the mean across ``axis`` (NaN-aware)."""
    A = np.asarray(A, float)
    m = np.nanmean(A, axis=axis)
    n = np.sum(~np.isnan(A), axis=axis)
    sem = np.nanstd(A, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    return m, sem


def bounce_aligned(x, y, cx, cy, contacts, turn_sign, fs, half_win_s=1.2):
    """Stack wall-bounce excursions in a common contact-centred frame.

    Each contact is re-expressed in a frame centred on the contact point, with the
    outward wall normal as +u (so the local wall is the line u=0 and the interior is
    u<0) and the wall tangent as +v. The tangential axis is flipped by the turn
    handedness so every bounce curves the same way and they reinforce rather than
    cancel when averaged. Bounces whose window runs off either end are dropped.

    Returns ``(tau, U, V)``: ``tau`` is time relative to contact (s); ``U`` and ``V``
    are ``(n_bounce, n_tau)`` arrays of radial-offset and tangential position (mm).
    """
    w = int(half_win_s * fs)
    tau = np.arange(-w, w + 1) / fs
    n, U, V = len(x), [], []
    for p, s in zip(contacts, turn_sign):
        if p - w < 0 or p + w >= n:
            continue
        sl = slice(p - w, p + w + 1)
        en = np.array([x[p] - cx, y[p] - cy])
        en = en / (np.hypot(*en) + 1e-12)
        et = np.array([-en[1], en[0]])
        rx, ry = x[sl] - x[p], y[sl] - y[p]
        U.append(rx * en[0] + ry * en[1])
        V.append((rx * et[0] + ry * et[1]) * (s if s else 1.0))
    return tau, np.asarray(U), np.asarray(V)


def bounce_cycle_aligned(sig, contacts, n_half=60):
    """Average a signal over a full bounce *cycle*: previous contact -> this contact
    -> next contact, time-normalised so each contact lands at a fixed phase.

    Inter-contact intervals vary several-fold, so a fixed-time window would smear
    the neighbouring bounces; instead each half-cycle (prev->this, this->next) is
    resampled onto ``n_half`` points. Returns ``(phase, S)`` where ``phase`` runs
    -1 (previous bounce) -> 0 (this bounce) -> +1 (next bounce) and ``S`` is
    ``(n_cycle, 2*n_half+1)``.
    """
    phase = np.linspace(-1, 1, 2 * n_half + 1)
    grid = np.linspace(0, 1, n_half + 1)
    c, S = np.asarray(contacts), []
    for i in range(1, len(c) - 1):
        p0, p1, p2 = c[i - 1], c[i], c[i + 1]
        left = np.interp(grid, np.linspace(0, 1, p1 - p0 + 1), sig[p0:p1 + 1])
        right = np.interp(grid, np.linspace(0, 1, p2 - p1 + 1), sig[p1:p2 + 1])
        S.append(np.concatenate([left[:-1], right]))
    return phase, np.asarray(S)


def bounce_cycle_aligned_xy(x, y, cx, cy, contacts, turn_sign, n_half=60):
    """Like :func:`bounce_aligned` but spanning the full cycle prev->this->next
    contact, phase-normalised (each half-cycle resampled to ``n_half`` points).

    Everything is expressed in the *this-contact* frame (origin at the contact,
    +u outward wall normal, +v tangent, handedness folded). Returns
    ``(phase, U, V)`` with phase -1 (prev bounce) -> 0 (this bounce) -> +1 (next).
    """
    phase = np.linspace(-1, 1, 2 * n_half + 1)
    grid = np.linspace(0, 1, n_half + 1)
    c, U_all, V_all = np.asarray(contacts), [], []
    for i in range(1, len(c) - 1):
        p0, p1, p2 = c[i - 1], c[i], c[i + 1]
        en = np.array([x[p1] - cx, y[p1] - cy])
        en = en / (np.hypot(*en) + 1e-12)
        et = np.array([-en[1], en[0]])
        s = turn_sign[i] if turn_sign[i] else 1.0

        def frame(sl):
            rx, ry = x[sl] - x[p1], y[sl] - y[p1]
            return rx * en[0] + ry * en[1], (rx * et[0] + ry * et[1]) * s

        ul, vl = frame(slice(p0, p1 + 1))
        ur, vr = frame(slice(p1, p2 + 1))
        xl, xr = np.linspace(0, 1, p1 - p0 + 1), np.linspace(0, 1, p2 - p1 + 1)
        U_all.append(np.concatenate([np.interp(grid, xl, ul)[:-1], np.interp(grid, xr, ur)]))
        V_all.append(np.concatenate([np.interp(grid, xl, vl)[:-1], np.interp(grid, xr, vr)]))
    return phase, np.asarray(U_all), np.asarray(V_all)


def bounce_aligned_signal(sig, contacts, fs, half_win_s=1.2):
    """Stack a 1-D signal in windows centred on each contact.

    Returns ``(tau, S)`` with ``S`` of shape ``(n_bounce, n_tau)``; edge-truncated
    bounces are dropped (matching :func:`bounce_aligned`).
    """
    w = int(half_win_s * fs)
    tau = np.arange(-w, w + 1) / fs
    n, S = len(sig), []
    for p in contacts:
        if p - w < 0 or p + w >= n:
            continue
        S.append(sig[p - w:p + w + 1])
    return tau, np.asarray(S)


def net_circling(contact_azimuth):
    """Net signed rotations of the contact point around the rim.

    Unwraps the per-contact azimuth and returns total turns (positive = CCW).
    """
    if len(contact_azimuth) < 2:
        return 0.0
    return float((np.unwrap(contact_azimuth)[-1] - contact_azimuth[0]) / (2 * np.pi))


# --- speed coupling --------------------------------------------------------

def speed_vs_radius(r, speed, radius, nbins=12):
    """Mean speed in radial bins from centre to wall. Returns (bin_centres, mean_speed)."""
    edges = np.linspace(0, radius, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    means = np.array([speed[idx == b].mean() if np.any(idx == b) else np.nan
                      for b in range(nbins)])
    return centres, means


# --- spectral --------------------------------------------------------------

def psd(sig, fs, nperseg=4096, detrend_s=5.0):
    """Welch PSD of a signal after removing slow drift.

    Subtracting a ``detrend_s``-second moving average suppresses the 1/f trend so
    a rhythmic (scalloping) band is not swamped. Returns (freqs, power).
    """
    sig = np.asarray(sig, float)
    if detrend_s:
        sig = sig - smooth(sig, max(1, int(detrend_s * fs)) | 1)
    else:
        sig = sig - np.nanmean(sig)
    nperseg = min(nperseg, len(sig))
    f, pxx = welch(sig, fs=fs, nperseg=nperseg)
    return f, pxx


def spectral_peak(f, pxx, band=(0.1, 3.0)):
    """Dominant frequency and its power within ``band``. Returns (f_peak, power)."""
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return np.nan, np.nan
    i = np.argmax(pxx[m])
    return float(f[m][i]), float(pxx[m][i])


# --- circular statistics (for menotaxis heading) ---------------------------

def circ_mean_R(angles, weights=None):
    """Circular mean (rad) and resultant length R in [0, 1] of ``angles`` (rad).

    ``weights`` (e.g. per-sample dwell time or per-bout duration) lets longer
    fixations count proportionally more.
    """
    angles = np.asarray(angles, float)
    w = np.ones_like(angles) if weights is None else np.asarray(weights, float)
    c = np.sum(w * np.cos(angles)) / np.sum(w)
    s = np.sum(w * np.sin(angles)) / np.sum(w)
    return float(np.arctan2(s, c)), float(np.hypot(c, s))


def circ_sem(angles, n_eff=None):
    """Standard error of the circular mean (rad) for ``angles`` (rad).

    Uses the circular standard deviation ``sqrt(-2 ln R)`` over ``sqrt(N)``. Pass
    ``n_eff`` to discount autocorrelation (e.g. one independent sample per second)
    instead of the raw sample count, which otherwise makes the SEM far too small.
    """
    _, R = circ_mean_R(angles)
    n = len(angles) if n_eff is None else n_eff
    R = min(max(R, 1e-9), 1 - 1e-9)
    return float(np.sqrt(-2 * np.log(R)) / np.sqrt(max(n, 1)))


def straight_bouts(omega, speed, fs, max_omega_deg=20.0, min_speed=1.0, min_dur_s=2.0):
    """Index intervals of sustained straight walking (menotaxis fixation bouts).

    A bout is a maximal run where ``|omega| < max_omega_deg`` and
    ``speed > min_speed`` lasting at least ``min_dur_s``. Returns a list of
    ``(start, stop)`` index pairs.
    """
    ok = (np.abs(omega) < np.radians(max_omega_deg)) & (speed > min_speed)
    bouts, i, n, lo = [], 0, len(ok), int(min_dur_s * fs)
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j]:
                j += 1
            if j - i >= lo:
                bouts.append((i, j))
            i = j
        else:
            i += 1
    return bouts
