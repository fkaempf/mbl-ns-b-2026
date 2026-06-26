"""Compute for the single-bounce (wall-bounce) analysis.

A *bounce* is a laser-on event: the aversive laser fires (logged in
``light_sugar_commands.csv`` as ``laser_exponential_set_end_level -> 255``) when the
fly enters a barrier's hurt zone. Events within ``REFRACTORY_S`` collapse to one
bounce. Each bounce is matched to the nearest collision (``vrcollisions.csv``) to get
the wall surface point (origin) and its normal, then the vrpos trajectory around the
bounce is rotated so the wall is horizontal and mirror-flipped so the approach always
comes from the left. The incidence angle (0 deg grazing -> 90 deg head-on) is the
direction of travel over the 5 s before the hit, using only moving (>=5 mm/s) samples.
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd

from utils import DATA_DIR, load_combined

REFRACTORY_S = 1.0       # laser events closer than this are the same bounce
MIN_SPEED = 5.0          # mm/s; direction is only taken from moving samples
MAX_SPEED = 50.0         # mm/s; drop FicTrac glitch jumps
INCID_S = 5.0            # seconds before/after the hit to average direction
MATCH_S = 1.0            # max time gap to match a bounce to a collision
WALL_TH = 2.0            # barrier thickness (config)
LASER_MARGIN = 2.5       # laser_margin_y (config): hurt-zone half-width


def laser_on_times(folder):
    """Bounce times (unix s): laser ramps to 255, deduped with the refractory."""
    ls = glob.glob(os.path.join(DATA_DIR, folder, "*light_sugar*commands.csv"))[0]
    d = pd.read_csv(ls)
    on = d["laser_exponential_set_end_level"].astype(float) == 255
    t = np.sort(d.timestamp.to_numpy()[on.to_numpy()].astype(float) / 1e9)
    out, last = [], -np.inf
    for ti in t:
        if ti - last >= REFRACTORY_S:
            out.append(ti); last = ti
    return np.array(out)


def collisions(folder):
    """Collision events (unix s, wall x, wall y, normal x, normal y). The vrcollisions
    header is shifted (empty last_timestamp), so read it positionally."""
    cols = ["unity_time", "name", "fx", "fy", "fz", "bx", "by", "bz", "nx", "ny", "nz"]
    f = glob.glob(os.path.join(DATA_DIR, folder, "*vrcollisions.csv"))[0]
    c = pd.read_csv(f, skiprows=1, header=None, names=cols)
    return (c.unity_time.to_numpy(), c.bx.to_numpy(), c.by.to_numpy(),
            c.nx.to_numpy(), c.ny.to_numpy())


def _angle_from_wall(du, dv):
    """Angle (deg) of a displacement from the wall surface: 0 = grazing, 90 = head-on."""
    return np.degrees(np.arctan2(abs(dv), abs(du)))


def bounces(folder, half_win_s=120.0, dt=0.5):
    """Per-bounce wall-aligned data for one experiment.

    Returns a list of dicts with ``tau`` (shared time grid, s from hit), ``U``/``V``
    (wall-frame tangential/normal trajectory, mm; approach from the left, fly side
    v>0, wall at v=0), ``incidence`` and ``outgoing`` angles (deg).
    """
    bt = laser_on_times(folder)
    ct, bx, by, nx, ny = collisions(folder)
    if len(bt) == 0 or len(ct) == 0:
        return []
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=MAX_SPEED)
    t, x, y, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(),
                   df.vry.to_numpy(), df.speed.to_numpy())
    tau = np.arange(-half_win_s, half_win_s + dt, dt)
    out = []
    for tb in bt:
        j = int(np.argmin(np.abs(ct - tb)))
        if abs(ct[j] - tb) > MATCH_S:
            continue
        n = np.array([nx[j], ny[j]], float)
        n /= np.hypot(*n) + 1e-12
        w = (t >= tb - half_win_s) & (t <= tb + half_win_s)
        if w.sum() < 10:
            continue
        tw, xw, yw, spw = t[w] - tb, x[w] - bx[j], y[w] - by[j], sp[w]

        pre = (tw >= -INCID_S) & (tw < 0)
        if pre.sum() < 3:
            continue
        if np.array([xw[pre].mean(), yw[pre].mean()]) @ n < 0:   # orient normal to fly side
            n = -n
        eu = np.array([n[1], -n[0]])                              # tangent (right-handed)
        u = xw * eu[0] + yw * eu[1]
        v = xw * n[0] + yw * n[1]

        mov = pre & (spw >= MIN_SPEED)
        if mov.sum() < 3:
            continue
        du, dv = u[mov][-1] - u[mov][0], v[mov][-1] - v[mov][0]   # approach displacement
        if du < 0:                                                # flip: approach from left
            u, du = -u, -du
        incidence = _angle_from_wall(du, dv)

        post = (tw > 0) & (tw <= INCID_S) & (spw >= MIN_SPEED)
        if post.sum() >= 3:
            duo, dvo = u[post][-1] - u[post][0], v[post][-1] - v[post][0]
            outgoing = _angle_from_wall(duo, dvo)
            # signed wall-frame exit direction: 0 = continue along wall (same way as
            # the approach), 90 = straight off the wall, >90 = reverse (turn back).
            out_dir = np.degrees(np.arctan2(dvo, duo))
        else:
            outgoing = out_dir = np.nan

        out.append(dict(
            tau=tau,
            U=np.interp(tau, tw, u, left=np.nan, right=np.nan),
            V=np.interp(tau, tw, v, left=np.nan, right=np.nan),
            incidence=incidence, outgoing=outgoing, out_dir=out_dir))
    return out


def mean_sem(A):
    """Column-wise mean and SEM of a (n, m) array, NaN-aware (empty columns -> NaN)."""
    A = np.asarray(A, float)
    n = np.sum(~np.isnan(A), axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(A, axis=0)
        sem = np.nanstd(A, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return m, sem
