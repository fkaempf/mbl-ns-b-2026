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
import re
import warnings

import numpy as np
import pandas as pd

from utils import DATA_DIR, PLOTS_DIR, experiment_id, load_combined

REFRACTORY_S = 1.0       # laser events closer than this are the same bounce
MIN_SPEED = 5.0          # mm/s; direction is only taken from moving samples
MAX_SPEED = 50.0         # mm/s; drop FicTrac glitch jumps
INCID_S = 5.0            # seconds before/after the hit to average direction
MATCH_S = 1.0            # max time gap to match a bounce to a collision
WALL_TH = 2.0            # barrier thickness (config)
LASER_MARGIN = 2.5       # laser_margin_y (config): hurt-zone half-width


GOOD_CSV = os.path.join(PLOTS_DIR, "bounce_gallery", "good_bounces.csv")


def good_set():
    """The ``(eid, laser-on index)`` set hand-tagged in the napari selector
    (``bounce_select.py``), or ``None`` if no curation file exists yet. The downstream
    analyses use it to restrict to the curated bounces and write to a ``*_good`` folder."""
    if not os.path.exists(GOOD_CSV):
        return None
    d = pd.read_csv(GOOD_CSV)
    return {(r.eid, int(r.bounce)) for _, r in d[d.good].iterrows()}


def laser_on_times(folder):
    """Bounce times (unix s): laser ramps to a positive level, deduped with the
    refractory. Use ``> 0`` (not ``== 255``) so it works whatever the laser power is
    set to - newer runs ramp to 125, older ones to 255."""
    ls = glob.glob(os.path.join(DATA_DIR, folder, "*light_sugar*commands.csv"))[0]
    d = pd.read_csv(ls)
    on = d["laser_exponential_set_end_level"].astype(float) > 0
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


def rectmaze_walls(folder):
    """``CREATE RectMaze`` commands -> (cx, cy, rot_deg, width, thickness, t_create)."""
    cmd = glob.glob(os.path.join(DATA_DIR, folder, "*vrcmd.csv"))[0]
    out = []
    with open(cmd) as fh:
        next(fh, None)
        for line in fh:
            p = line.split(",")
            if len(p) > 13 and p[1] == "CREATE" and p[2] == "RectMaze":
                out.append((float(p[5]), float(p[6]), float(p[10]),
                            float(p[11]), float(p[12]), float(p[0])))
    return out


def wall_trials(folder):
    """Barrier *presentations* as trials: one trial = a wall from its CREATE (wall-on)
    to its matching DESTROY (wall-off). Returns a list of
    ``(t_on, t_off, cx, cy, rot_deg, width, thickness)`` sorted by onset.

    CREATE and DESTROY are paired by the serial in field ``p[3]`` (same for both); the
    DESTROY row carries the serial in ``p[2]`` (not the generic ``RectMaze``), which is
    why a naive ``p[2]=='RectMaze'`` filter sees only creations."""
    cmd = glob.glob(os.path.join(DATA_DIR, folder, "*vrcmd.csv"))[0]
    ev, geom = [], {}
    with open(cmd) as fh:
        next(fh, None)
        for line in fh:
            p = line.split(",")
            if len(p) <= 13:
                continue
            if p[1] == "CREATE" and p[2] == "RectMaze":
                geom[p[3]] = (float(p[5]), float(p[6]), float(p[10]),
                              float(p[11]), float(p[12]))
                ev.append((float(p[0]), "on", p[3]))
            elif p[1] == "DESTROY" and p[2].startswith("RectMaze#"):
                ev.append((float(p[0]), "off", p[3]))
    ev.sort()
    on, out = {}, []
    for t, a, s in ev:
        if a == "on":
            on[s] = t
        elif s in on and s in geom:
            out.append((on.pop(s), t) + geom[s])
    return sorted(out)


# Hand-picked good barrier runs (date/experiment), across all datasets. Analyses
# default to these; pass mode="all" to use every barrier run instead.
GOOD_EXPERIMENTS = [
    "20260625/rig1_experiment_72", "20260625/rig1_experiment_55",
    "20260625/rig1_experiment_46",
    "20260625/rig1_experiment_60", "20260627/rig1_experiment_29",
    "20260627/rig1_experiment_26", "20260627/rig1_experiment_25",
    "20260627/rig1_experiment_23", "20260627/rig1_experiment_21",
    "20260626/rig1_experiment_30",
    "20260629/rig1_experiment_01", "20260629/rig1_experiment_02",
    "20260629/rig1_experiment_08", "20260629/rig1_experiment_15",
    "20260629/rig1_experiment_16",
    "20260630/rig1_experiment_05", "20260630/rig1_experiment_06",
]
BARRIER_DATASETS = ["20260625", "20260626", "20260627", "20260629", "20260630"]


def all_barrier_experiments():
    """Every barrier run (RectMaze + vrpos) across BARRIER_DATASETS, as relpaths."""
    out = []
    for ds in BARRIER_DATASETS:
        for cmd in sorted(glob.glob(os.path.join(DATA_DIR, ds, "*", "*vrcmd.csv"))):
            folder = os.path.dirname(cmd)
            if glob.glob(os.path.join(folder, "*vrpos.csv")) and rectmaze_walls(folder):
                out.append(os.path.relpath(folder, DATA_DIR))
    return out


BARRIER_WIDTH = 100    # restrict analyses to runs with this barrier width (mm); None = all
STIMULUS = os.environ.get("FLY_STIMULUS") or None   # 'green'/'white' (env FLY_STIMULUS); None = all


def experiment_barrier_width(folder):
    """The dominant barrier width (mm) of a run (each run uses one width here)."""
    ws = [round(w) for *_, w, th in wall_trials(folder)]
    return max(set(ws), key=ws.count) if ws else None


def sun_color(folder):
    """The Sun landmark's RGB (from the run's config.yaml ``sun_color``), or None."""
    cfs = glob.glob(os.path.join(DATA_DIR, folder, "*config.yaml"))
    if not cfs:
        return None
    m = re.search(r"sun_color:\s*\(([^)]+)\)", open(cfs[0]).read())
    return tuple(float(v) for v in m.group(1).split(",")[:3]) if m else None


def sun_group(folder):
    """Stimulus group from the sun colour: 'green' (green sun) vs 'white' (default)."""
    c = sun_color(folder)
    if c is None:
        return "unknown"
    r, g, b = c
    return "green" if (g > 0.5 and r < 0.5 and b < 0.5) else "white"


def barrier_experiments(mode="good"):
    """Experiment list for an analysis: 'good' = curated runs, 'all' = every run;
    restricted to BARRIER_WIDTH-wide barriers and the STIMULUS sun-group when set."""
    exps = list(GOOD_EXPERIMENTS) if mode == "good" else all_barrier_experiments()
    if BARRIER_WIDTH is not None:
        exps = [e for e in exps if experiment_barrier_width(e) == BARRIER_WIDTH]
    if STIMULUS is not None:
        exps = [e for e in exps if sun_group(e) == STIMULUS]
    return exps


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
    eid = experiment_id(folder)
    out = []
    for i, tb in enumerate(bt, 1):            # i = laser-on index (matches the gallery/CSV)
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
            eid=eid, i=i,
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
