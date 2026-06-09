#!/usr/bin/env python
"""Shared-axes versions of all 7 final figure types for BOTH cohorts.

Builds 14 PNGs (adult + juvenile of each of the 7 figure types) into
  "plots closer selection/shared axes/"
so the ADULT and JUVENILE equivalents can be compared side by side: every
matched pair of figures uses IDENTICAL axis limits / color-scale limits
(and, for the occupancy heatmap, identical cm extent + cell size).

Computation, calibration, the colorblind (Okabe-Ito) palette and the
same-dish grouping are copied VERBATIM from:
  analysis_fed_leech/scripts/make_selection.py   (adult)
  analysis_juvenile/scripts/make_selection.py    (juvenile)
  scripts/design_2x2_shared_y.py                 (the 2x2 shared-y logic)
Only AXIS LIMITS (and the heatmap cell size / shared color scale) are
changed so they match across cohorts. The original scripts are NOT modified.

No em dashes anywhere in any plot text.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter


def two_shades(c):
    """Two distinguishable shades of a base color (for the 2 animals in a dish)."""
    r, g, b = to_rgb(c)
    return c, (r * 0.45, g * 0.45, b * 0.45)

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
OUT = f"{BASE}/plots closer selection/shared axes"
os.makedirs(OUT, exist_ok=True)

FPS = 29.99
DT = 1.0 / FPS

# Shared-frame constants used across cohorts.
SHARED_HALF = 5.0      # symmetric cm half-extent for arena scatter + heatmaps
SHARED_HM_CELL_CM = 0.25  # SAME heatmap cell size for both cohorts (overrides juv 0.10)
R_ADULT = 4.5
R_JUV = 1.75

NOTE = "axes matched across adult and juvenile"

# ===========================================================================
# ADULT computation (verbatim from analysis_fed_leech/scripts/make_selection.py)
# ===========================================================================
A_CSV = f"{BASE}/DL/all_predictions.csv"
A_MET = f"{BASE}/analysis_fed_leech/metrics"
A_R = 4.5
A_CELL_CM = 0.25

A_TREAT_COLOR = {
    "Veh+Food":   "#c6dbef",
    "DA+Food":    "#6baed6",
    "Veh+NoFood": "#2171b5",
    "DA+NoFood":  "#08306b",
}

A_ARENA = {
    1: dict(cx=303, cy=295, cmpp=0.0200),
    2: dict(cx=290, cy=311, cmpp=0.0205),
    3: dict(cx=292, cy=284, cmpp=0.0202),
    4: dict(cx=311, cy=293, cmpp=0.0206),
}
A_VIDEO = {1: "IMG_2859_dish1.mp4", 2: "IMG_2859_dish2.mp4",
           3: "IMG_2859_dish3.mp4", 4: "IMG_2859_dish4.mp4"}

A_ENTITIES = [(1, 0), (2, 0), (3, 0), (4, 0), (4, 1)]
A_ORDER = ["dish1", "dish2", "dish3", "dish4_leech0", "dish4_leech1"]
A_LOWCONF = {"dish4_leech0", "dish4_leech1"}
A_ENT_TREAT = {
    "dish1": "Veh+Food",
    "dish2": "DA+Food",
    "dish3": "Veh+NoFood",
    "dish4_leech0": "DA+NoFood",
    "dish4_leech1": "DA+NoFood",
}
A_DISH_TREAT = {1: "Veh+Food", 2: "DA+Food", 3: "Veh+NoFood", 4: "DA+NoFood"}

df = pd.read_csv(A_CSV)
df["dish"] = df["video"].str.extract(r"dish(\d)").astype(int)


def smooth_thig(a):
    a = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    if len(a) < 11:
        return a
    return savgol_filter(a, 11, 2)


a_thig = {}
for dish, track in A_ENTITIES:
    g = df[(df.dish == dish) & (df.track == track)]
    h = g[g.node == 0].sort_values("frame")
    t = g[g.node == 1].sort_values("frame")
    m = pd.merge(h[["frame", "x", "y"]], t[["frame", "x", "y"]],
                 on="frame", suffixes=("_h", "_t"))
    if len(m) < 100:
        continue
    a = A_ARENA[dish]
    cmpp = a["cmpp"]
    hx = (smooth_thig(m.x_h.to_numpy()) - a["cx"]) * cmpp
    hy = (smooth_thig(m.y_h.to_numpy()) - a["cy"]) * cmpp
    tx = (smooth_thig(m.x_t.to_numpy()) - a["cx"]) * cmpp
    ty = (smooth_thig(m.y_t.to_numpy()) - a["cy"]) * cmpp
    cx = (hx + tx) / 2.0
    cy = (hy + ty) / 2.0
    r_cm = np.sqrt(cx**2 + cy**2)
    key = f"dish{dish}" if dish != 4 else f"dish4_leech{track}"
    a_thig[key] = dict(dish=dish, track=track, frames=m.frame.to_numpy(),
                       cx=cx, cy=cy, r_cm=r_cm, mean_r=float(np.mean(r_cm)))

A_WIN = 11


def smooth_space(a):
    a = np.asarray(a, float)
    if len(a) < A_WIN:
        return a
    nans = np.isnan(a)
    if nans.any():
        idx = np.arange(len(a))
        a = a.copy()
        if (~nans).any():
            a[nans] = np.interp(idx[nans], idx[~nans], a[~nans])
    return savgol_filter(a, A_WIN, 2)


dfp = df.pivot_table(index=["video", "track", "frame", "time_s"],
                     columns="node", values=["x", "y"]).reset_index()
dfp.columns = ["video", "track", "frame", "time_s", "x0", "x1", "y0", "y1"]
dfp = dfp.sort_values(["video", "track", "frame"])
dfp["cx_px"] = (dfp.x0 + dfp.x1) / 2.0
dfp["cy_px"] = (dfp.y0 + dfp.y1) / 2.0

a_space_results = {}
a_expcurves = {}
a_cmcoords = {}
for dish, track in A_ENTITIES:
    v = A_VIDEO[dish]
    name = f"dish{dish}" if dish != 4 else f"dish4_leech{track}"
    sub = dfp[(dfp.video == v) & (dfp.track == track)].sort_values("frame")
    if len(sub) < 100:
        continue
    d = A_ARENA[dish]
    cx = (smooth_space(sub.cx_px.values) - d["cx"]) * d["cmpp"]
    cy = (smooth_space(sub.cy_px.values) - d["cy"]) * d["cmpp"]
    tt = sub.time_s.values
    valid = np.isfinite(cx) & np.isfinite(cy)
    cx, cy, tt = cx[valid], cy[valid], tt[valid]
    a_cmcoords[name] = (cx, cy)

    gx = np.floor(cx / A_CELL_CM).astype(np.int64)
    gy = np.floor(cy / A_CELL_CM).astype(np.int64)
    cellid = gx * 100000 + gy
    seen = set()
    cum_cells = np.empty(len(cellid), dtype=np.int64)
    for i, c in enumerate(cellid):
        seen.add(c)
        cum_cells[i] = len(seen)
    unique_cells = int(cum_cells[-1])
    cell_area = A_CELL_CM * A_CELL_CM
    cum_area_cm2 = cum_cells * cell_area
    area_grid_cm2 = unique_cells * cell_area
    a_expcurves[name] = (tt, cum_area_cm2)
    a_space_results[name] = dict(area_grid_cm2=float(area_grid_cm2))


def smooth_kin(arr):
    n = len(arr)
    if n < 11:
        return arr
    return savgol_filter(arr, 11, 2)


a_kin = {}
for dish, track in A_ENTITIES:
    name = f"dish{dish}" if dish != 4 else f"dish4_leech{track}"
    sub = df[(df.dish == dish) & (df.track == track)]
    if len(sub) < 1000:
        continue
    cal = A_ARENA[dish]
    piv = sub.pivot_table(index="frame", columns="node",
                          values=["x", "y"]).sort_index()

    def to_cm(px, c, cal=cal):
        return (px - c) * cal["cmpp"]

    hx = smooth_kin(to_cm(piv[("x", 0)].to_numpy(), cal["cx"]))
    hy = smooth_kin(to_cm(piv[("y", 0)].to_numpy(), cal["cy"]))
    dx = np.diff(hx)
    dy = np.diff(hy)
    total_head_cm = float(np.sqrt(dx**2 + dy**2).sum())
    a_kin[name] = dict(total_head_cm=total_head_cm)

A_DISP5 = {"dish1": "Veh+Food", "dish2": "DA+Food", "dish3": "Veh+NoFood",
           "dish4_leech0": "DA+NoFood L0", "dish4_leech1": "DA+NoFood L1"}

# ===========================================================================
# JUVENILE computation (verbatim from analysis_juvenile/scripts/make_selection.py)
# ===========================================================================
J_BASE = f"{BASE}/analysis_juvenile"
J_CSV = f"{BASE}/DL/juv_all_predictions.csv"
J_MET = f"{J_BASE}/metrics"
J_R = 1.75

J_CAL = {
    "Video1_dish0.mp4": dict(cx=274.2, cy=275.4, cmpp=0.0067609),
    "Video1_dish1.mp4": dict(cx=271.8, cy=267.0, cmpp=0.0069246),
    "Video2_dish0.mp4": dict(cx=282.6, cy=269.4, cmpp=0.0070045),
    "Video2_dish1.mp4": dict(cx=252.6, cy=243.0, cmpp=0.0075799),
}

J_ENTITIES = [
    ("Video1_dish0.mp4", 0, "Video1_dish0_L0"),
    ("Video1_dish0.mp4", 1, "Video1_dish0_L1"),
    ("Video1_dish1.mp4", 0, "Video1_dish1_L0"),
    ("Video1_dish1.mp4", 1, "Video1_dish1_L1"),
    ("Video2_dish0.mp4", 0, "Video2_dish0_L0"),
    ("Video2_dish0.mp4", 1, "Video2_dish0_L1"),
    ("Video2_dish1.mp4", 0, "Video2_dish1_L0"),
    ("Video2_dish1.mp4", 1, "Video2_dish1_L1"),
]

J_DISHES = [
    ("Video1_dish0.mp4", "Video1_dish0", "Dopamine (DA+NoFood)", "#ffd92e", True),
    ("Video1_dish1.mp4", "Video1_dish1", "DA+Food",              "#e0a800", True),
    ("Video2_dish0.mp4", "Video2_dish0", "Veh+NoFood",           "#a86a00", False),
    ("Video2_dish1.mp4", "Video2_dish1", "Veh+Food",             "#5c3d00", True),
]

J_LOWCONF = {"Video1_dish0_L0", "Video1_dish0_L1", "Video1_dish1_L0",
             "Video1_dish1_L1", "Video2_dish1_L0", "Video2_dish1_L1"}

J_CELL_CM = 0.25
J_HM_CELL_CM = 0.10  # original heatmap grid (NOT used in shared figure; see SHARED_HM_CELL_CM)
J_ARENA_AREA = np.pi * J_R ** 2

jdf = pd.read_csv(J_CSV)


def jsmooth(a):
    a = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    if len(a) < 11:
        return a
    return savgol_filter(a, 11, 2)


j_ent = {}
for video, track, key in J_ENTITIES:
    g = jdf[(jdf.video == video) & (jdf.track == track)]
    h = g[g.node == 0].sort_values("frame")
    t = g[g.node == 1].sort_values("frame")
    m = pd.merge(h[["frame", "x", "y"]], t[["frame", "x", "y"]],
                 on="frame", suffixes=("_h", "_t"))
    if len(m) < 100:
        continue
    a = J_CAL[video]
    cmpp = a["cmpp"]
    hx = (jsmooth(m.x_h.to_numpy()) - a["cx"]) * cmpp
    hy = (jsmooth(m.y_h.to_numpy()) - a["cy"]) * cmpp
    tx = (jsmooth(m.x_t.to_numpy()) - a["cx"]) * cmpp
    ty = (jsmooth(m.y_t.to_numpy()) - a["cy"]) * cmpp
    cx = (hx + tx) / 2.0
    cy = (hy + ty) / 2.0
    frames = m.frame.to_numpy()
    tsec = frames * DT
    valid = np.isfinite(cx) & np.isfinite(cy)
    cxv, cyv, tv = cx[valid], cy[valid], tsec[valid]

    r_cm = np.sqrt(cx ** 2 + cy ** 2)

    gx = np.floor(cxv / J_CELL_CM).astype(np.int64)
    gy = np.floor(cyv / J_CELL_CM).astype(np.int64)
    cellid = gx * 100000 + gy
    seen = set()
    cum_cells = np.empty(len(cellid), dtype=np.int64)
    for i, c in enumerate(cellid):
        seen.add(c)
        cum_cells[i] = len(seen)
    cum_area_cm2 = cum_cells * (J_CELL_CM * J_CELL_CM)
    area_grid_cm2 = float(cum_cells[-1] * J_CELL_CM * J_CELL_CM)

    j_ent[key] = dict(
        video=video, cx=cx, cy=cy, cxv=cxv, cyv=cyv,
        r_cm=r_cm, frames=frames,
        exp_t=tv, exp_cum=cum_area_cm2, area_grid_cm2=area_grid_cm2,
        low_conf=key in J_LOWCONF,
    )


def lc_tag(is_low):
    return ""


# ===========================================================================
# CROSS-COHORT SHARED LIMITS (max across BOTH cohorts; applied to both)
# ===========================================================================
# Figure 2: total head path bars (AVERAGED per treatment, one bar each)
# adult: dish1/2/3 single animal; dish4 = mean of its 2 leeches
a_head_vals = [
    a_kin["dish1"]["total_head_cm"],
    a_kin["dish2"]["total_head_cm"],
    a_kin["dish3"]["total_head_cm"],
    float(np.mean([a_kin["dish4_leech0"]["total_head_cm"],
                   a_kin["dish4_leech1"]["total_head_cm"]])),
]
a_head_ns = [1, 1, 1, 2]
j_kin_csv = pd.read_csv(f"{J_MET}/kinematics_summary.csv")
j_kin_csv["cell"] = j_kin_csv["treatment"].str.rsplit(" L", n=1).str[0]
J_TREAT_ORDER = [(t, c, l) for (_v, _s, t, c, l) in J_DISHES]
j_head_vals, j_head_labels, j_head_ns = [], [], []
for treat, tcol, is_low in J_TREAT_ORDER:
    csv_name = {"Dopamine (DA+NoFood)": "Dopamine"}.get(treat, treat)
    sub = j_kin_csv[j_kin_csv["cell"] == csv_name]
    j_head_vals.append(float(sub["total_head_path_cm"].mean()))
    j_head_ns.append(int(sub["total_head_path_cm"].notna().sum()))
    j_head_labels.append(treat + lc_tag(is_low))
HEAD_YMAX = max(a_head_vals + j_head_vals) * 1.10

# Figure 3: area barchart (AVERAGED per treatment, one bar each)
a_area_vals = [
    a_space_results["dish1"]["area_grid_cm2"],
    a_space_results["dish2"]["area_grid_cm2"],
    a_space_results["dish3"]["area_grid_cm2"],
    float(np.mean([a_space_results["dish4_leech0"]["area_grid_cm2"],
                   a_space_results["dish4_leech1"]["area_grid_cm2"]])),
]
a_area_ns = [1, 1, 1, 2]
j_area_vals, j_area_ns = [], []
for (_v, stem, _t, _c, _l) in J_DISHES:
    vv = [j_ent[k]["area_grid_cm2"] for k in (f"{stem}_L0", f"{stem}_L1") if k in j_ent]
    j_area_vals.append(float(np.mean(vv)))
    j_area_ns.append(len(vv))
AREA_YMAX = max(a_area_vals + j_area_vals) * 1.10

# Figure 4: exploration curves (shared y and x)
a_exp_ymax = max(cum.max() for (_t, cum) in a_expcurves.values())
j_exp_ymax = max(j_ent[k]["exp_cum"].max() for k in j_ent if len(j_ent[k]["exp_cum"]))
EXP_YMAX = max(a_exp_ymax, j_exp_ymax) * 1.05
a_exp_xmax = max(tt.max() for (tt, _c) in a_expcurves.values()) / 60.0
j_exp_xmax = max(j_ent[k]["exp_t"].max() for k in j_ent if len(j_ent[k]["exp_t"])) / 60.0
EXP_XMAX = max(a_exp_xmax, j_exp_xmax)

# Figure 5: radial timeline (shared y = larger dish radius 4.5; shared x = max time)
a_rad_xmax = max((v["frames"] / FPS / 60.0).max() for v in a_thig.values())
j_rad_xmax = max((j_ent[k]["frames"] / FPS / 60.0).max() for k in j_ent)
RAD_XMAX = max(a_rad_xmax, j_rad_xmax)
RAD_YMAX = R_ADULT  # 4.5 cm (larger dish radius)

# Figure 7 (occupancy heatmaps): AVERAGE per treatment -> ONE panel per treatment
# = MEAN of the treatment's animals' dwell grids. Shared color scale (LogNorm)
# across ALL panels of BOTH cohorts, using SHARED_HM_CELL_CM for both.
HM_VMAX = 1
a_hm_edges = np.arange(-SHARED_HALF, SHARED_HALF + SHARED_HM_CELL_CM, SHARED_HM_CELL_CM)

# ADULT: per-animal dwell grids, then average per treatment.
a_anim_hist = {}
for n in [x for x in A_ORDER if x in a_cmcoords]:
    cx, cy = a_cmcoords[n]
    H, _, _ = np.histogram2d(cx, cy, bins=[a_hm_edges, a_hm_edges])
    a_anim_hist[n] = H
# Adult treatment -> list of animal keys (in A_ORDER).
A_TREAT_PANELS = [
    ("Veh+Food",   ["dish1"]),
    ("DA+Food",    ["dish2"]),
    ("Veh+NoFood", ["dish3"]),
    ("DA+NoFood",  ["dish4_leech0", "dish4_leech1"]),
]
a_hists = {}
for treat, keys in A_TREAT_PANELS:
    keys = [k for k in keys if k in a_anim_hist]
    if not keys:
        continue
    Hmean = np.mean([a_anim_hist[k] for k in keys], axis=0)
    a_hists[treat] = dict(H=Hmean, n=len(keys))
    HM_VMAX = max(HM_VMAX, Hmean.max())

# JUVENILE: per-animal dwell grids, then average per treatment (the 2 leeches).
j_anim_hist = {}
for (_v, stem, _t, _c, _l) in J_DISHES:
    for ln in ("L0", "L1"):
        key = f"{stem}_{ln}"
        if key not in j_ent:
            continue
        v = j_ent[key]
        H, _, _ = np.histogram2d(v["cxv"], v["cyv"], bins=[a_hm_edges, a_hm_edges])
        j_anim_hist[key] = H
j_hists = {}
for (_v, stem, treat, _c, is_low) in J_DISHES:
    keys = [f"{stem}_{ln}" for ln in ("L0", "L1") if f"{stem}_{ln}" in j_anim_hist]
    if not keys:
        continue
    Hmean = np.mean([j_anim_hist[k] for k in keys], axis=0)
    j_hists[treat] = dict(H=Hmean, n=len(keys), low=is_low)
    HM_VMAX = max(HM_VMAX, Hmean.max())

# ===========================================================================
# PER-TREATMENT FRAME-ALIGNED AVERAGES (radial timeline + exploration curves)
# ===========================================================================

def avg_by_frame(series_list):
    """Average a list of (frames, values) onto the union of frame indices.

    Aligns by frame: at each frame, take the mean over the series that have a
    value there. Returns (frames_sorted, mean_values).
    """
    series_list = [(np.asarray(f), np.asarray(v)) for f, v in series_list if len(f)]
    if not series_list:
        return np.array([]), np.array([])
    if len(series_list) == 1:
        f, v = series_list[0]
        return f.astype(float), v.astype(float)
    all_frames = np.unique(np.concatenate([f for f, _ in series_list]))
    acc = np.zeros(len(all_frames))
    cnt = np.zeros(len(all_frames))
    for f, v in series_list:
        idx = np.searchsorted(all_frames, f)
        finite = np.isfinite(v)
        np.add.at(acc, idx[finite], v[finite])
        np.add.at(cnt, idx[finite], 1.0)
    cnt[cnt == 0] = np.nan
    return all_frames.astype(float), acc / cnt


# Adult radial averages per treatment (one line each).
A_RAD_AVG = []
for treat, keys in A_TREAT_PANELS:
    keys = [k for k in keys if k in a_thig]
    if not keys:
        continue
    f, r = avg_by_frame([(a_thig[k]["frames"], a_thig[k]["r_cm"]) for k in keys])
    A_RAD_AVG.append((treat, A_TREAT_COLOR[treat], f, r, len(keys)))

# Juvenile radial averages per treatment.
J_RAD_AVG = []
for (_v, stem, treat, tcol, is_low) in J_DISHES:
    keys = [f"{stem}_{ln}" for ln in ("L0", "L1") if f"{stem}_{ln}" in j_ent]
    if not keys:
        continue
    f, r = avg_by_frame([(j_ent[k]["frames"], j_ent[k]["r_cm"]) for k in keys])
    J_RAD_AVG.append((treat, tcol, f, r, len(keys), is_low))

# Adult exploration-curve averages per treatment (cumulative area vs frame).
# exp curves are stored as (time_s, cum_area); convert time to frame index.
A_EXP_AVG = []
for treat, keys in A_TREAT_PANELS:
    keys = [k for k in keys if k in a_expcurves]
    if not keys:
        continue
    series = []
    for k in keys:
        tt, cum = a_expcurves[k]
        series.append((np.round(tt * FPS).astype(np.int64), cum))
    fr, cum = avg_by_frame(series)
    A_EXP_AVG.append((treat, A_TREAT_COLOR[treat], fr / FPS, cum, len(keys)))

# Juvenile exploration-curve averages per treatment.
J_EXP_AVG = []
for (_v, stem, treat, tcol, is_low) in J_DISHES:
    keys = [f"{stem}_{ln}" for ln in ("L0", "L1") if f"{stem}_{ln}" in j_ent]
    if not keys:
        continue
    series = []
    for k in keys:
        v = j_ent[k]
        series.append((np.round(v["exp_t"] * FPS).astype(np.int64), v["exp_cum"]))
    fr, cum = avg_by_frame(series)
    J_EXP_AVG.append((treat, tcol, fr / FPS, cum, len(keys), is_low))


# ===========================================================================
# RENDER ADULT FIGURES
# ===========================================================================

# ---- design_2x2_effects_adult / juvenile : reuse design_2x2_shared_y logic ----
VEH_C, DA_C = "#999999", "#7B3FA0"
CELL_ADULT = {
    "Veh+Food": ("Veh", "Food"), "DA+Food": ("DA", "Food"),
    "Veh+NoFood": ("Veh", "NoFood"), "DA+NoFood": ("DA", "NoFood"),
}
CELL_JUV = {
    "Veh+Food": ("Veh", "Food"), "DA+Food": ("DA", "Food"),
    "Veh+NoFood": ("Veh", "NoFood"), "Dopamine": ("DA", "NoFood"),
}
D2_PANELS = [
    ("kinematics_summary.csv", "pct_time_moving", None, "% time moving"),
    ("kinematics_summary.csv", "total_head_path_cm", "total_tail_path_cm",
     "head:tail path (foraging probe)"),
    ("spaceuse_metrics.csv", "area_grid_cm2", None, "area explored (cm2)"),
    ("turning_metrics.csv", "turning_per_min_deg", None, "turning (deg/min)"),
]


def base_treat(s):
    return str(s).rsplit(" L", 1)[0]


def cell_means(metrics_dir, cell_map):
    out = []
    for csv, col, den, _ in D2_PANELS:
        d = pd.read_csv(f"{metrics_dir}/{csv}")
        d["cell"] = d["treatment"].map(base_treat).map(cell_map)
        d = d.dropna(subset=["cell"])
        d["_v"] = d[col] / d[den] if den else d[col]
        out.append(d.groupby("cell")["_v"].mean().to_dict())
    return out


def cell_n(metrics_dir, cell_map):
    # n per cell = number of leeches (entities) per base treatment,
    # keyed by the (drug, food) tuple
    d = pd.read_csv(f"{metrics_dir}/kinematics_summary.csv")
    d["cell"] = d["treatment"].map(base_treat).map(cell_map)
    d = d.dropna(subset=["cell"])
    return {k: int(v) for k, v in d.groupby("cell").size().to_dict().items()}


d2_adult = cell_means(A_MET, CELL_ADULT)
d2_juv = cell_means(J_MET, CELL_JUV)
n_adult = cell_n(A_MET, CELL_ADULT)
n_juv = cell_n(J_MET, CELL_JUV)
D2_YMAX = []
for pi in range(len(D2_PANELS)):
    vals = list(d2_adult[pi].values()) + list(d2_juv[pi].values())
    D2_YMAX.append(max(vals) * 1.18)

foods = ["NoFood", "Food"]
centers = np.array([0.0, 1.0])
w = 0.4


def make_2x2(data, cohort_label, fname, cell_n_map, bar_colors):
    veh_c, da_c = bar_colors
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, panel, ylim in zip(axes, D2_PANELS, D2_YMAX):
        title = panel[3]
        for di, (drug, col) in enumerate([("Veh", veh_c), ("DA", da_c)]):
            vals = [data[D2_PANELS.index(panel)].get((drug, f), np.nan) for f in foods]
            xpos = centers + (di - 0.5) * w
            ax.bar(xpos, vals, w, color=col,
                   label="Vehicle" if drug == "Veh" else "Dopamine")
            for x, v in zip(xpos, vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.3g}", (x, v), ha="center", va="bottom",
                                fontsize=9, xytext=(0, 2),
                                textcoords="offset points")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, ylim)
        ax.set_xticks([centers[0] - w / 2, centers[0] + w / 2,
                       centers[1] - w / 2, centers[1] + w / 2])
        # x-tick order: [NoFood-Veh, NoFood-DA, Food-Veh, Food-DA]
        ax.set_xticklabels([
            f"Veh\nn={cell_n_map.get(('Veh', 'NoFood'), 0)}",
            f"DA\nn={cell_n_map.get(('DA', 'NoFood'), 0)}",
            f"Veh\nn={cell_n_map.get(('Veh', 'Food'), 0)}",
            f"DA\nn={cell_n_map.get(('DA', 'Food'), 0)}",
        ])
        ax.tick_params(axis="x", length=0)
        ax.text(0.25, -0.26, "NoFood", transform=ax.transAxes, ha="center",
                va="top", fontsize=11)
        ax.text(0.75, -0.26, "Food", transform=ax.transAxes, ha="center",
                va="top", fontsize=11)
        ax.set_title(title, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True, ncol=2)
    fig.suptitle(f"2x2 design: dopamine x food effects ({cohort_label}); "
                 f"{NOTE} for direct comparison", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(f"{OUT}/{fname}", dpi=130)
    plt.close(fig)


# 2x2 bars in the cohort hue: Vehicle = light shade, Dopamine = dark shade
make_2x2(d2_adult, "adults", "design_2x2_effects_adult.png", n_adult, ("#9ecae1", "#08519c"))
make_2x2(d2_juv, "juveniles", "design_2x2_effects_juvenile.png", n_juv, ("#f2d24b", "#8c6d00"))

# ---- kinematics_total_head_path_bar : shared y = HEAD_YMAX, AVERAGED per treatment ----
# Bars recolored to TREATMENT shade (cohort = hue).
A_BAR_TREATS = ["Veh+Food", "DA+Food", "Veh+NoFood", "DA+NoFood"]
A_BAR_COLORS = [A_TREAT_COLOR[t] for t in A_BAR_TREATS]
J_BAR_COLORS = [tcol for (_v, _s, _t, tcol, _l) in J_DISHES]
# ADULT
fig, ax = plt.subplots(figsize=(9, 5))
labels6 = ["dish1\nVeh+Food", "dish2\nDA+Food", "dish3\nVeh+NoFood",
           "dish4\nDA+NoFood"]
vals6 = a_head_vals
positions6 = list(range(len(labels6)))
ax.bar(positions6, vals6, 0.6, color=A_BAR_COLORS, edgecolor="black",
       linewidth=0.6)
for i, v in enumerate(vals6):
    ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(positions6)
ax.set_xticklabels(labels6, fontsize=9)
ax.tick_params(axis="x", pad=4)
for p, n in zip(positions6, a_head_ns):
    ax.annotate(f"n={n}", xy=(p, 0), xytext=(0, -52),
                textcoords="offset points", ha="center", va="top", fontsize=8)
ax.set_ylabel("total HEAD path (cm)")
ax.set_ylim(0, HEAD_YMAX)
ax.set_title("Kinematics: total head path averaged per treatment (whole clip, cm)\n"
             f"{NOTE}")
fig.tight_layout()
fig.savefig(f"{OUT}/kinematics_total_head_path_bar_adult.png", dpi=120)
plt.close(fig)

# JUVENILE
fig, ax = plt.subplots(figsize=(9, 5))
positionsj = list(range(len(j_head_labels)))
bars = ax.bar(positionsj, j_head_vals, 0.6, color=J_BAR_COLORS,
              edgecolor="black", linewidth=0.6)
for b, v in zip(bars, j_head_vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
            ha="center", va="bottom", fontsize=9)
ax.set_xticks(positionsj)
ax.set_xticklabels(j_head_labels, rotation=15, ha="right", fontsize=9)
ax.tick_params(axis="x", pad=4)
for p, n in zip(positionsj, j_head_ns):
    ax.annotate(f"n={n}", xy=(p, 0), xytext=(0, -64),
                textcoords="offset points", ha="center", va="top", fontsize=8)
ax.set_ylabel("total HEAD path (cm)")
ax.set_ylim(0, HEAD_YMAX)
ax.set_title("Kinematics: total head path averaged per treatment\n"
             f"(mean of the 2 leeches per treatment, whole clip); {NOTE}")
fig.tight_layout()
fig.savefig(f"{OUT}/kinematics_total_head_path_bar_juvenile.png", dpi=130)
plt.close(fig)

# ---- spaceuse_area_barchart : shared y = AREA_YMAX, AVERAGED per treatment ----
# ADULT
fig, ax = plt.subplots(figsize=(9, 5.5))
a_area_labels = ["dish1\nVeh+Food", "dish2\nDA+Food", "dish3\nVeh+NoFood",
                 "dish4\nDA+NoFood"]
positions = list(range(len(a_area_labels)))
ax.bar(positions, a_area_vals, 0.6, color=A_BAR_COLORS, edgecolor="black",
       linewidth=0.6)
for p, hgt in zip(positions, a_area_vals):
    ax.text(p, hgt, f"{hgt:.1f}", ha="center", va="bottom", fontsize=9)
ax.axhline(np.pi * A_R ** 2, color="k", ls=":", lw=1, label="full arena (63.6 cm$^2$)")
ax.set_xticks(positions)
ax.set_xticklabels(a_area_labels, fontsize=9)
ax.tick_params(axis="x", pad=4)
for p, n in zip(positions, a_area_ns):
    ax.annotate(f"n={n}", xy=(p, 0), xytext=(0, -52),
                textcoords="offset points", ha="center", va="top", fontsize=8)
ax.set_ylabel("area explored (cm$^2$, 0.25 cm cells)")
ax.set_ylim(0, AREA_YMAX)
ax.set_title("Space use averaged per treatment (one bar per treatment)\n"
             f"{NOTE}")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/spaceuse_area_barchart_adult.png", dpi=120)
plt.close(fig)

# JUVENILE
fig, ax = plt.subplots(figsize=(11, 5.5))
xticks, xlabels = [], []
for gi, (video, stem, treat, tcol, is_low) in enumerate(J_DISHES):
    xticks.append(gi)
    xlabels.append(f"{treat}{lc_tag(is_low)}")
ax.bar(xticks, j_area_vals, 0.6, color=J_BAR_COLORS, edgecolor="black",
       linewidth=0.6)
for p, hgt in zip(xticks, j_area_vals):
    ax.text(p, hgt, f"{hgt:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=15, ha="right", fontsize=9)
ax.tick_params(axis="x", length=0, pad=4)
for p, n in zip(xticks, j_area_ns):
    ax.annotate(f"n={n}", xy=(p, 0), xytext=(0, -52),
                textcoords="offset points", ha="center", va="top", fontsize=8)
ax.set_ylabel("area explored (cm$^2$, 0.25 cm grid)")
ax.set_ylim(0, AREA_YMAX)
ax.set_title("Area explored averaged per treatment (one bar per treatment)\n"
             + NOTE)
fig.tight_layout()
fig.savefig(f"{OUT}/spaceuse_area_barchart_juvenile.png", dpi=130)
plt.close(fig)

# ---- spaceuse_exploration_curves : shared y = EXP_YMAX, shared x = EXP_XMAX ----
# AVERAGED per treatment -> one cumulative-area line per treatment.
# ADULT
fig, ax = plt.subplots(figsize=(9, 6))
for i, (treat, col, tt, cum, n) in enumerate(A_EXP_AVG):
    ax.plot(tt / 60.0, cum, color=col, label=treat, lw=2,
            linestyle=["-", "--", "-.", ":"][i % 4])
ax.set_xlabel("time (min)")
ax.set_ylabel("cumulative area explored (cm$^2$)")
ax.set_xlim(0, EXP_XMAX)
ax.set_ylim(0, EXP_YMAX)
ax.set_title("Exploration curves: cumulative unique area vs time, averaged per treatment "
             "(centroid, 0.25 cm cells)\n" + NOTE)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/spaceuse_exploration_curves_adult.png", dpi=120)
plt.close(fig)

# JUVENILE
fig, ax = plt.subplots(figsize=(10, 6))
for i, (treat, tcol, tt, cum, n, is_low) in enumerate(J_EXP_AVG):
    ax.plot(tt / 60.0, cum, color=tcol, lw=2, alpha=0.95,
            label=f"{treat}{lc_tag(is_low)}",
            linestyle=["-", "--", "-.", ":"][i % 4])
ax.set_xlabel("time (min)")
ax.set_ylabel("cumulative area explored (cm$^2$)")
ax.set_xlim(0, EXP_XMAX)
ax.set_ylim(0, EXP_YMAX)
ax.set_title("Exploration curves: cumulative unique area vs time, averaged per treatment "
             "(centroid, 0.25 cm cells)\n" + NOTE)
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/spaceuse_exploration_curves_juvenile.png", dpi=130)
plt.close(fig)

# ---- thigmotaxis_radial_timeline : shared y = 4.5 cm, shared x = RAD_XMAX ----
# AVERAGED per treatment -> one radial line per treatment.
# ADULT
fig, ax = plt.subplots(figsize=(11, 5))
for i, (treat, col, f, r, n) in enumerate(A_RAD_AVG):
    ts = f / FPS / 60.0
    ax.plot(ts, r, lw=1.0, color=col, label=treat,
            linestyle=["-", "--", "-.", ":"][i % 4])
ax.axhline(A_R, color="k", ls=":", lw=0.8, label="dish wall (4.5 cm)")
ax.set_ylim(0, RAD_YMAX)
ax.set_xlim(0, RAD_XMAX)
ax.set_xlabel("time (min)")
ax.set_ylabel("radial position r (cm)")
ax.set_title("Thigmotaxis: radial distance from center over time, averaged per treatment (cm)\n"
             + NOTE)
leg = ax.legend(fontsize=8, loc="upper right")
for line in leg.get_lines():
    line.set_linewidth(2.0)
fig.tight_layout()
fig.savefig(f"{OUT}/thigmotaxis_radial_timeline_adult.png", dpi=120)
plt.close(fig)

# JUVENILE
fig, ax = plt.subplots(figsize=(12, 6))
for i, (treat, tcol, f, r, n, is_low) in enumerate(J_RAD_AVG):
    ts = f / FPS / 60.0
    ax.plot(ts, r, color=tcol, lw=1.0, alpha=0.95,
            label=f"{treat}{lc_tag(is_low)}",
            linestyle=["-", "--", "-.", ":"][i % 4])
ax.axhline(J_R, color="k", ls=":", lw=0.8, label="dish wall (1.75 cm)")
ax.set_ylim(0, RAD_YMAX)
ax.set_xlim(0, RAD_XMAX)
ax.set_xlabel("time (min)")
ax.set_ylabel("radial position r (cm)")
ax.set_title("Thigmotaxis: radial distance from center over time, averaged per treatment\n"
             + NOTE)
leg = ax.legend(fontsize=8, ncol=2, loc="upper right")
for line in leg.get_lines():
    line.set_linewidth(2.0)
fig.tight_layout()
fig.savefig(f"{OUT}/thigmotaxis_radial_timeline_juvenile.png", dpi=130)
plt.close(fig)

# ---- thigmotaxis_arena_scatter : shared symmetric (-5,5) cm; true dish circles ----
# One panel PER DISH; both animals overlaid in the SAME treatment shade.
# ADULT (one panel per dish; dish4 overlays its 2 leeches in the DA+NoFood shade)
A_SCATTER_PANELS = [
    (1, "Veh+Food",   ["dish1"]),
    (2, "DA+Food",    ["dish2"]),
    (3, "Veh+NoFood", ["dish3"]),
    (4, "DA+NoFood",  ["dish4_leech0", "dish4_leech1"]),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, (dish, treat, keys) in zip(axes, A_SCATTER_PANELS):
    col = A_TREAT_COLOR[treat]
    shades = two_shades(col)
    n_animals = 0
    for li, name in enumerate(keys):
        if name not in a_thig:
            continue
        v = a_thig[name]
        ax.scatter(v["cx"], v["cy"], s=0.3, alpha=0.16, color=shades[li])
        n_animals += 1
    if n_animals == 2:
        ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=6, color=shades[0], label="animal 1"),
                           Line2D([0], [0], marker="o", ls="", ms=6, color=shades[1], label="animal 2")],
                  loc="upper right", fontsize=7, framealpha=0.9)
    ax.add_patch(plt.Circle((0, 0), A_R, fill=False, color="black", lw=1.0))
    ax.add_patch(plt.Circle((0, 0), 0.7 * A_R, fill=False, color="#666666",
                            ls="--", lw=1.0))
    ax.set_xlim(-SHARED_HALF, SHARED_HALF)
    ax.set_ylim(-SHARED_HALF, SHARED_HALF)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (cm)")
    ax.set_title(f"{treat} (n={n_animals})", fontsize=10)
axes[0].set_ylabel("y (cm)")
fig.suptitle("Centroid cloud in cm: 4.5 cm dish boundary (black), inner annulus 3.15 cm "
             "(dashed grey); one panel per dish (the 2 animals as two shades)\n"
             "shared frame (-5, 5) cm; " + NOTE)
fig.tight_layout()
fig.savefig(f"{OUT}/thigmotaxis_arena_scatter_adult.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# JUVENILE (one panel per dish; both leeches overlaid in the treatment shade)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
for ax, (video, stem, treat, tcol, is_low) in zip(axes, J_DISHES):
    shades = two_shades(tcol)
    n_animals = 0
    for li, lname in enumerate(("L0", "L1")):
        key = f"{stem}_{lname}"
        if key not in j_ent:
            continue
        v = j_ent[key]
        ax.scatter(v["cx"], v["cy"], s=0.3, alpha=0.16, color=shades[li])
        n_animals += 1
    if n_animals == 2:
        ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=6, color=shades[0], label="animal 1"),
                           Line2D([0], [0], marker="o", ls="", ms=6, color=shades[1], label="animal 2")],
                  loc="upper right", fontsize=7, framealpha=0.9)
    ax.add_patch(plt.Circle((0, 0), J_R, fill=False, color="black", lw=1.0))
    ax.add_patch(plt.Circle((0, 0), 0.7 * J_R, fill=False,
                            color="#666666", ls="--", lw=1.0))
    ax.plot(0, 0, "x", color="black", ms=5)
    ax.set_xlim(-SHARED_HALF, SHARED_HALF)
    ax.set_ylim(-SHARED_HALF, SHARED_HALF)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (cm)")
    ax.set_title(f"{treat} (n={n_animals})", fontsize=10)
axes[0].set_ylabel("y (cm)")
fig.suptitle("Centroid clouds per dish (the 2 animals shown as two shades of the treatment color)\n"
             "outer solid circle = 1.75 cm dish boundary; dashed grey = inner annulus "
             "(1.225 cm); shared frame (-5, 5) cm; " + NOTE, y=1.08)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/thigmotaxis_arena_scatter_juvenile.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- spaceuse_occupancy_heatmaps : AVERAGED per treatment (mean dwell grid),
#       shared (-5,5) cm extent, 0.25 cm cells, shared LogNorm (vmin=1, vmax=HM_VMAX)
#       across BOTH cohorts. Cohort colormap: adult "Blues", juvenile "YlOrBr". ----
shared_norm = LogNorm(vmin=1, vmax=HM_VMAX)

# ADULT (one panel per treatment = mean of the treatment's animals' dwell grids)
a_hm_treats = [t for (t, _k) in A_TREAT_PANELS if t in a_hists]
fig, axs = plt.subplots(1, len(a_hm_treats),
                        figsize=(3.2 * len(a_hm_treats) + 1.2, 3.6))
im = None
for ax, treat in zip(axs, a_hm_treats):
    H = a_hists[treat]["H"].T
    Hm = np.ma.masked_less(H, 1)
    im = ax.pcolormesh(a_hm_edges, a_hm_edges, Hm, norm=shared_norm, cmap="Blues")
    ax.add_patch(mpatches.Circle((0, 0), A_R, fill=False, color="black", lw=1.2))
    ax.set_aspect("equal")
    ax.set_xlim(-SHARED_HALF, SHARED_HALF)
    ax.set_ylim(-SHARED_HALF, SHARED_HALF)
    ax.set_xlabel("cm")
    ax.set_title(f"{treat}\n(n={a_hists[treat]['n']})", fontsize=10)
axs[0].set_ylabel("cm")
cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
cbar.set_label("mean dwell (frames, log)")
fig.suptitle("Occupancy averaged per treatment: shared frame (-5, 5) cm, 0.25 cm cells, "
             "shared log color scale\ndish circle at 4.5 cm (Blues); " + NOTE, fontsize=12)
fig.savefig(f"{OUT}/spaceuse_occupancy_heatmaps_adult.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# JUVENILE (one panel per treatment = mean of the treatment's 2 leeches' dwell grids)
j_hm_treats = [t for (_v, _s, t, _c, _l) in J_DISHES if t in j_hists]
fig, axs = plt.subplots(1, len(j_hm_treats), figsize=(3.2 * len(j_hm_treats) + 1.2, 3.7))
im = None
for ax, treat in zip(axs, j_hm_treats):
    H = j_hists[treat]["H"].T
    Hm = np.ma.masked_less(H, 1)
    im = ax.pcolormesh(a_hm_edges, a_hm_edges, Hm, norm=shared_norm, cmap="YlOrBr")
    ax.add_patch(mpatches.Circle((0, 0), J_R, fill=False, color="black", lw=1.0))
    ax.set_aspect("equal")
    ax.set_xlim(-SHARED_HALF, SHARED_HALF)
    ax.set_ylim(-SHARED_HALF, SHARED_HALF)
    ax.set_xlabel("cm")
    is_low = j_hists[treat]["low"]
    ax.set_title(f"{treat}{lc_tag(is_low)}\n(n={j_hists[treat]['n']})", fontsize=9)
axs[0].set_ylabel("cm")
cbar = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
cbar.set_label("mean dwell (frames, log)")
fig.suptitle("Occupancy averaged per treatment: shared frame (-5, 5) cm, 0.25 cm cells, "
             "shared log color scale\ndish circle at 1.75 cm (YlOrBr); " + NOTE, fontsize=12)
fig.savefig(f"{OUT}/spaceuse_occupancy_heatmaps_juvenile.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
print("=== SHARED LIMITS ===")
print(f"design_2x2 per-panel ymax: " +
      ", ".join(f"{D2_PANELS[i][3]}={D2_YMAX[i]:.2f}" for i in range(len(D2_PANELS))))
print(f"kinematics_total_head_path_bar ymax: {HEAD_YMAX:.2f} cm")
print(f"spaceuse_area_barchart ymax: {AREA_YMAX:.2f} cm2")
print(f"spaceuse_exploration_curves: ymax={EXP_YMAX:.2f} cm2, xmax={EXP_XMAX:.2f} min")
print(f"thigmotaxis_radial_timeline: ymax={RAD_YMAX:.2f} cm, xmax={RAD_XMAX:.2f} min")
print(f"thigmotaxis_arena_scatter: x/y in (-{SHARED_HALF}, {SHARED_HALF}) cm")
print(f"spaceuse_occupancy_heatmaps: (-{SHARED_HALF}, {SHARED_HALF}) cm, "
      f"cell={SHARED_HM_CELL_CM} cm, LogNorm vmax={HM_VMAX:.0f}")
print("DONE")
