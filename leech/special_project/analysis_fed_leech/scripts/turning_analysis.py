import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os, json

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
OUT = os.path.join(BASE, "analysis_fed_leech/plots")
MET = os.path.join(BASE, "analysis_fed_leech/metrics")
FPS = 149784 / 4994.0  # 29.99

# Per-dish arena geometry (REAL-WORLD SCALE). Dish = 9 cm diameter, radius 4.5 cm.
ARENA = {
    "dish1": {"cx": 303.0, "cy": 295.0, "cmpp": 0.0200},
    "dish2": {"cx": 290.0, "cy": 311.0, "cmpp": 0.0205},
    "dish3": {"cx": 292.0, "cy": 284.0, "cmpp": 0.0202},
    "dish4": {"cx": 311.0, "cy": 293.0, "cmpp": 0.0206},
}
RADIUS_CM = 4.5
# median body length (cm) per dish, for reference only
BL_CM = {"dish1": 1.14, "dish2": 1.13, "dish3": 0.88, "dish4": 0.50}

print("loading...")
df = pd.read_csv(os.path.join(BASE, "DL/all_predictions.csv"))
df["dish"] = df["video"].str.extract(r"(dish\d)")

def get_track(d, track):
    sub = df[(df["dish"] == d) & (df["track"] == track)]
    if len(sub) == 0:
        return None
    h = sub[sub["node"] == 0].sort_values("frame")
    t = sub[sub["node"] == 1].sort_values("frame")
    m = h[["frame", "time_s", "x", "y"]].merge(
        t[["frame", "x", "y"]], on="frame", suffixes=("_h", "_t"))
    return m

def smooth(a, w=11):
    a = np.asarray(a, float)
    if len(a) < w:
        return a
    return savgol_filter(a, w, 2)

def angdiff(a):
    d = np.diff(a)
    return (d + np.pi) % (2 * np.pi) - np.pi

# 5 entities. dish4 leeches are low-confidence (PCK 39%).
ENTITIES = [
    ("dish1", 0, "dish1"),
    ("dish2", 0, "dish2"),
    ("dish3", 0, "dish3"),
    ("dish4", 0, "dish4_leech0"),
    ("dish4", 1, "dish4_leech1"),
]
LOWCONF = {"dish4_leech0", "dish4_leech1"}
DISPLAY = {"dish1": "Veh+Food", "dish2": "DA+Food", "dish3": "Veh+NoFood",
           "dish4_leech0": "DA+NoFood L0", "dish4_leech1": "DA+NoFood L1"}

WIN = 11  # Savitzky-Golay window (~0.37 s), polyorder 2, applied per coordinate in px

results = {}
turn_timeseries = {}   # label -> (bin_centers_min, turning_deg_per_min, speed_cm_s)
thirds_data = {}       # label -> {"speed":[...], "turn":[...]}
heading_timeseries = {}  # label -> (time_min, heading_deg)

for d, tr, label in ENTITIES:
    m = get_track(d, tr)
    if m is None:
        continue
    cmpp = ARENA[d]["cmpp"]
    cx0 = ARENA[d]["cx"]; cy0 = ARENA[d]["cy"]
    n = len(m)
    t = m["time_s"].values

    # smooth coords in px, then convert to cm relative to arena center
    hx = (smooth(m["x_h"].values, WIN) - cx0) * cmpp
    hy = (smooth(m["y_h"].values, WIN) - cy0) * cmpp
    tx = (smooth(m["x_t"].values, WIN) - cx0) * cmpp
    ty = (smooth(m["y_t"].values, WIN) - cy0) * cmpp
    cx = (hx + tx) / 2.0
    cy = (hy + ty) / 2.0

    # heading = atan2(head - tail)
    heading = np.arctan2(hy - ty, hx - tx)   # radians
    heading_deg = np.degrees(np.unwrap(heading))

    dt = np.diff(t)
    dt[dt <= 0] = 1.0 / FPS
    dhead = angdiff(heading)                 # radians
    abs_dhead_deg = np.abs(np.degrees(dhead))
    angvel = np.degrees(dhead) / dt          # deg/s

    # centroid speed in cm/s
    dcx = np.diff(cx); dcy = np.diff(cy)
    step_cm = np.sqrt(dcx**2 + dcy**2)
    speed_cm_s = step_cm / dt

    total_turn_deg = np.sum(abs_dhead_deg)
    duration_min = (t[-1] - t[0]) / 60.0
    turn_per_min = total_turn_deg / duration_min

    # reorientation events: accumulated |heading change| over 0.5 s window exceeds 45 deg
    win_f = max(1, int(round(0.5 * FPS)))
    roll = pd.Series(abs_dhead_deg).rolling(win_f, min_periods=1).sum().values
    over = roll > 45.0
    events = int(np.sum((over[1:]) & (~over[:-1])))
    reorient_per_min = events / duration_min

    # tortuosity over sliding 10 s windows (net displacement cm / total path cm)
    win_s = 10.0
    wf = int(round(win_s * FPS))
    tort_vals = []
    net_list = []; path_list = []
    for start in range(0, len(cx) - wf, wf):
        sx = cx[start:start + wf]; sy = cy[start:start + wf]
        path = np.sum(np.sqrt(np.diff(sx)**2 + np.diff(sy)**2))
        net = np.sqrt((sx[-1] - sx[0])**2 + (sy[-1] - sy[0])**2)
        if path > 1e-6:
            tort_vals.append(net / path)     # straightness index (1=straight, 0=tortuous)
            net_list.append(net); path_list.append(path)
    straightness = float(np.nanmean(tort_vals)) if tort_vals else np.nan
    mean_net_cm = float(np.nanmean(net_list)) if net_list else np.nan
    mean_path_cm = float(np.nanmean(path_list)) if path_list else np.nan

    # per-minute time series
    tmin = t[1:] / 60.0
    nbins = int(np.ceil(tmin[-1]))
    bins = np.arange(0, nbins + 1)
    idx = np.digitize(tmin, bins) - 1
    turn_binned = np.array([np.sum(abs_dhead_deg[idx == b]) for b in range(nbins)])
    speed_binned = np.array([np.nanmean(speed_cm_s[idx == b]) if np.any(idx == b) else np.nan
                             for b in range(nbins)])
    turn_timeseries[label] = (np.arange(nbins) + 0.5, turn_binned, speed_binned)

    # heading over time (downsample for plotting)
    ds = max(1, n // 4000)
    heading_timeseries[label] = (t[::ds] / 60.0, heading_deg[::ds])

    # early / mid / late thirds
    third = n // 3
    seg_idx = [(0, third), (third, 2 * third), (2 * third, 3 * third)]
    seg_speed = []; seg_turn = []
    for a, b in seg_idx:
        aa = min(a, len(speed_cm_s)); bb = min(b, len(speed_cm_s))
        seg_speed.append(float(np.nanmean(speed_cm_s[aa:bb])) if bb > aa else np.nan)
        dur_m = (t[min(b, n - 1)] - t[a]) / 60.0
        seg_turn.append(float(np.sum(abs_dhead_deg[aa:bb]) / dur_m) if dur_m > 0 else np.nan)
    thirds_data[label] = {"speed": seg_speed, "turn": seg_turn}

    results[label] = {
        "label": label,
        "treatment": DISPLAY[label],
        "dish": d, "track": tr, "low_confidence": label in LOWCONF,
        "n_frames": int(n),
        "body_len_cm": BL_CM[d],
        "cm_per_px": cmpp,
        "duration_min": round(duration_min, 2),
        "total_turning_deg": round(total_turn_deg, 1),
        "turning_per_min_deg": round(turn_per_min, 1),
        "mean_abs_angvel_deg_s": round(float(np.nanmean(np.abs(angvel))), 3),
        "median_abs_angvel_deg_s": round(float(np.nanmedian(np.abs(angvel))), 3),
        "reorient_events_per_min": round(reorient_per_min, 3),
        "mean_straightness_10s": round(straightness, 4),
        "mean_tortuosity_10s": round(1.0 / straightness if straightness > 0 else np.nan, 3),
        "mean_net_disp_cm_10s": round(mean_net_cm, 3),
        "mean_path_len_cm_10s": round(mean_path_cm, 3),
        "mean_speed_cm_s": round(float(np.nanmean(speed_cm_s)), 4),
        "speed_early_cm_s": round(seg_speed[0], 4),
        "speed_mid_cm_s": round(seg_speed[1], 4),
        "speed_late_cm_s": round(seg_speed[2], 4),
        "turn_early_deg_min": round(seg_turn[0], 1),
        "turn_mid_deg_min": round(seg_turn[1], 1),
        "turn_late_deg_min": round(seg_turn[2], 1),
    }
    print(label, "done")

labels = list(results.keys())
colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
cmap = {l: c for l, c in zip(labels, colors)}

def style(l):
    return {"ls": "--" if l in LOWCONF else "-", "color": cmap[l]}

def disp(l):
    return DISPLAY[l] + " (low-conf)" if l in LOWCONF else DISPLAY[l]

# Fig 1: turning rate over session (1-min bins) + heading-over-time subplot
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
for l in labels:
    x, turn, _ = turn_timeseries[l]
    axes[0].plot(x, turn, label=disp(l), alpha=0.85, lw=1.2, **style(l))
axes[0].set_ylabel("Total turning (deg / min)")
axes[0].set_title("Turning rate over session (1-min bins; Savitzky-Golay w=11 on px coords)")
axes[0].legend(fontsize=8)
for l in labels:
    tmn, hd = heading_timeseries[l]
    axes[1].plot(tmn, hd, label=disp(l), alpha=0.8, lw=0.7, **style(l))
axes[1].set_xlabel("Time (min)")
axes[1].set_ylabel("Heading, unwrapped (deg)")
axes[1].set_title("Heading over time (head relative to tail)")
axes[1].legend(fontsize=8)
fig.tight_layout()
f1 = os.path.join(OUT, "turning_rate_over_time.png"); fig.savefig(f1, dpi=120); plt.close(fig)

# Fig 3: tortuosity bar (straightness, net cm / path cm over 10 s windows)
fig, ax = plt.subplots(figsize=(8, 5))
sx = [results[l]["mean_straightness_10s"] for l in labels]
bcolors = [cmap[l] for l in labels]
bars = ax.bar([disp(l) for l in labels], sx, color=bcolors)
ax.set_ylabel("Mean straightness (net cm / path cm, 10 s windows)")
ax.set_title("Path tortuosity per leech (1 = straight, low = tortuous)")
ax.tick_params(axis="x", rotation=20)
for b, v in zip(bars, sx):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
f3 = os.path.join(OUT, "turning_tortuosity_bar.png"); fig.savefig(f3, dpi=120); plt.close(fig)

# Fig 4: early/mid/late thirds, grouped bars (speed cm/s and turning deg/min)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
xpos = np.arange(len(labels)); w = 0.25
seg_names = ["early", "mid", "late"]
seg_cols = ["#4c72b0", "#dd8452", "#55a868"]
for i, seg in enumerate(seg_names):
    sp = [thirds_data[l]["speed"][i] for l in labels]
    axs[0].bar(xpos + (i - 1) * w, sp, w, label=seg, color=seg_cols[i])
axs[0].set_xticks(xpos); axs[0].set_xticklabels([disp(l) for l in labels], rotation=20, fontsize=8)
axs[0].set_ylabel("Mean speed (cm/s)"); axs[0].set_title("Speed by session third")
axs[0].legend()
for i, seg in enumerate(seg_names):
    tn = [thirds_data[l]["turn"][i] for l in labels]
    axs[1].bar(xpos + (i - 1) * w, tn, w, label=seg, color=seg_cols[i])
axs[1].set_xticks(xpos); axs[1].set_xticklabels([disp(l) for l in labels], rotation=20, fontsize=8)
axs[1].set_ylabel("Turning (deg/min)"); axs[1].set_title("Turning by session third")
axs[1].legend()
fig.tight_layout()
f4 = os.path.join(OUT, "turning_thirds_grouped.png"); fig.savefig(f4, dpi=120); plt.close(fig)

# CSV
pd.DataFrame([results[l] for l in labels]).to_csv(
    os.path.join(MET, "turning_metrics.csv"), index=False)

print(json.dumps(results, indent=2))
print("FIGS", f1, f3, f4)
