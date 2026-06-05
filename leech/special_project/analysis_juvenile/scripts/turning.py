"""Turning and path-tortuosity analysis for JUVENILE leeches.

Mirrors the adult analysis (analysis_fed_leech/scripts/turning_analysis.py)
but for 4 juvenile videos x 2 leeches each = 8 entities.

Input:
  DL/juv_all_predictions.csv : video,treatment,frame,time_s,track,node,x,y
    node0 = anterior (head), node1 = posterior (tail). track 0/1 = the 2 leeches.
  DL/juv_calibration.json    : per-video {treatment,pck,cx,cy,r_px,cmpp}.

Heading definition: the anterior->posterior body axis, atan2(head - tail).
We use the body-axis heading (not centroid movement direction) because the
keypoint pair gives a direct, instantaneous body orientation that is robust
even when the animal is momentarily stationary.

Outputs:
  analysis_juvenile/plots/turn_rate_over_time.png
  analysis_juvenile/plots/turn_tortuosity_bars.png
  analysis_juvenile/plots/turn_thirds_grouped.png
  analysis_juvenile/metrics/turning_metrics.csv
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
OUT = os.path.join(BASE, "analysis_juvenile/plots")
MET = os.path.join(BASE, "analysis_juvenile/metrics")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MET, exist_ok=True)

FPS = 29.99
WIN = 11          # Savitzky-Golay window, polyorder 2, applied to px coords
TORT_WIN_S = 10.0  # sliding window for tortuosity (seconds)
LOWCONF_PCK = 60   # pck below this -> tagged low-conf

# Treatment mapping is taken from the calibration JSON, but the spec fixes it:
TREATMENT = {
    "Video1_dish0": "Dopamine",
    "Video1_dish1": "Dopamine+Food",
    "Video2_dish0": "Food",
    "Video2_dish1": "Veh+Food",
}


def smooth(a, w=WIN):
    a = np.asarray(a, float)
    if len(a) < w:
        return a
    return savgol_filter(a, w, 2)


def angdiff(a):
    """Framewise wrapped difference of an angle array (radians)."""
    d = np.diff(a)
    return (d + np.pi) % (2 * np.pi) - np.pi


print("loading...")
df = pd.read_csv(os.path.join(BASE, "DL/juv_all_predictions.csv"))
df["vid"] = df["video"].str.replace(".mp4", "", regex=False)

with open(os.path.join(BASE, "DL/juv_calibration.json")) as fh:
    CAL = json.load(fh)


def get_track(vid, track):
    sub = df[(df["vid"] == vid) & (df["track"] == track)]
    if len(sub) == 0:
        return None
    h = sub[sub["node"] == 0].sort_values("frame")
    t = sub[sub["node"] == 1].sort_values("frame")
    m = h[["frame", "time_s", "x", "y"]].merge(
        t[["frame", "x", "y"]], on="frame", suffixes=("_h", "_t"))
    return m


# 8 entities: each of 4 videos has 2 leeches (track 0 and 1).
ENTITIES = []
for vid in ["Video1_dish0", "Video1_dish1", "Video2_dish0", "Video2_dish1"]:
    for tr in (0, 1):
        ENTITIES.append((vid, tr, f"{vid}_L{tr}"))

results = {}
turn_timeseries = {}   # label -> (bin_centers_min, turning_deg_per_min, speed_cm_s)
thirds_data = {}       # label -> {"speed":[...], "turn":[...]}

for vid, tr, label in ENTITIES:
    m = get_track(vid, tr)
    if m is None or len(m) < WIN + 2:
        print("skip", label, "no/short data")
        continue
    cal = CAL[vid]
    cmpp = cal["cmpp"]
    cx0 = cal["cx"]
    cy0 = cal["cy"]
    pck = cal["pck"]
    treatment = TREATMENT[vid]
    low_conf = pck < LOWCONF_PCK

    n = len(m)
    t = m["time_s"].values

    # smooth coords in px, convert to cm relative to arena center
    hx = (smooth(m["x_h"].values) - cx0) * cmpp
    hy = (smooth(m["y_h"].values) - cy0) * cmpp
    tx = (smooth(m["x_t"].values) - cx0) * cmpp
    ty = (smooth(m["y_t"].values) - cy0) * cmpp
    cx = (hx + tx) / 2.0
    cy = (hy + ty) / 2.0

    # Heading = anterior->posterior body axis (head relative to tail)
    heading = np.arctan2(hy - ty, hx - tx)   # radians

    dt = np.diff(t)
    dt[dt <= 0] = 1.0 / FPS
    dhead = angdiff(heading)                  # radians
    abs_dhead_deg = np.abs(np.degrees(dhead))
    angvel = np.degrees(dhead) / dt           # deg/s (angular velocity)

    # centroid speed cm/s
    dcx = np.diff(cx)
    dcy = np.diff(cy)
    step_cm = np.sqrt(dcx**2 + dcy**2)
    speed_cm_s = step_cm / dt

    total_turn_deg = float(np.sum(abs_dhead_deg))
    duration_min = (t[-1] - t[0]) / 60.0
    turn_per_min = total_turn_deg / duration_min if duration_min > 0 else np.nan

    # Tortuosity over sliding 10 s windows = path length / net displacement.
    # 1 = perfectly straight; larger = more tortuous.
    wf = int(round(TORT_WIN_S * FPS))
    tort_vals = []
    net_list = []
    path_list = []
    for start in range(0, max(0, len(cx) - wf), wf):
        sx = cx[start:start + wf]
        sy = cy[start:start + wf]
        path = np.sum(np.sqrt(np.diff(sx)**2 + np.diff(sy)**2))
        net = np.sqrt((sx[-1] - sx[0])**2 + (sy[-1] - sy[0])**2)
        if net > 1e-6:
            tort_vals.append(path / net)
            net_list.append(net)
            path_list.append(path)
    mean_tort = float(np.nanmean(tort_vals)) if tort_vals else np.nan
    mean_net = float(np.nanmean(net_list)) if net_list else np.nan
    mean_path = float(np.nanmean(path_list)) if path_list else np.nan

    # per-minute time series (turning deg/min, mean speed cm/s)
    tmin = t[1:] / 60.0
    nbins = max(1, int(np.ceil(tmin[-1])))
    idx = np.minimum(np.floor(tmin).astype(int), nbins - 1)
    turn_binned = np.array([np.sum(abs_dhead_deg[idx == b]) for b in range(nbins)])
    speed_binned = np.array(
        [np.nanmean(speed_cm_s[idx == b]) if np.any(idx == b) else np.nan
         for b in range(nbins)])
    turn_timeseries[label] = (np.arange(nbins) + 0.5, turn_binned, speed_binned)

    # early / mid / late thirds
    third = n // 3
    seg_idx = [(0, third), (third, 2 * third), (2 * third, n)]
    seg_speed = []
    seg_turn = []
    for a, b in seg_idx:
        aa = min(a, len(speed_cm_s))
        bb = min(b, len(speed_cm_s))
        seg_speed.append(float(np.nanmean(speed_cm_s[aa:bb])) if bb > aa else np.nan)
        dur_m = (t[min(b, n - 1)] - t[a]) / 60.0
        seg_turn.append(float(np.sum(abs_dhead_deg[aa:bb]) / dur_m)
                        if dur_m > 0 else np.nan)
    thirds_data[label] = {"speed": seg_speed, "turn": seg_turn}

    results[label] = {
        "entity": label,
        "video": vid,
        "treatment": treatment,
        "leech": tr,
        "low_conf": bool(low_conf),
        "pck": pck,
        "n_frames": int(n),
        "duration_min": round(duration_min, 2),
        "total_turning_deg": round(total_turn_deg, 1),
        "turning_per_min_deg": round(turn_per_min, 1),
        "mean_abs_angvel_deg_s": round(float(np.nanmean(np.abs(angvel))), 3),
        "median_abs_angvel_deg_s": round(float(np.nanmedian(np.abs(angvel))), 3),
        "mean_tortuosity_10s": round(mean_tort, 3),
        "mean_net_disp_cm_10s": round(mean_net, 3),
        "mean_path_len_cm_10s": round(mean_path, 3),
        "mean_speed_cm_s": round(float(np.nanmean(speed_cm_s)), 4),
        "speed_early_cm_s": round(seg_speed[0], 4),
        "speed_mid_cm_s": round(seg_speed[1], 4),
        "speed_late_cm_s": round(seg_speed[2], 4),
        "turn_early_deg_min": round(seg_turn[0], 1),
        "turn_mid_deg_min": round(seg_turn[1], 1),
        "turn_late_deg_min": round(seg_turn[2], 1),
    }
    print(label, treatment, "done", "(low-conf)" if low_conf else "")

labels = list(results.keys())

# Color by treatment, with a distinct shade per leech within a treatment.
TREAT_COLORS = {
    "Dopamine": "#c44e52",
    "Dopamine+Food": "#dd8452",
    "Food": "#55a868",
    "Veh+Food": "#4c72b0",
}


def ecolor(l):
    return TREAT_COLORS[results[l]["treatment"]]


def estyle(l):
    # dashed for second leech and/or low-conf, so leeches are distinguishable
    ls = "--" if results[l]["leech"] == 1 else "-"
    return {"color": ecolor(l), "ls": ls}


def disp(l):
    base = f"{l} [{results[l]['treatment']}]"
    return base + " (low-conf)" if results[l]["low_conf"] else base


# ---------------------------------------------------------------------------
# Fig 1: turning rate (deg/min, 1-min bins) over time + speed panel
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
for l in labels:
    x, turn, _ = turn_timeseries[l]
    axes[0].plot(x, turn, label=disp(l), alpha=0.85, lw=1.3, **estyle(l))
axes[0].set_ylabel("Turning rate (deg / min)")
axes[0].set_title("Juvenile turning rate over session (1-min bins; body-axis heading, "
                  "Savitzky-Golay w=11)")
axes[0].legend(fontsize=7, ncol=2)
for l in labels:
    x, _, sp = turn_timeseries[l]
    axes[1].plot(x, sp, label=disp(l), alpha=0.85, lw=1.3, **estyle(l))
axes[1].set_xlabel("Time (min)")
axes[1].set_ylabel("Mean speed (cm/s)")
axes[1].set_title("Centroid speed over session (1-min bins)")
axes[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
f1 = os.path.join(OUT, "turn_rate_over_time.png")
fig.savefig(f1, dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: grouped bars per entity, turning per min and mean tortuosity
fig, ax1 = plt.subplots(figsize=(13, 6))
xpos = np.arange(len(labels))
w = 0.4
tpm = [results[l]["turning_per_min_deg"] for l in labels]
tort = [results[l]["mean_tortuosity_10s"] for l in labels]
bcolors = [ecolor(l) for l in labels]
b1 = ax1.bar(xpos - w / 2, tpm, w, color=bcolors, edgecolor="black", label="Turning (deg/min)")
ax1.set_ylabel("Turning per minute (deg/min)")
ax1.set_xticks(xpos)
ax1.set_xticklabels([disp(l) for l in labels], rotation=30, ha="right", fontsize=7)
ax2 = ax1.twinx()
b2 = ax2.bar(xpos + w / 2, tort, w, color=bcolors, alpha=0.45, hatch="//",
             edgecolor="black", label="Tortuosity (10 s)")
ax2.set_ylabel("Mean tortuosity (path / net, 10 s; 1 = straight)")
ax2.axhline(1.0, color="gray", ls=":", lw=1)
for x, v in zip(xpos, tpm):
    ax1.text(x - w / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
for x, v in zip(xpos, tort):
    ax2.text(x + w / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
ax1.set_title("Turning rate (solid) and path tortuosity (hatched) per juvenile leech "
              "(color = treatment)")
ax1.legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)
fig.tight_layout()
f2 = os.path.join(OUT, "turn_tortuosity_bars.png")
fig.savefig(f2, dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3: early/mid/late thirds, averaged per treatment (speed and turning)
treatments = ["Dopamine", "Dopamine+Food", "Food", "Veh+Food"]
seg_names = ["early", "mid", "late"]
seg_cols = ["#4c72b0", "#dd8452", "#55a868"]


def treat_seg_mean(treatment, key, seg_i):
    vals = [thirds_data[l][key][seg_i] for l in labels
            if results[l]["treatment"] == treatment]
    return np.nanmean(vals) if vals else np.nan


fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
xpos = np.arange(len(treatments))
w = 0.25
for i, seg in enumerate(seg_names):
    sp = [treat_seg_mean(tt, "speed", i) for tt in treatments]
    axs[0].bar(xpos + (i - 1) * w, sp, w, label=seg, color=seg_cols[i], edgecolor="black")
axs[0].set_xticks(xpos)
axs[0].set_xticklabels(treatments, rotation=20, ha="right", fontsize=8)
axs[0].set_ylabel("Mean speed (cm/s)")
axs[0].set_title("Speed by session third (mean of 2 leeches / treatment)")
axs[0].legend(title="third")
for i, seg in enumerate(seg_names):
    tn = [treat_seg_mean(tt, "turn", i) for tt in treatments]
    axs[1].bar(xpos + (i - 1) * w, tn, w, label=seg, color=seg_cols[i], edgecolor="black")
axs[1].set_xticks(xpos)
axs[1].set_xticklabels(treatments, rotation=20, ha="right", fontsize=8)
axs[1].set_ylabel("Turning (deg/min)")
axs[1].set_title("Turning by session third (mean of 2 leeches / treatment)")
axs[1].legend(title="third")
fig.tight_layout()
f3 = os.path.join(OUT, "turn_thirds_grouped.png")
fig.savefig(f3, dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# CSV (column order per spec, plus a few extras at the end)
cols = ["entity", "video", "treatment", "leech", "low_conf", "n_frames",
        "duration_min", "total_turning_deg", "turning_per_min_deg",
        "mean_abs_angvel_deg_s", "median_abs_angvel_deg_s", "mean_tortuosity_10s",
        "mean_speed_cm_s", "turn_early_deg_min", "turn_mid_deg_min",
        "turn_late_deg_min",
        # extras
        "pck", "mean_net_disp_cm_10s", "mean_path_len_cm_10s",
        "speed_early_cm_s", "speed_mid_cm_s", "speed_late_cm_s"]
out_df = pd.DataFrame([results[l] for l in labels])[cols]
csv_path = os.path.join(MET, "turning_metrics.csv")
out_df.to_csv(csv_path, index=False)

print("\n=== summary ===")
print(out_df[["entity", "treatment", "turning_per_min_deg", "mean_tortuosity_10s",
              "mean_speed_cm_s"]].to_string(index=False))
print("\nFIGS:", f1, f2, f3)
print("CSV :", csv_path)
