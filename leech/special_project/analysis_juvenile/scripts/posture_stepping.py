#!/usr/bin/env python
"""Body-length rhythm and stepping (crawling) of JUVENILE leeches from DL keypoints.

Mirrors the adult fed-leech analyses (posture rhythm + stepping frequency) but for the
4 juvenile videos (8 leeches, 2 per video). A leech crawls in STEPS: the body ELONGATES
(extension) then SHORTENS (contraction) once per step, advancing the animal. So one
extension-contraction cycle = one step, seen as one oscillation of body length
|anterior - posterior|. We compute body length over time, its oscillation (stepping)
frequency, the step rate, the extension-contraction amplitude, and the mean centroid
speed for context. No em dashes anywhere."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, welch, find_peaks

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
OUT = BASE + "/analysis_juvenile/plots"
MET = BASE + "/analysis_juvenile/metrics"
CSV = BASE + "/DL/juv_all_predictions.csv"
CALJSON = BASE + "/DL/juv_calibration.json"
FPS = 29.99

# EXACT treatment labels per video (one per dish/video; both leeches share it)
TREATMENT = {
    "Video1_dish0.mp4": "Dopamine",
    "Video1_dish1.mp4": "Dopamine+Food",
    "Video2_dish0.mp4": "Food",
    "Video2_dish1.mp4": "Veh+Food",
}
# distinct colour per video (treatment); both leeches in a video share it
VCOLOR = {
    "Video1_dish0.mp4": "#1f77b4",   # Dopamine
    "Video1_dish1.mp4": "#9467bd",   # Dopamine+Food
    "Video2_dish0.mp4": "#2ca02c",   # Food
    "Video2_dish1.mp4": "#d62728",   # Veh+Food
}

with open(CALJSON) as fh:
    CALRAW = json.load(fh)
# calibration keyed without ".mp4"; map to the data's video names
CAL = {k + ".mp4": v for k, v in CALRAW.items()}


def smooth(a, win=11, poly=2):
    """Interpolate gaps then Savitzky-Golay smooth a trace."""
    a = pd.Series(a).interpolate(limit_direction="both").values
    if len(a) > win:
        a = savgol_filter(a, win, poly)
    return a


print("loading...")
df = pd.read_csv(CSV)
df = df.pivot_table(index=["video", "track", "frame", "time_s"],
                    columns="node", values=["x", "y"]).reset_index()
df.columns = ["video", "track", "frame", "time_s", "ax", "px", "ay", "py"]
df = df.sort_values(["video", "track", "frame"])

# 8 entities: 2 tracks (leeches) per video
entities = []
for v in ["Video1_dish0.mp4", "Video1_dish1.mp4", "Video2_dish0.mp4", "Video2_dish1.mp4"]:
    for trk in sorted(df[df.video == v].track.unique()):
        key = v.replace(".mp4", "") + f"_L{int(trk)}"
        entities.append((v, int(trk), key))
order = [e[2] for e in entities]

results = {}
series = {}   # key -> (t, bl_smoothed, peak_idx)
specs = {}    # key -> (freqs, psd)

for v, trk, key in entities:
    sub = df[(df.video == v) & (df.track == trk)].sort_values("frame")
    cal = CAL[v]
    cmpp = cal["cmpp"]
    low_conf = cal["pck"] < 60
    # body length |anterior - posterior| in cm
    bl = np.hypot(sub.ax - sub.px, sub.ay - sub.py).values * cmpp
    cx = ((sub.ax + sub.px) / 2).values * cmpp
    cy = ((sub.ay + sub.py) / 2).values * cmpp
    t = sub.time_s.values

    bl_s = smooth(bl, 11, 2)           # smooth trace (savgol 11, 2)
    cx_s, cy_s = smooth(cx), smooth(cy)
    speed = np.hypot(np.gradient(cx_s), np.gradient(cy_s)) * FPS   # cm/s
    med_bl = float(np.nanmedian(bl_s))

    # stepping frequency: Welch PSD of detrended body length (long savgol ~301 detrend)
    dwin = 301 if len(bl_s) > 301 else (len(bl_s) // 2 * 2 - 1)
    sig = bl_s - savgol_filter(bl_s, max(dwin, 5), 2)
    f, p = welch(sig, fs=FPS, nperseg=min(4096, len(sig)))
    band = (f >= 0.05) & (f <= 2.0)
    dom_f = float(f[band][np.argmax(p[band])])
    specs[key] = (f, p)

    # step count: extension peaks of body length, spacing from dom_f,
    # prominence = 25% of this leech's median body length so tracking
    # wobbles on a near-immobile leech are not miscounted as steps.
    min_dist = max(int(0.4 / dom_f * FPS), 5)
    prom = 0.25 * med_bl
    pks, _ = find_peaks(bl_s, distance=min_dist, prominence=prom)
    troughs, _ = find_peaks(-bl_s, distance=min_dist, prominence=prom)
    dur_min = (t[-1] - t[0]) / 60.0
    steps_per_min = len(pks) / dur_min
    amp = np.nan
    if len(pks) and len(troughs):
        amp = float(np.nanmedian(bl_s[pks]) - np.nanmedian(bl_s[troughs]))

    series[key] = (t, bl_s, pks)
    results[key] = dict(
        video=v, treatment=TREATMENT[v], leech=int(trk), low_conf=bool(low_conf),
        dom_freq_hz=dom_f, period_s=float(1.0 / dom_f),
        steps_per_min=float(steps_per_min), n_steps=int(len(pks)),
        ext_contr_amp_cm=amp, median_body_len_cm=med_bl,
        mean_speed_cm_s=float(np.nanmean(speed)), duration_min=float(dur_min))
    print(key, results[key]["treatment"],
          {k: round(v2, 3) for k, v2 in results[key].items()
           if isinstance(v2, (int, float))})


def vof(key):
    return results[key]["video"]


def disp(key):
    r = results[key]
    tag = " (low-conf)" if r["low_conf"] else ""
    return f"{r['treatment']} L{r['leech']}{tag}"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---- Figure 1: body-length traces with detected extension peaks (representative 60 s) ----
fig, axes = plt.subplots(len(order), 1, figsize=(13, 1.5 * len(order)), sharex=True)
for ax, key in zip(axes, order):
    t, bl_s, pks = series[key]
    # representative ~60 s window in the middle of the recording
    tmid = (t[0] + t[-1]) / 2.0
    lo, hi = tmid - 30, tmid + 30
    m = (t >= lo) & (t <= hi)
    ax.plot(t[m], bl_s[m], color=VCOLOR[vof(key)], lw=0.9)
    pk_m = pks[(t[pks] >= lo) & (t[pks] <= hi)]
    ax.plot(t[pk_m], bl_s[pk_m], "v", color="k", ms=5)
    ax.set_ylabel(f"{disp(key)}\nbody len (cm)", fontsize=7)
    style(ax)
axes[-1].set_xlabel("time (s)")
fig.suptitle("Juvenile stepping: body length |ant-post| over time, peaks = extensions "
             "(one step per cycle), ~60 s window", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/step_body_length_traces.png", dpi=130)
plt.close(fig)

# ---- Figure 2: body-length power spectra, dominant stepping frequency ----
fig, ax = plt.subplots(figsize=(10, 6))
for key in order:
    f, p = specs[key]
    m = (f > 0.02) & (f < 2.5)
    ls = "--" if results[key]["low_conf"] else "-"
    ax.semilogy(f[m], p[m], ls, color=VCOLOR[vof(key)], lw=1.3, label=disp(key))
    ax.axvline(results[key]["dom_freq_hz"], color=VCOLOR[vof(key)], ls=":", lw=0.7)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("body-length PSD (cm^2 / Hz)")
ax.set_title("Juvenile stepping rhythm: Welch power spectrum of body length "
             "(dotted = dominant stepping frequency)")
ax.legend(fontsize=7, ncol=2)
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/step_spectrum.png", dpi=130)
plt.close(fig)

# ---- Figure 3: stepping frequency + amplitude + speed bars per entity ----
fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
lbls = [disp(k).replace(" (low-conf)", "\n(low-conf)") for k in order]
cols = [VCOLOR[vof(k)] for k in order]
spm = [results[k]["steps_per_min"] for k in order]
amp = [results[k]["ext_contr_amp_cm"] for k in order]
spd = [results[k]["mean_speed_cm_s"] for k in order]
for ax, vals, title, ylab in [
        (axs[0], spm, "Stepping frequency", "steps / min"),
        (axs[1], amp, "Extension-contraction amplitude", "body-length change (cm)"),
        (axs[2], spd, "Mean crawling speed", "cm / s")]:
    ax.bar(lbls, vals, color=cols)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    for i, val in enumerate(vals):
        if np.isfinite(val):
            ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    style(ax)
fig.suptitle("Juvenile stepping metrics per leech (labeled by treatment); "
             "low-conf = pck < 60", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/step_metrics_bars.png", dpi=130)
plt.close(fig)

# ---- Figure 4: stepping frequency vs mean speed ----
fig, ax = plt.subplots(figsize=(8.5, 6))
seen = set()
for key in order:
    v = vof(key)
    lab = results[key]["treatment"]
    show = lab if v not in seen else None
    seen.add(v)
    ax.scatter(results[key]["mean_speed_cm_s"], results[key]["steps_per_min"],
               s=150, color=VCOLOR[v], edgecolors="k",
               marker="s" if results[key]["low_conf"] else "o", label=show)
    ax.annotate(f"L{results[key]['leech']}",
                (results[key]["mean_speed_cm_s"], results[key]["steps_per_min"]),
                fontsize=7, ha="center", va="center", color="white")
ax.set_xlabel("mean crawling speed (cm/s)")
ax.set_ylabel("stepping frequency (steps/min)")
ax.set_title("Juvenile stepping frequency vs locomotion "
             "(square = low-conf tracking)")
ax.legend(fontsize=8, title="treatment")
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/step_vs_movement.png", dpi=130)
plt.close(fig)

# ---- metrics CSV ----
cols_order = ["entity", "video", "treatment", "leech", "low_conf", "dom_freq_hz",
              "period_s", "steps_per_min", "n_steps", "ext_contr_amp_cm",
              "median_body_len_cm", "mean_speed_cm_s", "duration_min"]
rows = []
for k in order:
    r = dict(results[k])
    r["entity"] = k
    rows.append({c: r[c] for c in cols_order})
pd.DataFrame(rows)[cols_order].to_csv(f"{MET}/stepping_metrics.csv", index=False)

print("FIGS step_body_length_traces, step_spectrum, step_metrics_bars, step_vs_movement")
print("DONE")
