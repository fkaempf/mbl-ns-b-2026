#!/usr/bin/env python
"""Stepping frequency for the densely tracked juvenile-leech videos (DL/juv_all_predictions.csv).

A leech crawls in steps: the body ELONGATES (extension) then SHORTENS (contraction)
once per step, advancing the animal. So one extension-contraction cycle = one step, and
it shows up as one oscillation of body length |anterior - posterior|. Stepping frequency
is the rate of those body-length oscillations; we also report the accompanying movement.
No em dashes."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, welch, find_peaks

OUT = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/analysis_juvenile/plots"
MET = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/analysis_juvenile/metrics"
CSV = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/DL/juv_all_predictions.csv"
FPS = 29.99

# per-video pixel->cm scale (dish = 3.5 cm diameter); same calibration as the other scripts
CMPP = {"Video1_dish0.mp4": 0.0067609, "Video1_dish1.mp4": 0.0069246,
        "Video2_dish0.mp4": 0.0070045, "Video2_dish1.mp4": 0.0075799}
DISPLAY = {"Video1_dish0_L0": "Dopamine L0", "Video1_dish0_L1": "Dopamine L1",
           "Video1_dish1_L0": "DA+Food L0", "Video1_dish1_L1": "DA+Food L1",
           "Video2_dish0_L0": "Veh+NoFood L0", "Video2_dish0_L1": "Veh+NoFood L1",
           "Video2_dish1_L0": "Veh+Food L0", "Video2_dish1_L1": "Veh+Food L1"}
COLORS = {"Video1_dish0_L0": "#1f77b4", "Video1_dish0_L1": "#ff7f0e",
          "Video1_dish1_L0": "#2ca02c", "Video1_dish1_L1": "#d62728",
          "Video2_dish0_L0": "#9467bd", "Video2_dish0_L1": "#8c564b",
          "Video2_dish1_L0": "#e377c2", "Video2_dish1_L1": "#7f7f7f"}
LOWCONF = {"Video1_dish0_L0", "Video1_dish0_L1", "Video1_dish1_L0",
           "Video1_dish1_L1", "Video2_dish1_L0", "Video2_dish1_L1"}

groups = [
    ("Video1_dish0.mp4", 0, "Video1_dish0_L0"),
    ("Video1_dish0.mp4", 1, "Video1_dish0_L1"),
    ("Video1_dish1.mp4", 0, "Video1_dish1_L0"),
    ("Video1_dish1.mp4", 1, "Video1_dish1_L1"),
    ("Video2_dish0.mp4", 0, "Video2_dish0_L0"),
    ("Video2_dish0.mp4", 1, "Video2_dish0_L1"),
    ("Video2_dish1.mp4", 0, "Video2_dish1_L0"),
    ("Video2_dish1.mp4", 1, "Video2_dish1_L1"),
]
order = [g[2] for g in groups]

# Each treatment is ONE video with 2 tracked leeches (L0, L1). Group the 8 entities
# into 4 treatments for the per-treatment summary plots (mean of the 2 leeches).
TREATMENTS = [
    ("Dopamine", ["Video1_dish0_L0", "Video1_dish0_L1"], True),
    ("DA+Food",  ["Video1_dish1_L0", "Video1_dish1_L1"], True),
    ("Veh+NoFood",     ["Video2_dish0_L0", "Video2_dish0_L1"], False),
    ("Veh+Food", ["Video2_dish1_L0", "Video2_dish1_L1"], True),
]
TREAT_COLORS = {"Dopamine": "#1f77b4", "DA+Food": "#2ca02c",
                "Veh+NoFood": "#9467bd", "Veh+Food": "#e377c2"}

WIN = 15  # ~0.5 s smoothing


def smooth(a, win=WIN):
    a = np.asarray(a, float)
    if len(a) < win:
        return a
    nan = np.isnan(a)
    if nan.any():
        idx = np.arange(len(a))
        if (~nan).any():
            a = a.copy(); a[nan] = np.interp(idx[nan], idx[~nan], a[~nan])
    return savgol_filter(a, win, 2)


print("loading...")
df = pd.read_csv(CSV)
df = df.pivot_table(index=["video", "track", "frame", "time_s"],
                    columns="node", values=["x", "y"]).reset_index()
df.columns = ["video", "track", "frame", "time_s", "ax", "px", "ay", "py"]
df = df.sort_values(["video", "track", "frame"])

results = {}
series = {}     # name -> (t, body_len_cm, peaks_idx)
specs = {}      # name -> (freqs, psd)

for v, trk, name in groups:
    sub = df[(df.video == v) & (df.track == trk)].sort_values("frame")
    cmpp = CMPP[v]
    bl = np.hypot(sub.ax - sub.px, sub.ay - sub.py).values * cmpp   # body length in cm
    cx = ((sub.ax + sub.px) / 2).values * cmpp
    cy = ((sub.ay + sub.py) / 2).values * cmpp
    t = sub.time_s.values
    bl_s = smooth(bl)
    cx_s, cy_s = smooth(cx), smooth(cy)
    speed = np.hypot(np.gradient(cx_s), np.gradient(cy_s)) * FPS      # cm/s

    # stepping frequency via spectrum of the (detrended) body-length signal
    sig = bl_s - savgol_filter(bl_s, 301 if len(bl_s) > 301 else (len(bl_s) // 2 * 2 - 1), 2)
    f, p = welch(sig, fs=FPS, nperseg=min(4096, len(sig)))
    band = (f >= 0.05) & (f <= 2.0)                                  # plausible crawl band
    dom_f = f[band][np.argmax(p[band])]
    specs[name] = (f, p)

    # count extension peaks (each peak = one elongation = one step), min spacing from dom_f.
    # A REAL step elongates the body by a sizeable fraction of its length; require the
    # peak prominence to exceed 25% of median body length so tracking wobbles on a
    # near-immobile leech are not miscounted as steps.
    min_dist = max(int(0.4 / dom_f * FPS), 5)
    prom = 0.25 * np.nanmedian(bl_s)
    pks, _ = find_peaks(bl_s, distance=min_dist, prominence=prom)
    dur_min = (t[-1] - t[0]) / 60.0
    steps_per_min = len(pks) / dur_min
    # extension-contraction amplitude (peak-to-trough of body length)
    troughs, _ = find_peaks(-bl_s, distance=min_dist, prominence=prom)
    amp = np.nan
    if len(pks) and len(troughs):
        amp = np.nanmedian(bl_s[pks]) - np.nanmedian(bl_s[troughs])
    # fraction of time moving (speed > 0.05 cm/s)
    pct_moving = 100 * np.mean(speed > 0.05)

    series[name] = (t, bl_s, pks)
    results[name] = dict(
        dom_freq_hz=float(dom_f), period_s=float(1 / dom_f),
        steps_per_min=float(steps_per_min), n_steps=int(len(pks)),
        ext_contr_amp_cm=float(amp), median_body_len_cm=float(np.nanmedian(bl_s)),
        mean_speed_cm_s=float(np.nanmean(speed)), pct_time_moving=float(pct_moving),
        duration_min=float(dur_min))
    print(name, {k: round(v, 3) for k, v in results[name].items()})


def disp(n):
    return DISPLAY[n] + (" (low-conf)" if n in LOWCONF else "")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---- Figure 1: body-length traces with detected steps (representative 60 s window) ----
fig, axes = plt.subplots(len(order), 1, figsize=(13, 10), sharex=True)
for ax, name in zip(axes, order):
    t, bl_s, pks = series[name]
    m = (t >= 600) & (t <= 660)                                     # 10-11 min window
    ax.plot(t[m], bl_s[m], color=COLORS[name], lw=0.9)
    pk_m = pks[(t[pks] >= 600) & (t[pks] <= 660)]
    ax.plot(t[pk_m], bl_s[pk_m], "v", color="k", ms=5)
    ax.set_ylabel(f"{disp(name)}\nbody len (cm)", fontsize=8)
    style(ax)
axes[-1].set_xlabel("time (s)")
fig.suptitle("Stepping: body length |ant-post| over time, peaks = extensions (one step per cycle), 10-11 min window",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/stepping_traces.png", dpi=130); plt.close(fig)

# ---- Figure 2: body-length power spectra, dominant stepping frequency ----
fig, ax = plt.subplots(figsize=(10, 6))
for name in order:
    f, p = specs[name]
    m = (f > 0.02) & (f < 2.5)
    ls = "--" if name in LOWCONF else "-"
    ax.semilogy(f[m], p[m], ls, color=COLORS[name], lw=1.3, label=disp(name))
    ax.axvline(results[name]["dom_freq_hz"], color=COLORS[name], ls=":", lw=0.8)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("body-length PSD (cm^2 / Hz)")
ax.set_title("Stepping rhythm: power spectrum of body length per leech (dotted = dominant frequency)")
ax.legend(fontsize=8); style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/stepping_spectrum.png", dpi=130); plt.close(fig)

# ---- Figure 3: stepping metrics per treatment (one bar = mean of the 2 leeches) ----
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
treat_lbl = [t[0] + ("\n(low-conf)" if t[2] else "") for t in TREATMENTS]
treat_cols = [TREAT_COLORS[t[0]] for t in TREATMENTS]


def treat_mean(metric):
    return [float(np.nanmean([results[n][metric] for n in members]))
            for _, members, _ in TREATMENTS]


spm = treat_mean("steps_per_min")
amp = treat_mean("ext_contr_amp_cm")
spd = treat_mean("mean_speed_cm_s")
for ax, vals, title, ylab in [
        (axs[0], spm, "Stepping frequency", "steps / min"),
        (axs[1], amp, "Extension-contraction amplitude", "body-length change (cm)"),
        (axs[2], spd, "Mean crawling speed", "cm / s")]:
    ax.bar(treat_lbl, vals, color=treat_cols)
    ax.set_title(title); ax.set_ylabel(ylab); ax.tick_params(axis="x", rotation=20)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    style(ax)
fig.suptitle("Stepping metrics per treatment (each bar = mean of the 2 leeches; dense full-video tracking)",
             fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/stepping_metrics_bars.png", dpi=130); plt.close(fig)

# ---- Figure 4: stepping frequency vs movement (one point = mean of the 2 leeches) ----
fig, ax = plt.subplots(figsize=(8, 6))
for tname, members, lowconf in TREATMENTS:
    x = float(np.nanmean([results[n]["mean_speed_cm_s"] for n in members]))
    y = float(np.nanmean([results[n]["steps_per_min"] for n in members]))
    lbl = tname + (" (low-conf)" if lowconf else "")
    ax.scatter(x, y, s=160, color=TREAT_COLORS[tname], edgecolors="k",
               marker="s" if lowconf else "o", label=lbl)
ax.set_xlabel("mean crawling speed (cm/s)"); ax.set_ylabel("stepping frequency (steps/min)")
ax.set_title("Stepping frequency vs locomotion per treatment (mean of the 2 leeches)")
ax.legend(fontsize=8); style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/stepping_vs_movement.png", dpi=130); plt.close(fig)

# ---- metrics CSV ----
rows = []
for n in order:
    r = {"entity": n, "treatment": DISPLAY[n], "low_confidence": n in LOWCONF}
    r.update(results[n])
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{MET}/stepping_metrics.csv", index=False)
print("FIGS stepping_traces, stepping_spectrum, stepping_metrics_bars, stepping_vs_movement")
print("DONE")
