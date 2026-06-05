import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, welch, find_peaks

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project"
OUT = BASE + "/analysis_juvenile/plots"
MET = BASE + "/analysis_juvenile/metrics"
FPS = 29.99

# Per-video arena calibration (center px, cm per px). Dish = 3.5 cm diameter, radius 1.75 cm.
CAL = {
    "Video1_dish0.mp4": dict(cx=274.2, cy=275.4, cmpp=0.0067609),
    "Video1_dish1.mp4": dict(cx=271.8, cy=267.0, cmpp=0.0069246),
    "Video2_dish0.mp4": dict(cx=282.6, cy=269.4, cmpp=0.0070045),
    "Video2_dish1.mp4": dict(cx=252.6, cy=243.0, cmpp=0.0075799),
}

df = pd.read_csv(BASE + "/DL/juv_all_predictions.csv")


def get_series(sub):
    p = sub.pivot_table(index="frame", columns="node", values=["x", "y", "time_s"])
    t = p[("time_s", 0)].values
    hx = p[("x", 0)].values; hy = p[("y", 0)].values
    tx = p[("x", 1)].values; ty = p[("y", 1)].values
    return t, hx, hy, tx, ty


def smooth(a, win=11):
    # interpolate gaps, rolling-median despike, then Savitzky-Golay (win~11)
    a = pd.Series(a).interpolate(limit_direction="both").values
    a = pd.Series(a).rolling(win, center=True, min_periods=1).median().values
    if len(a) > 11:
        a = savgol_filter(a, 11, 3)
    return a


# Build the 8 entities keyed "<videostem>_L<track>", grouped by video.
units = [
    ("Video1_dish0.mp4", 0, "Video1_dish0_L0"),
    ("Video1_dish0.mp4", 1, "Video1_dish0_L1"),
    ("Video1_dish1.mp4", 0, "Video1_dish1_L0"),
    ("Video1_dish1.mp4", 1, "Video1_dish1_L1"),
    ("Video2_dish0.mp4", 0, "Video2_dish0_L0"),
    ("Video2_dish0.mp4", 1, "Video2_dish0_L1"),
    ("Video2_dish1.mp4", 0, "Video2_dish1_L0"),
    ("Video2_dish1.mp4", 1, "Video2_dish1_L1"),
]
LOWCONF = {"Video1_dish0_L0", "Video1_dish0_L1", "Video1_dish1_L0",
           "Video1_dish1_L1", "Video2_dish1_L0", "Video2_dish1_L1"}

results = {}
for video, tr, label in units:
    sub = df[(df["video"] == video) & (df["track"] == tr)]
    t, hx, hy, tx, ty = get_series(sub)
    cal = CAL[video]; cmpp = cal["cmpp"]; cx = cal["cx"]; cy = cal["cy"]
    # to cm (centered on arena)
    hxc = (hx - cx) * cmpp; hyc = (hy - cy) * cmpp
    txc = (tx - cx) * cmpp; tyc = (ty - cy) * cmpp
    hxc, hyc, txc, tyc = [smooth(a) for a in (hxc, hyc, txc, tyc)]
    bl_cm = np.hypot(hxc - txc, hyc - tyc)
    head_path = np.nansum(np.hypot(np.diff(hxc), np.diff(hyc)))  # cm
    tail_path = np.nansum(np.hypot(np.diff(txc), np.diff(tyc)))  # cm
    results[label] = dict(t=t, bl_cm=bl_cm, head_path=head_path,
                          tail_path=tail_path, video=video, lowconf=(label in LOWCONF))

# ---- metrics ----
metrics = {}
SPEC = {}
for label, r in results.items():
    bl = r["bl_cm"]
    med = np.nanmedian(bl)
    bl_min = np.nanmin(bl); bl_max = np.nanmax(bl)
    frac_elong = np.nanmean(bl > med)
    cv = np.nanstd(bl) / np.nanmean(bl)
    # elongation / contraction events: peaks above/below own median, >=1 s apart,
    # prominence = 0.10 * median body length (cm)
    sig = bl - med
    prom = 0.10 * med
    pk_e, _ = find_peaks(sig, prominence=prom, distance=int(FPS * 1))
    pk_c, _ = find_peaks(-sig, prominence=prom, distance=int(FPS * 1))
    dur_min = (r["t"][-1] - r["t"][0]) / 60.0
    elong_rate = len(pk_e) / dur_min
    contr_rate = len(pk_c) / dur_min
    # FFT/Welch on body length (cm, detrended)
    x = pd.Series(bl).interpolate(limit_direction="both").values
    x = x - np.nanmean(x)
    nper = int(FPS * 120)  # 2-min segments
    f, Pxx = welch(x, fs=FPS, nperseg=min(nper, len(x)))
    band = (f > 0.01) & (f < 2.0)  # plausible peristalsis / crawl band
    fb, Pb = f[band], Pxx[band]
    peak_f = fb[np.argmax(Pb)]
    rel_pow = Pb.max() / np.sum(Pb)  # fraction of in-band power at dominant peak
    ht_ratio = r["head_path"] / r["tail_path"] if r["tail_path"] > 0 else np.nan
    metrics[label] = dict(
        dish=r["video"], low_conf=int(r["lowconf"]),
        median_bl_cm=med, bl_min_cm=bl_min, bl_max_cm=bl_max,
        bl_range_cm=bl_max - bl_min, cv_bl=cv, pct_time_elongated=100 * frac_elong,
        elong_rate_per_min=elong_rate, contr_rate_per_min=contr_rate,
        dom_freq_hz=peak_f, dom_rel_power=rel_pow,
        head_path_cm=r["head_path"], tail_path_cm=r["tail_path"], head_tail_path_ratio=ht_ratio)
    SPEC[label] = (f, Pxx)

DISPLAY = {"Video1_dish0_L0": "Dopamine L0", "Video1_dish0_L1": "Dopamine L1",
           "Video1_dish1_L0": "DA+Food L0", "Video1_dish1_L1": "DA+Food L1",
           "Video2_dish0_L0": "Veh+NoFood L0", "Video2_dish0_L1": "Veh+NoFood L1",
           "Video2_dish1_L0": "Veh+Food L0", "Video2_dish1_L1": "Veh+Food L1"}

mdf = pd.DataFrame(metrics).T
mdf.insert(0, "treatment", [DISPLAY[i] for i in mdf.index])
mdf.index.name = "entity"
mdf.to_csv(MET + "/posture_rhythm_metrics.csv")
print(mdf.round(4).to_string())

COLORS = {"Video1_dish0_L0": "#1f77b4", "Video1_dish0_L1": "#ff7f0e",
          "Video1_dish1_L0": "#2ca02c", "Video1_dish1_L1": "#d62728",
          "Video2_dish0_L0": "#9467bd", "Video2_dish0_L1": "#8c564b",
          "Video2_dish1_L0": "#e377c2", "Video2_dish1_L1": "#7f7f7f"}

# ---- Figure 1: body length over a representative 5-min mid-clip window (cm) ----
n = len(results)
fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 * n), sharex=True)
if n == 1: axes = [axes]
for ax, (label, r) in zip(axes, results.items()):
    t = r["t"]; bl = r["bl_cm"]
    m0 = (t >= 1200) & (t <= 1500)  # 20-25 min window
    ax.plot(t[m0] / 60.0, bl[m0], lw=0.6, color=COLORS[label])
    ax.axhline(np.nanmedian(bl), color="grey", ls="--", lw=0.7)
    tag = " (low conf)" if r["lowconf"] else ""
    ax.set_ylabel(f"{DISPLAY[label]}{tag}\nbody length (cm)", fontsize=8)
axes[-1].set_xlabel("time (min)")
fig.suptitle("Body length |head-tail| over time: 20-25 min window, smoothed (SavGol win 11)")
fig.tight_layout()
fig.savefig(OUT + "/posture_rhythm_bodylength_timeseries.png", dpi=130)
plt.close(fig)

# ---- Figure 2: power spectra ----
fig, ax = plt.subplots(figsize=(9, 6))
for label, (f, Pxx) in SPEC.items():
    m = (f > 0.005) & (f < 1.0)
    ls = "--" if results[label]["lowconf"] else "-"
    lbl = DISPLAY[label] + (" (low conf)" if results[label]["lowconf"] else "")
    ax.semilogy(f[m], Pxx[m], ls, lw=1.2, color=COLORS[label], label=lbl)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD (cm^2 / Hz)")
ax.set_title("Welch power spectra of body length (crawling rhythm)")
ax.legend(fontsize=8); ax.set_xlim(0, 0.6)
fig.tight_layout(); fig.savefig(OUT + "/posture_rhythm_powerspectra.png", dpi=130); plt.close(fig)

# ---- Figure 3: bar charts (one bar PER TREATMENT = mean of that treatment's 2 leeches) ----
# Each treatment is one video containing 2 tracked leeches (L0, L1).
TREATMENTS = [
    ("Dopamine", ["Video1_dish0_L0", "Video1_dish0_L1"], True),
    ("DA+Food",  ["Video1_dish1_L0", "Video1_dish1_L1"], True),
    ("Veh+NoFood",     ["Video2_dish0_L0", "Video2_dish0_L1"], False),
    ("Veh+Food", ["Video2_dish1_L0", "Video2_dish1_L1"], False),
]
treat_names = [name for name, _, _ in TREATMENTS]
treat_bar_labels = [name + (" (low-conf)" if low else "") for name, _, low in TREATMENTS]
# one color per treatment (use the first leech's color of each treatment)
treat_cols = [COLORS[members[0]] for _, members, _ in TREATMENTS]

def treat_means(key):
    return [float(np.nanmean([metrics[m][key] for m in members]))
            for _, members, _ in TREATMENTS]

fig, axs = plt.subplots(2, 3, figsize=(16, 9))

def bar(ax, vals, title, ylab, hline=None):
    bars = ax.bar(treat_bar_labels, vals, color=treat_cols)
    ax.set_title(title); ax.set_ylabel(ylab)
    ax.tick_params(axis='x', rotation=45)
    if hline is not None: ax.axhline(hline, color="k", ls="--", lw=0.8)
    ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=2)

bar(axs[0, 0], treat_means("median_bl_cm"), "Median body length", "cm")
bar(axs[0, 1], [100 * v for v in treat_means("cv_bl")],
    "Body-length variability (CV)", "% of median (higher = more length change)")
bar(axs[0, 2], treat_means("dom_freq_hz"), "Dominant rhythm frequency", "Hz")
bar(axs[1, 0], treat_means("dom_rel_power"), "Dominant-peak relative power", "fraction of in-band power")
bar(axs[1, 1], treat_means("elong_rate_per_min"), "Elongation event rate", "events / min")
bar(axs[1, 2], treat_means("head_tail_path_ratio"), "Head : tail path ratio", "ratio", hline=1.0)
fig.suptitle("Posture and crawling rhythm metrics per treatment (cm units); "
             "each bar = mean of the 2 leeches in that treatment; low-conf videos flagged", fontsize=13)
fig.tight_layout(); fig.savefig(OUT + "/posture_rhythm_barcharts.png", dpi=130); plt.close(fig)
print("DONE")
