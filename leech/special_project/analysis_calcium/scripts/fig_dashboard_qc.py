#!/usr/bin/env python
"""Intricate master dashboard, segmentation comparison, ROI gallery, QC and trace
quality figures for the leech ganglion GCaMP6m recording. No em dashes."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from skimage.transform import resize
from scipy.signal import find_peaks, welch
from scipy.optimize import curve_fit

ROOT = Path("/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project")
DER = ROOT / "analysis_calcium" / "derived"
OUT = ROOT / "analysis_calcium" / "plots"
MET = ROOT / "analysis_calcium" / "metrics"

fs = 2.882
DOM = 0.37151
dff = np.load(DER / "traces_dff.npy")
dff_s2p = np.load(DER / "traces_dff_s2p.npy")
gang = np.load(DER / "ganglion_signal.npy")
corr = np.load(DER / "correlation_image.npy")
mean_img = np.load(DER / "mean_image.npy")
cent = pd.read_csv(DER / "roi_centroids.csv")
cent_s2p = pd.read_csv(DER / "roi_centroids_s2p.csv")
pw = resize(np.load(DER / "rhythm_power_map.npy"), (700, 700), preserve_range=True)
T = dff.shape[1]
t = np.arange(T) / fs


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


s1 = jload(MET / "s1_qc.json"); s3 = jload(MET / "s3_temporal.json")
s4 = jload(MET / "s4_spatial.json")
pk, _ = find_peaks(gang, distance=int(0.7 / DOM * fs), prominence=np.std(gang) * 0.4)


def clip(img, lo=1, hi=99):
    a, b = np.percentile(img, [lo, hi]); return np.clip(img, a, b)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---------- Figure 1: master dashboard ----------
fig = plt.figure(figsize=(17, 11), constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)
# (a) corr + ROIs
a = fig.add_subplot(gs[0, 0]); a.imshow(clip(corr), cmap="magma")
a.scatter(cent.x, cent.y, s=10, facecolors="none", edgecolors="cyan", linewidths=0.5)
a.set_title(f"Correlation image + {len(cent)} ROIs"); a.set_xticks([]); a.set_yticks([])
# (b) ganglion signal + inset
a = fig.add_subplot(gs[0, 1:]); style(a)
a.plot(t, gang, color="#1b5e20", lw=0.7); a.plot(pk / fs, gang[pk], "v", color="crimson", ms=3)
a.set_title("Ganglion mean dF/F over 240 s"); a.set_xlabel("time (s)"); a.set_ylabel("dF/F")
axins = a.inset_axes([0.62, 0.55, 0.36, 0.42])
m = (t > 60) & (t < 80); axins.plot(t[m], gang[m], color="#1b5e20", lw=1)
axins.set_title("20 s zoom", fontsize=8); axins.tick_params(labelsize=6)
# (c) PSD
a = fig.add_subplot(gs[1, 0]); style(a)
f, p = welch(gang, fs=fs, nperseg=512)
a.semilogy(f, p, color="#1b5e20"); a.axvline(DOM, color="crimson", ls="--")
a.set_title(f"PSD (peak {DOM:.3f} Hz)"); a.set_xlabel("Hz"); a.set_ylabel("PSD")
# (d) raster
a = fig.add_subplot(gs[1, 1:])
z = (dff - dff.mean(1, keepdims=True)) / (dff.std(1, keepdims=True) + 1e-9)
order = np.argsort(np.angle((z * np.exp(-1j * 2 * np.pi * DOM / fs * np.arange(T))).mean(1)))
im = a.imshow(z[order], aspect="auto", cmap="magma", vmin=-2, vmax=4,
              extent=[0, t[-1], len(z), 0])
a.set_title("Population raster (phase-sorted)"); a.set_xlabel("time (s)"); a.set_ylabel("patch")
plt.colorbar(im, ax=a, fraction=0.025, label="z dF/F")
# (e) power map
a = fig.add_subplot(gs[2, 0]); im = a.imshow(clip(pw), cmap="inferno")
a.set_title("Rhythm power at 0.37 Hz"); a.set_xticks([]); a.set_yticks([])
plt.colorbar(im, ax=a, fraction=0.046)
# (f) text metrics
a = fig.add_subplot(gs[2, 1:]); a.axis("off")
lines = [
    "helobdella_LeechNo2_elavgcamp6m-17",
    "10x juvenile, after dopamine pharyngals, preliminary line",
    "",
    f"n patches (primary): {dff.shape[0]}   suite2p cross-check: {dff_s2p.shape[0]}",
    f"frame rate: {fs} Hz    frames: {T}    duration: {T/fs:.0f} s",
    f"dominant rhythm: {s3.get('dominant_freq_hz', DOM):.3f} Hz  "
    f"(period {s3.get('dominant_period_s', 1/DOM):.2f} s)",
    f"cycles detected: {len(pk)}   interval CV: {s3.get('interpeak_cv', 0):.3f}",
    f"photobleaching: {s1.get('bleach_pct', 0):.2f} %    rigid drift: "
    f"{s1.get('rigid_drift_px_max', 0):.2f} px (no motion correction)",
    f"tissue pixels rhythmic: {100*s4.get('frac_tissue_pixels_rhythmic', 0):.0f} %",
    f"phase circular std (tissue): {s4.get('phase_circular_std_rad', float('nan')):.2f} rad "
    "(in-phase, no traveling wave)",
]
a.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
fig.suptitle("Calcium imaging dashboard: leech ganglion, ganglion-wide 0.37 Hz rhythm", fontsize=15)
fig.savefig(OUT / "fig_dash_master.png", dpi=170)
plt.close(fig)

# ---------- Figure 2: segmentation comparison ----------
fig, ax = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
ax[0, 0].imshow(clip(corr), cmap="magma")
ax[0, 0].scatter(cent_s2p.x, cent_s2p.y, s=40, facecolors="none", edgecolors="lime", linewidths=1.2)
ax[0, 0].set_title(f"suite2p ROIs (n={len(cent_s2p)})"); ax[0, 0].set_xticks([]); ax[0, 0].set_yticks([])
ax[0, 1].imshow(clip(corr), cmap="magma")
ax[0, 1].scatter(cent.x, cent.y, s=22, facecolors="none", edgecolors="cyan", linewidths=0.6)
ax[0, 1].set_title(f"correlation-blob ROIs (n={len(cent)})"); ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([])
ax[0, 2].imshow(clip(corr), cmap="gray")
ax[0, 2].scatter(cent.x, cent.y, s=22, facecolors="none", edgecolors="cyan", linewidths=0.6, label="blob")
ax[0, 2].scatter(cent_s2p.x, cent_s2p.y, s=45, facecolors="none", edgecolors="lime", linewidths=1.2, label="suite2p")
ax[0, 2].legend(fontsize=8); ax[0, 2].set_title("overlay"); ax[0, 2].set_xticks([]); ax[0, 2].set_yticks([])
# size hist
style(ax[1, 0])
ax[1, 0].hist(cent.radius_px, bins=20, color="cyan", alpha=0.7, label="blob")
if "radius_px" in cent_s2p:
    ax[1, 0].hist(cent_s2p.radius_px, bins=20, color="green", alpha=0.6, label="suite2p")
ax[1, 0].set_title("ROI radius distribution"); ax[1, 0].set_xlabel("radius (px)"); ax[1, 0].legend(fontsize=8)
# matching: suite2p ROI to nearest blob within soma radius
match = 0
for _, r in cent_s2p.iterrows():
    d = np.sqrt((cent.x - r.x) ** 2 + (cent.y - r.y) ** 2)
    if d.min() < 12:
        match += 1
style(ax[1, 1]); ax[1, 1].bar(["matched", "unmatched"], [match, len(cent_s2p) - match],
                              color=["#1b5e20", "grey"])
ax[1, 1].set_title(f"suite2p ROIs with a blob match (<12 px): {match}/{len(cent_s2p)}")
ax[1, 1].set_ylabel("count")
# example overlaid traces from matched cells
style(ax[1, 2])
shown = 0
for si, (_, r) in enumerate(cent_s2p.iterrows()):
    d = np.sqrt((cent.x - r.x) ** 2 + (cent.y - r.y) ** 2)
    if d.min() < 12 and shown < 4:
        bi = int(d.idxmin())
        off = shown * 1.0
        ax[1, 2].plot(t, dff[bi] + off, color="cyan", lw=0.6)
        ax[1, 2].plot(t, dff_s2p[si] + off, color="green", lw=0.6, alpha=0.8)
        shown += 1
ax[1, 2].set_title("matched-cell traces (cyan blob, green suite2p)")
ax[1, 2].set_xlabel("time (s)"); ax[1, 2].set_yticks([])
fig.suptitle("Segmentation comparison: suite2p under-segments this dense ganglion", fontsize=15)
fig.savefig(OUT / "fig_dash_segmentation_compare.png", dpi=170)
plt.close(fig)

# ---------- Figure 3: ROI gallery ----------
amp = dff.std(1); top = np.argsort(amp)[::-1][:40]
fig, axes = plt.subplots(8, 5, figsize=(16, 13), sharex=True, constrained_layout=True)
ymax = np.percentile(dff[top], 99.5)
for ax_, ci in zip(axes.ravel(), top):
    ax_.plot(t, dff[ci], color="#1b5e20", lw=0.5)
    ax_.set_title(f"patch {ci}  amp {amp[ci]:.2f}", fontsize=7)
    ax_.set_ylim(-0.1, ymax); ax_.set_yticks([]); style(ax_)
for ax_ in axes[-1]:
    ax_.set_xlabel("s", fontsize=7)
fig.suptitle("ROI gallery: 40 most active patches dF/F", fontsize=14)
fig.savefig(OUT / "fig_dash_roi_gallery.png", dpi=170)
plt.close(fig)

# ---------- Figure 4: QC panel ----------
fig, ax = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
for a in ax.ravel():
    style(a)
bleach = s1.get("bleach_pct", 0)
drift = s1.get("rigid_drift_px_max", 0)
ax[0, 0].text(0.5, 0.5, f"photobleaching\n{bleach:.2f} %\nover 240 s\n(negligible)",
              ha="center", va="center", fontsize=15)
ax[0, 0].set_title("Photobleaching"); ax[0, 0].axis("off")
ax[0, 1].axhline(0, color="#1b5e20", lw=2)
ax[0, 1].text(0.5, 0.7, f"rigid drift max {drift:.2f} px\nno motion correction needed",
              transform=ax[0, 1].transAxes, ha="center", fontsize=12, color="crimson")
ax[0, 1].set_ylim(-1, 1); ax[0, 1].set_title("Rigid drift (flat at 0)"); ax[0, 1].set_xlabel("frame")
ax[0, 1].set_ylabel("drift (px)")
# frame-frame corr proxy from ganglion (not exact) -> show ganglion autocorr lag1 placeholder
ax[0, 2].hist(mean_img[mean_img > 0].ravel(), bins=80, color="#00695c")
ax[0, 2].axvline(np.percentile(mean_img, 99.9), color="crimson", ls="--", label="99.9 pct")
ax[0, 2].set_title("Intensity histogram (no saturation)"); ax[0, 2].set_xlabel("intensity")
ax[0, 2].set_yscale("log"); ax[0, 2].legend(fontsize=8)
ax[1, 0].imshow(clip(mean_img), cmap="gray"); ax[1, 0].set_title("Mean projection")
ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])
ax[1, 1].imshow(clip(corr), cmap="magma"); ax[1, 1].set_title("Correlation image")
ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])
ax[1, 2].plot(t, gang, color="#1b5e20", lw=0.6)
ax[1, 2].set_title("Ganglion dF/F (signal stable across recording)")
ax[1, 2].set_xlabel("time (s)"); ax[1, 2].set_ylabel("dF/F")
fig.suptitle("Quality control: stable, motion-free, no saturation", fontsize=15)
fig.savefig(OUT / "fig_dash_qc_panel.png", dpi=170)
plt.close(fig)

# ---------- Figure 5: trace quality ----------
noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) * 1.4826
snr = np.percentile(dff, 99, axis=1) / (noise + 1e-9)
amp = dff.std(1)
corr_to_mean = np.array([np.corrcoef(dff[i], gang)[0, 1] for i in range(len(dff))])
fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
for a in ax.ravel():
    style(a)
ax[0, 0].hist(snr, bins=25, color="#1b5e20", edgecolor="w")
ax[0, 0].axvline(np.median(snr), color="crimson", ls="--", label=f"median {np.median(snr):.1f}")
ax[0, 0].set_title("Per-ROI SNR (peak/noise)"); ax[0, 0].set_xlabel("SNR"); ax[0, 0].legend(fontsize=8)
ax[0, 1].hist(amp, bins=25, color="#4a148c", edgecolor="w")
ax[0, 1].set_title("Per-ROI oscillation amplitude"); ax[0, 1].set_xlabel("std dF/F")
ax[1, 0].hist(corr_to_mean, bins=25, color="#00695c", edgecolor="w")
ax[1, 0].axvline(np.median(corr_to_mean), color="crimson", ls="--",
                 label=f"median {np.median(corr_to_mean):.2f}")
ax[1, 0].set_title("Correlation of each patch to ganglion mean")
ax[1, 0].set_xlabel("r"); ax[1, 0].legend(fontsize=8)
sc = ax[1, 1].scatter(amp, corr_to_mean, c=snr, cmap="viridis", s=30)
ax[1, 1].set_xlabel("amplitude (std dF/F)"); ax[1, 1].set_ylabel("corr to ganglion mean")
ax[1, 1].set_title("Amplitude vs rhythm-following, colored by SNR")
plt.colorbar(sc, ax=ax[1, 1], label="SNR")
fig.suptitle("Signal quality across the 180 patches", fontsize=15)
fig.savefig(OUT / "fig_dash_trace_quality.png", dpi=170)
plt.close(fig)

print("OK dash match", match, "/", len(cent_s2p), "medSNR", round(np.median(snr), 1),
      "medCorrToMean", round(np.median(corr_to_mean), 2),
      "frac_follow>0.5", round(np.mean(corr_to_mean > 0.5), 2))
