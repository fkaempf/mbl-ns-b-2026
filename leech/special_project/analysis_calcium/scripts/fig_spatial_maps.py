#!/usr/bin/env python
"""Intricate spatial-map figures for the leech ganglion GCaMP6m recording.
Reuses derived arrays. No em dashes."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

ROOT = Path("/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project")
DER = ROOT / "analysis_calcium" / "derived"
OUT = ROOT / "analysis_calcium" / "plots"

fs = 2.882
DOM = 0.37151
dff = np.load(DER / "traces_dff.npy")          # (180, T)
corr = np.load(DER / "correlation_image.npy")  # 700x700
mean_img = np.load(DER / "mean_image.npy")
masks = np.load(DER / "roi_masks.npy")         # labeled 700x700, label=id+1
pw = np.load(DER / "rhythm_power_map.npy")      # 350x350
phs = np.load(DER / "phase_map.npy")            # 350x350
cent = pd.read_csv(DER / "roi_centroids.csv")
T = dff.shape[1]
t = np.arange(T) / fs

pw_full = resize(pw, (700, 700), order=1, preserve_range=True)
phs_full = resize(phs, (700, 700), order=0, preserve_range=True)
tissue = gaussian_filter(corr, 3) > np.percentile(corr, 70)


def clip(img, lo=1, hi=99):
    a, b = np.percentile(img, [lo, hi])
    return np.clip(img, a, b)


# per-ROI complex Fourier coefficient at DOM -> amplitude + phase
k = DOM / fs
n = np.arange(T)
basis = np.exp(-1j * 2 * np.pi * k * n)
coef = (dff - dff.mean(1, keepdims=True)) @ basis / T
roi_amp_f = np.abs(coef)
roi_phase = np.angle(coef)
roi_std = dff.std(1)
roi_mean = dff.mean(1)
roi_peak = np.percentile(dff, 99, axis=1)
ys, xs = cent["y"].values, cent["x"].values

# ---------- Figure 1: power & phase 2x2 ----------
fig, ax = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
ax[0, 0].imshow(clip(corr), cmap="magma")
ax[0, 0].contour(tissue, levels=[0.5], colors="cyan", linewidths=0.8)
ax[0, 0].set_title("Correlation image with tissue outline")
im = ax[0, 1].imshow(np.log10(pw_full + pw_full[pw_full > 0].min()), cmap="inferno")
ax[0, 1].contour(tissue, levels=[0.5], colors="cyan", linewidths=0.6)
ax[0, 1].set_title("Rhythm power at 0.37 Hz (log)")
plt.colorbar(im, ax=ax[0, 1], fraction=0.046, label="log power")
total = resize(np.load(DER / "rhythm_power_map.npy"), (700, 700), preserve_range=True)
norm = pw_full / (gaussian_filter(pw_full, 30) + 1e-9)
im = ax[1, 0].imshow(clip(pw_full, 1, 99), cmap="viridis")
ax[1, 0].contour(tissue, levels=[0.5], colors="w", linewidths=0.5)
ax[1, 0].set_title("Rhythm power (linear)")
plt.colorbar(im, ax=ax[1, 0], fraction=0.046, label="power")
phm = np.ma.masked_where(~(tissue & (pw_full > np.percentile(pw_full[tissue], 40))), phs_full)
im = ax[1, 1].imshow(phm, cmap="twilight", vmin=-np.pi, vmax=np.pi)
ax[1, 1].contour(tissue, levels=[0.5], colors="k", linewidths=0.5)
ax[1, 1].set_title("Phase at 0.37 Hz (high-power tissue)")
plt.colorbar(im, ax=ax[1, 1], fraction=0.046, label="phase (rad)")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.savefig(OUT / "fig_spatial_power_phase.png", dpi=170)
plt.close(fig)

# ---------- Figure 2: ROI property maps ----------
fig, ax = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
props = [(roi_std, "oscillation amplitude (std dF/F)", "viridis"),
         (roi_mean, "mean dF/F", "magma"),
         (roi_phase, "phase at 0.37 Hz (rad)", "twilight"),
         (roi_peak, "peak dF/F (99th pct)", "inferno")]
for a, (val, title, cm) in zip(ax.ravel(), props):
    a.imshow(clip(mean_img), cmap="gray")
    vmin, vmax = (-np.pi, np.pi) if "phase" in title else (np.percentile(val, 2), np.percentile(val, 98))
    sc = a.scatter(xs, ys, c=val, cmap=cm, s=55, vmin=vmin, vmax=vmax, edgecolors="w", linewidths=0.3)
    a.set_title(title); a.set_xticks([]); a.set_yticks([])
    plt.colorbar(sc, ax=a, fraction=0.046)
fig.suptitle("180 ROI properties on the mean image", fontsize=14)
fig.savefig(OUT / "fig_spatial_roi_properties.png", dpi=170)
plt.close(fig)

# ---------- Figure 3: phase gradient / traveling wave test ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
mask = tissue & (pw_full > np.percentile(pw_full[tissue], 50))
uph = phs_full.copy()
gy, gx = np.gradient(gaussian_filter(uph, 5))
ax[0].imshow(np.ma.masked_where(~mask, phs_full), cmap="twilight", vmin=-np.pi, vmax=np.pi)
step = 25
Y, X = np.mgrid[0:700, 0:700]
m2 = mask & (Y % step == 0) & (X % step == 0)
ax[0].quiver(X[m2], Y[m2], gx[m2], -gy[m2], color="k", scale=3, width=0.003)
ax[0].set_title("Phase map with phase-gradient quiver"); ax[0].set_xticks([]); ax[0].set_yticks([])
# ROI phase vs long axis
pca = PCA(2).fit(np.c_[xs, ys])
axis1 = pca.transform(np.c_[xs, ys])[:, 0]
ax[1].scatter(axis1, np.unwrap(np.sort(roi_phase)[np.argsort(np.argsort(axis1))]) if False else roi_phase,
              s=30, c="#1b5e20")
slope = np.polyfit(axis1, roi_phase, 1)[0]
xl = np.array([axis1.min(), axis1.max()])
ax[1].plot(xl, np.polyval(np.polyfit(axis1, roi_phase, 1), xl), color="crimson",
           label=f"slope {slope:.4f} rad/px")
ax[1].set_xlabel("position along ganglion long axis (px)")
ax[1].set_ylabel("ROI phase (rad)")
ax[1].set_title("ROI phase vs long axis (flat = synchronous)")
ax[1].legend(fontsize=9); ax[1].spines[["top", "right"]].set_visible(False)
axp = fig.add_subplot(1, 3, 3, projection="polar")
axp.hist(roi_phase, bins=24, color="#4a148c", alpha=0.8)
R = np.abs(np.mean(np.exp(1j * roi_phase)))
axp.set_title(f"ROI phase distribution (R={R:.2f})")
fig.savefig(OUT / "fig_spatial_phase_gradient.png", dpi=170)
plt.close(fig)

# ---------- Figure 4: seed correlation ----------
C = np.corrcoef(dff)
# pick 3 seeds spread along axis
order_ax = np.argsort(axis1)
seeds = [order_ax[10], order_ax[len(order_ax) // 2], order_ax[-10]]
fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
for a, sd in zip(ax, seeds):
    a.imshow(clip(mean_img), cmap="gray")
    sc = a.scatter(xs, ys, c=C[sd], cmap="coolwarm", vmin=-1, vmax=1, s=55,
                   edgecolors="k", linewidths=0.2)
    a.scatter([xs[sd]], [ys[sd]], marker="*", s=300, color="lime", edgecolors="k")
    a.set_title(f"correlation to seed patch {sd}"); a.set_xticks([]); a.set_yticks([])
    plt.colorbar(sc, ax=a, fraction=0.046, label="corr")
fig.suptitle("Seed-based dF/F correlation maps", fontsize=14)
fig.savefig(OUT / "fig_spatial_seed_correlation.png", dpi=170)
plt.close(fig)

# ---------- Figure 5: correlation vs distance ----------
iu = np.triu_indices(len(dff), 1)
dist = np.sqrt((xs[iu[0]] - xs[iu[1]]) ** 2 + (ys[iu[0]] - ys[iu[1]]) ** 2)
cc = C[iu]
fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
ax.scatter(dist, cc, s=4, alpha=0.15, color="grey")
bins = np.linspace(0, dist.max(), 18)
bc = 0.5 * (bins[1:] + bins[:-1])
bm = [cc[(dist >= bins[i]) & (dist < bins[i + 1])].mean() for i in range(len(bins) - 1)]
ax.plot(bc, bm, "o-", color="crimson", lw=2, label="binned mean")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("pairwise distance (px)"); ax.set_ylabel("dF/F correlation")
ax.set_title(f"Correlation vs distance (overall mean r = {cc.mean():.2f})")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "fig_spatial_distance_correlation.png", dpi=170)
plt.close(fig)

print("OK spatial phaseR", round(R, 3), "slope", round(slope, 5),
      "meanpaircorr", round(cc.mean(), 3))
