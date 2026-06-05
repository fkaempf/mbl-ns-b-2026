#!/usr/bin/env python
"""Intricate cycle-resolved spatiotemporal figures (and optional movie) for the leech
ganglion GCaMP6m recording. Reads the raw movie, downsampled. No em dashes."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

ROOT = Path("/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project")
DER = ROOT / "analysis_calcium" / "derived"
OUT = ROOT / "analysis_calcium" / "plots"
MOV = ROOT / "data" / "calcium" / "2026-06-04" / "helobdella_LeechNo2_elavgcamp6m-17.tif"

fs = 2.882
DOM = 0.37151
PERIOD = 1 / DOM
gang = np.load(DER / "ganglion_signal.npy")
corr = np.load(DER / "correlation_image.npy")
T = gang.shape[0]
t = np.arange(T) / fs

# load movie downsampled 2x (700->350) into RAM as float32 (~170 MB)
mm = tifffile.memmap(MOV)
ds = mm[:, ::2, ::2].astype(np.float32)         # (T, 350, 350)
H = ds.shape[1]
F0 = np.percentile(ds, 20, axis=0) + 1e-3
dffmov = (ds - F0) / F0                          # per-pixel dF/F
tissue = gaussian_filter(corr[::2, ::2], 2) > np.percentile(corr, 70)

pk, _ = find_peaks(gang, distance=int(0.7 * PERIOD * fs), prominence=np.std(gang) * 0.4)


def clip(img, lo=2, hi=99):
    a, b = np.percentile(img, [lo, hi]); return np.clip(img, a, b)


# ---------- phase assignment ----------
phase = np.full(T, np.nan)
for i in range(len(pk) - 1):
    a, b = pk[i], pk[i + 1]
    phase[a:b] = np.linspace(0, 1, b - a, endpoint=False)
valid = ~np.isnan(phase)

# ---------- Figure 1: phase montage ----------
nb = 8                                            # matches ~7.75 frames/cycle, fills all bins
bins = np.linspace(0, 1, nb + 1)
imgs = []
for i in range(nb):
    sel = valid & (phase >= bins[i]) & (phase < bins[i + 1])
    imgs.append(dffmov[sel].mean(0) if sel.any() else np.full((H, H), np.nan))
imgs = np.array(imgs)
# show MODULATION: subtract each pixel's across-phase mean so the oscillation is visible
imgs_mod = imgs - np.nanmean(imgs, axis=0, keepdims=True)
vmax = np.nanpercentile(np.abs(imgs_mod[:, tissue]), 99)
fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
for i, ax in enumerate(axes.ravel()):
    im = ax.imshow(np.where(tissue, imgs_mod[i], np.nan), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"phase {bins[i]:.2f} ({bins[i]*PERIOD:.2f} s)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.02, label="dF/F modulation (deviation from cycle mean)")
fig.suptitle("Cycle-averaged spatial pattern through one 0.37 Hz cycle (8 phase bins)", fontsize=14)
fig.savefig(OUT / "fig_cycle_phase_montage.png", dpi=170)
plt.close(fig)

# ---------- Figure 2: kymograph along long axis ----------
yy, xx = np.where(tissue)
pca = PCA(2).fit(np.c_[xx, yy])
axis_coord = pca.transform(np.c_[xx, yy])[:, 0]
nbpos = 40
pos_bins = np.linspace(axis_coord.min(), axis_coord.max(), nbpos + 1)
binidx = np.clip(np.digitize(axis_coord, pos_bins) - 1, 0, nbpos - 1)
kymo = np.zeros((nbpos, T), np.float32)
flat = dffmov[:, yy, xx]                         # (T, n_tissue_pix)
for b in range(nbpos):
    m = binidx == b
    if m.any():
        kymo[b] = flat[:, m].mean(1)
fig, ax = plt.subplots(2, 1, figsize=(15, 8), height_ratios=[3, 2], constrained_layout=True)
im = ax[0].imshow(kymo, aspect="auto", cmap="magma", extent=[0, t[-1], nbpos, 0],
                  vmin=np.percentile(kymo, 2), vmax=np.percentile(kymo, 98))
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("position along ganglion")
ax[0].set_title("Kymograph: activity vs position along ganglion (vertical stripes = synchronous)")
plt.colorbar(im, ax=ax[0], fraction=0.02, label="dF/F")
m = (t > 60) & (t < 80)
im2 = ax[1].imshow(kymo[:, m], aspect="auto", cmap="magma",
                   extent=[60, 80, nbpos, 0],
                   vmin=np.percentile(kymo, 2), vmax=np.percentile(kymo, 98))
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("position"); ax[1].set_title("20 s zoom")
plt.colorbar(im2, ax=ax[1], fraction=0.02, label="dF/F")
fig.savefig(OUT / "fig_cycle_kymograph.png", dpi=170)
plt.close(fig)

# ---------- Figure 3: peak/trough gallery ----------
ncyc = 5
sel_pk = pk[np.linspace(2, len(pk) - 3, ncyc).astype(int)]
fig, axes = plt.subplots(ncyc, 3, figsize=(10, 3 * ncyc), constrained_layout=True)
vlo, vhi = np.percentile(dffmov[:, tissue], [2, 99])     # sequential range for raw dF/F
vd = np.percentile(np.abs(dffmov[:, tissue]), 95)        # diverging range for difference
for r, p in enumerate(sel_pk):
    trough = min(p + int(PERIOD * fs / 2), T - 1)
    pf = np.where(tissue, dffmov[p], np.nan)
    tf = np.where(tissue, dffmov[trough], np.nan)
    diff = pf - tf
    panels = [(pf, f"peak t={p/fs:.1f}s", "magma", dict(vmin=vlo, vmax=vhi)),
              (tf, "trough", "magma", dict(vmin=vlo, vmax=vhi)),
              (diff, "peak - trough", "RdBu_r", dict(vmin=-vd, vmax=vd))]
    for c, (img, title, cm, kw) in enumerate(panels):
        im = axes[r, c].imshow(img, cmap=cm, **kw)
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            axes[r, c].set_title(title, fontsize=10)
    plt.colorbar(im, ax=axes[r, 2], fraction=0.046)
fig.suptitle("Peak vs trough vs difference across 5 cycles (consistent modulation)", fontsize=13)
fig.savefig(OUT / "fig_cycle_peak_trough_gallery.png", dpi=170)
plt.close(fig)

# ---------- Figure 4: amplitude maps over 4 quarters ----------
fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
qs = np.array_split(np.arange(T), 4)
amaps = [np.where(tissue, dffmov[q].std(0), np.nan) for q in qs]
vmax = np.nanpercentile(amaps, 99)
for i, (ax, am, q) in enumerate(zip(axes, amaps, qs)):
    im = ax.imshow(am, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(f"quarter {i+1}: {q[0]/fs:.0f}-{q[-1]/fs:.0f} s", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.02, label="oscillation amplitude (std dF/F)")
fig.suptitle("Spatial amplitude of the rhythm across the 4 recording quarters (stability check)", fontsize=14)
fig.savefig(OUT / "fig_cycle_amplitude_map_over_time.png", dpi=170)
plt.close(fig)

# ---------- Figure 5 optional: mp4 of cycle-averaged sequence ----------
mp4_done = False
try:
    import cv2
    seq = np.concatenate([imgs_mod, imgs_mod, imgs_mod], 0)  # loop 3x
    vmax2 = np.nanpercentile(np.abs(imgs_mod[:, tissue]), 99)
    vw = cv2.VideoWriter(str(OUT / "fig_cycle_movie.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 6, (H, H))
    cmap = plt.get_cmap("RdBu_r")
    for fr in seq:
        norm = np.clip((fr + vmax2) / (2 * vmax2), 0, 1)
        norm = np.where(tissue, norm, 0.5)
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]
        vw.write(np.ascontiguousarray(rgb))
    vw.release()
    mp4_done = True
except Exception as e:
    print("mp4 skipped:", e)

print("OK cycle nb", nb, "kymo", kymo.shape, "cycles", len(pk), "mp4", mp4_done)
