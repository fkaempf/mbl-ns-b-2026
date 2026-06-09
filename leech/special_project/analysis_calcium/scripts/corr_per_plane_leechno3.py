#!/usr/bin/env python
"""Per-pixel local correlation image for each plane of the LeechNo3 dual-label TIFs.

Each TIF is a hyperstack T,Z,Y,X with Z=2 planes (the two channels of the
elavgcamp6m + palm-mCherry prep: GCaMP6m functional + palm-mCherry structural).
For every plane we compute the classic correlation image: each pixel's value is the
mean Pearson correlation (over time) of its trace with its up-to-8 spatial
neighbors. Active, co-fluctuating tissue lights up; static structure / flat noise
stays dark. We save one PNG per (file, plane) plus a combined overview per file.
No em dashes."""
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project")
DATA = BASE / "data/calcium/drive-download-20260605T213726Z-3-001"
OUT = BASE / "analysis_calcium/leechno3_correlation"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    "helobdella_LeechNo3_elavgcamp6m+palmmcherry-05.tif",
    "helobdella_LeechNo3_elavgcamp6m+palmmcherry-06_dopa.tif",
]


def local_correlation(mov):
    """mov: (T,Y,X) float32. Return (Y,X) mean-neighbor correlation image."""
    T, H, W = mov.shape
    m = mov.mean(0)
    s = mov.std(0)
    s[s == 0] = 1.0
    N = (mov - m) / s                       # zero mean, unit (population) std over time
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            y0s, y1s = max(0, dy), H + min(0, dy)
            x0s, x1s = max(0, dx), W + min(0, dx)
            y0d, y1d = max(0, -dy), H + min(0, -dy)
            x0d, x1d = max(0, -dx), W + min(0, -dx)
            a = N[:, y0s:y1s, x0s:x1s]
            b = N[:, y0d:y1d, x0d:x1d]
            c = (a * b).sum(0) / T           # Pearson corr (N already normalized)
            acc[y0s:y1s, x0s:x1s] += c
            cnt[y0s:y1s, x0s:x1s] += 1
    return acc / np.maximum(cnt, 1)


def save_corr_png(corr, path, title):
    lo, hi = np.percentile(corr, [1, 99.5])
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(corr, cmap="inferno", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="mean neighbor correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


for fname in FILES:
    stem = Path(fname).stem
    print(f"\n=== {fname} ===", flush=True)
    arr = tifffile.imread(DATA / fname)          # (T, Z, Y, X) uint8
    if arr.ndim == 3:                            # safety: (T,Y,X) single plane
        arr = arr[:, None]
    T, Z, H, W = arr.shape
    print(f"  T={T} Z(planes)={Z} {H}x{W}", flush=True)

    corrs, stats = [], []
    for z in range(Z):
        mov = arr[:, z].astype(np.float32)
        # plane temporal dynamics, to flag which plane is the functional (GCaMP) one
        tstd = mov.std(0)
        dyn = float(np.median(tstd[mov.mean(0) > mov.mean(0).mean()]))  # median std in bright tissue
        corr = local_correlation(mov)
        corrs.append(corr)
        frac_hi = float(np.mean(corr > 0.3))
        stats.append((dyn, frac_hi))
        print(f"  plane {z}: median tissue temporal std {dyn:.2f}, "
              f"frac pixels corr>0.3 = {100*frac_hi:.1f}%", flush=True)
        save_corr_png(corr, OUT / f"corr_{stem}_plane{z}.png",
                      f"{stem}\nplane {z}: per-pixel local correlation image")
        del mov

    # likely-functional plane = higher fraction of correlated pixels
    func_plane = int(np.argmax([s[1] for s in stats]))

    # combined overview (mean image + correlation image per plane)
    fig, axes = plt.subplots(Z, 2, figsize=(11, 5.4 * Z))
    axes = np.atleast_2d(axes)
    for z in range(Z):
        mean_img = arr[:, z].astype(np.float32).mean(0)
        ax0 = axes[z, 0]
        ax0.imshow(mean_img, cmap="gray")
        ax0.set_title(f"plane {z}: mean intensity", fontsize=10)
        ax0.axis("off")
        ax1 = axes[z, 1]
        lo, hi = np.percentile(corrs[z], [1, 99.5])
        im = ax1.imshow(corrs[z], cmap="inferno", vmin=lo, vmax=hi)
        ax1.set_title(f"plane {z}: correlation image  (GCaMP z-plane)", fontsize=10)
        ax1.axis("off")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    fig.suptitle(f"{stem}: per-pixel local correlation image per plane", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / f"corr_overview_{stem}.png", dpi=140)
    plt.close(fig)
    del arr
    print(f"  wrote corr_{stem}_plane*.png + corr_overview_{stem}.png", flush=True)

print(f"\nDONE -> {OUT}")
