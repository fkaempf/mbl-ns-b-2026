"""Load a suite2p run and turn pixels into region traces.

Pure / testable: no Qt, no figure I/O. The heavy array ops here are shared by
both the annotation tool (background images) and the analysis stage (dF/F).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import percentile_filter


@dataclass
class Run:
    """A loaded suite2p plane: the registered movie + its mean/shape."""
    movie: np.ndarray          # (T, Ly, Lx) float32, memmapped from data.bin
    mean_img: np.ndarray       # (Ly, Lx)
    fs: float                  # frame rate (Hz)
    plane_dir: Path


def load_run(run_dir: str | Path) -> Run:
    """Load plane0 of a suite2p run produced by scripts/run_suite2p.py.

    Reads the registered binary (data.bin) as a memmap, plus ops.npy for the
    mean image, frame shape, and frame rate.
    """
    plane = Path(run_dir)
    # Accept the run root (…/<stem>), the suite2p folder, or plane0 directly.
    for sub in ("plane0", "suite2p/plane0"):
        if (plane / "ops.npy").exists():
            break
        if (plane / sub / "ops.npy").exists():
            plane = plane / sub
            break
    # allow_pickle: ops.npy is suite2p's own output, produced locally by our
    # own pipeline run (scripts/run_suite2p.py) -- trusted, not external data.
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    Ly, Lx = ops["meanImg"].shape
    raw = np.memmap(plane / "data.bin", dtype="int16", mode="r")
    movie = np.asarray(raw, dtype="float32").reshape(-1, Ly, Lx)
    fs = float(ops.get("fs") or 1.0)
    return Run(movie=movie, mean_img=np.asarray(ops["meanImg"], "float32"),
               fs=fs, plane_dir=plane)


def local_correlation_image(movie: np.ndarray) -> np.ndarray:
    """Per-pixel mean Pearson correlation with its 4 nearest neighbours.

    Highlights spatially-coherent (real) activity; noise stays dark. This is
    the same map used as the annotation background.
    """
    m = movie - movie.mean(axis=0, keepdims=True)
    sd = m.std(axis=0)
    sd_safe = sd + 1e-6
    corr = np.zeros(movie.shape[1:], dtype="float32")
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neigh = np.roll(np.roll(m, dy, axis=1), dx, axis=2)
        neigh_sd = np.roll(np.roll(sd_safe, dy, axis=0), dx, axis=1)
        corr += (m * neigh).mean(axis=0) / (sd_safe * neigh_sd)
    return corr / 4.0


def seed_correlation_map(movie: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Per-pixel Pearson correlation of every pixel's time series with `seed`.

    seed : (T,) reference trace (e.g. a ganglion's mean dF/F). Returns (Ly, Lx).
    Shows which pixels co-vary with that seed region.
    """
    seed = np.asarray(seed, "float32")
    s = seed - seed.mean()
    s_sd = s.std() + 1e-6
    m = movie - movie.mean(axis=0, keepdims=True)
    sd = m.std(axis=0) + 1e-6
    return (m * s[:, None, None]).mean(axis=0) / (sd * s_sd)


def region_traces(movie: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Mean raw fluorescence inside each region, per frame.

    masks : (n_regions, Ly, Lx) boolean. Returns (n_regions, T).
    """
    out = np.empty((masks.shape[0], movie.shape[0]), dtype="float32")
    for i, mask in enumerate(masks):
        if not mask.any():
            out[i] = np.nan
            continue
        out[i] = movie[:, mask].mean(axis=1)
    return out


def rolling_baseline(F: np.ndarray, fs: float, win_s: float = 60.0,
                     percentile: float = 8.0) -> np.ndarray:
    """F0 via a sliding low-percentile filter (suite2p 'maximin'-style).

    Robust to slow drift and to sparse positive transients. F is (n, T).
    """
    win = max(3, int(round(win_s * fs)))
    return percentile_filter(F, percentile=percentile, size=(1, win), mode="nearest")


def dff(F: np.ndarray, fs: float, win_s: float = 60.0,
        percentile: float = 8.0) -> np.ndarray:
    """(F - F0) / F0 with a rolling-percentile baseline. F is (n_regions, T)."""
    F = np.atleast_2d(np.asarray(F, dtype="float32"))
    F0 = rolling_baseline(F, fs, win_s=win_s, percentile=percentile)
    F0 = np.maximum(F0, 1e-6)
    return (F - F0) / F0
