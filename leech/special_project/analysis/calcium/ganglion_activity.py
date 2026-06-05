"""Stage 3: compare ganglion activity.

Pure math (correlation / lag / events) kept separate from the plotting so the
math can be unit-tested with synthetic signals. Entry point `analyze` ties it
to a run + saved regions and writes figures + numeric tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- pure math ---
def correlation_matrix(traces: np.ndarray) -> np.ndarray:
    """Pearson correlation between region dF/F traces. traces: (n, T) -> (n, n)."""
    return np.corrcoef(np.asarray(traces, dtype="float64"))


def regress_out_global(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove the common-mode (global mean) signal from each region.

    Widefield region-mean traces often share one dominant global signal
    (illumination, whole-prep rhythm) that makes every pair correlate ~1.
    For each region i: residual_i = trace_i - beta_i * global, with
    beta_i = cov(trace_i, global)/var(global). Returns (residuals, global).

    Per-region regression (not plain mean subtraction) avoids the
    sum-to-zero negative bias and scales each region's share of the global
    signal correctly. Returns (residuals (n,T), global (T,)).
    """
    traces = np.asarray(traces, dtype="float64")
    g = traces.mean(axis=0)
    gc = g - g.mean()
    denom = np.dot(gc, gc) + 1e-12
    beta = (traces - traces.mean(axis=1, keepdims=True)) @ gc / denom
    return traces - beta[:, None] * gc, g


def crosscorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Best integer lag (samples) of `b` relative to `a` within +/-max_lag.

    Positive lag => b follows a (a leads). Returns (lag, peak_correlation).
    """
    a = np.asarray(a, "float64"); b = np.asarray(b, "float64")
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    best_lag, best_r = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        # correlation of a[t] with b[t + lag]; peaks at lag = d when b[t]=a[t-d]
        if lag > 0:
            r = np.dot(a[:n - lag], b[lag:]) / (n - lag)
        elif lag < 0:
            r = np.dot(a[-lag:], b[:n + lag]) / (n + lag)
        else:
            r = np.dot(a, b) / n
        if r > best_r:
            best_r, best_lag = r, lag
    return best_lag, float(best_r)


def lag_matrix(traces: np.ndarray, fs: float, max_lag_s: float = 5.0) -> np.ndarray:
    """Pairwise lead/lag in seconds. Entry [i, j] = lag of j relative to i."""
    n = len(traces)
    max_lag = max(1, int(round(max_lag_s * fs)))
    out = np.zeros((n, n), dtype="float64")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lag, _ = crosscorr_lag(traces[i], traces[j], max_lag)
            out[i, j] = lag / fs
    return out


def detect_events(dff: np.ndarray, k: float = 3.0, min_sep_s: float = 1.0,
                  fs: float = 1.0) -> list[np.ndarray]:
    """Threshold-crossing event onsets per region.

    Threshold = k * robust noise (MAD) above 0. Returns a list (one array of
    onset sample-indices per region). Onsets closer than min_sep_s are merged.
    """
    dff = np.atleast_2d(np.asarray(dff, "float64"))
    min_sep = max(1, int(round(min_sep_s * fs)))
    events: list[np.ndarray] = []
    for trace in dff:
        mad = np.median(np.abs(trace - np.median(trace))) + 1e-12
        thresh = k * 1.4826 * mad
        above = trace > thresh
        onsets = np.where(above[1:] & ~above[:-1])[0] + 1
        if len(onsets):
            keep = [onsets[0]]
            for o in onsets[1:]:
                if o - keep[-1] >= min_sep:
                    keep.append(o)
            onsets = np.array(keep)
        events.append(onsets)
    return events


# ----------------------------------------------------------------- plotting ---
def _save_correlation(corr, names, path):
    import matplotlib.pyplot as plt
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="k")
    ax.set_title("ganglion × ganglion correlation (dF/F)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _save_traces_raster(dff, names, fs, path):
    import matplotlib.pyplot as plt
    n, T = dff.shape
    t = np.arange(T) / fs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 1.0 * n + 3),
                                   gridspec_kw={"height_ratios": [2, 1]})
    offset = np.nanmax(np.nanstd(dff, axis=1)) * 6 + 1e-6
    for i in range(n):
        ax1.plot(t, dff[i] + i * offset, lw=0.8)
        ax1.text(t[-1], i * offset, " " + names[i], va="center", fontsize=8)
    ax1.set_xlabel("time (s)"); ax1.set_yticks([]); ax1.set_title("per-ganglion dF/F")
    im = ax2.imshow(dff, aspect="auto", cmap="magma",
                    extent=[0, t[-1], n - 0.5, -0.5],
                    vmin=0, vmax=np.nanpercentile(dff, 99))
    ax2.set_yticks(range(n)); ax2.set_yticklabels(names)
    ax2.set_xlabel("time (s)"); ax2.set_title("dF/F raster")
    fig.colorbar(im, ax=ax2, fraction=0.03, label="dF/F")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _save_lag(lags, names, path):
    import matplotlib.pyplot as plt
    n = len(names)
    vmax = np.abs(lags).max() + 1e-6
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1))
    im = ax.imshow(lags, cmap="PuOr", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{lags[i, j]:+.2f}", ha="center", va="center",
                        fontsize=7)
    ax.set_title("lead/lag of column rel. to row (s)\n(+ = column follows row)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="lag (s)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _save_local_corr(corr_img, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(corr_img, cmap="inferno", vmin=0,
                   vmax=np.percentile(corr_img, 99))
    ax.set_title("local correlation image (pixel ↔ neighbours)"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="mean neighbour r")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _save_seed_maps(maps, mean_img, names, path):
    import matplotlib.pyplot as plt
    n = len(names)
    cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows),
                             squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]; ax.axis("off")
        if i >= n:
            continue
        ax.imshow(mean_img, cmap="gray")
        ax.imshow(maps[i], cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.6)
        ax.set_title(f"seed: {names[i]}", fontsize=9)
    fig.suptitle("per-ganglion seed correlation maps (pixel ↔ ganglion trace)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _save_events(dff, events, names, fs, path):
    import matplotlib.pyplot as plt
    n, T = dff.shape
    fig, ax = plt.subplots(figsize=(11, 0.5 * n + 2))
    for i, onsets in enumerate(events):
        ax.scatter(onsets / fs, np.full(len(onsets), i), marker="|", s=200)
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlim(0, T / fs); ax.set_xlabel("time (s)")
    ax.set_title("detected calcium events (onsets)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------- entrypoint ---
def analyze(run_dir: str | Path, regions_dir: str | Path,
            out_dir: str | Path, max_lag_s: float = 5.0,
            remove_global: bool = False) -> dict:
    """Full stage-3: load regions + run, compute dF/F, write figures + tables.

    remove_global: if True, regress out the shared common-mode signal before
    correlation/lag/events, exposing ganglion-specific structure.
    """
    from . import signals

    run = signals.load_run(run_dir)
    regions_dir = Path(regions_dir)
    masks = np.load(regions_dir / "masks.npy")            # (n, Ly, Lx) bool
    meta = json.loads((regions_dir / "ganglia.json").read_text())
    names = [g["name"] for g in meta["ganglia"]]

    F = signals.region_traces(run.movie, masks)
    dff = signals.dff(F, run.fs)
    if remove_global:
        dff, _global = regress_out_global(dff)

    corr = correlation_matrix(dff)
    lags = lag_matrix(dff, run.fs, max_lag_s=max_lag_s)
    events = detect_events(dff, fs=run.fs)

    # pixel-level correlation: global local-corr image + per-ganglion seed maps
    local_corr = signals.local_correlation_image(run.movie)
    seed_maps = np.stack([signals.seed_correlation_map(run.movie, dff[i])
                          for i in range(len(names))])

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    _save_correlation(corr, names, out / "correlation_heatmap.png")
    _save_traces_raster(dff, names, run.fs, out / "traces_raster.png")
    _save_lag(lags, names, out / "lag_matrix.png")
    _save_events(dff, events, names, run.fs, out / "events.png")
    _save_local_corr(local_corr, out / "local_correlation.png")
    _save_seed_maps(seed_maps, run.mean_img, names, out / "seed_correlation_maps.png")

    np.savez(out / "ganglion_activity.npz", dff=dff, F=F, corr=corr, lags=lags,
             local_corr=local_corr, seed_maps=seed_maps,
             names=np.array(names), fs=run.fs)
    t = np.arange(dff.shape[1]) / run.fs
    header = "time_s," + ",".join(names)
    np.savetxt(out / "ganglion_dff.csv", np.column_stack([t, dff.T]),
               delimiter=",", header=header, comments="")
    return {"names": names, "n_events": [len(e) for e in events],
            "out_dir": str(out)}


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Ganglion activity analysis")
    ap.add_argument("run_dir", help="suite2p run dir")
    ap.add_argument("regions_dir", help="dir with masks.npy + ganglia.json")
    ap.add_argument("--out", default=None, help="output dir for figures")
    ap.add_argument("--max-lag-s", type=float, default=5.0)
    ap.add_argument("--remove-global", action="store_true",
                    help="regress out the shared common-mode signal first")
    args = ap.parse_args(argv)
    out = args.out or str(Path(args.regions_dir) /
                          ("results_residual" if args.remove_global else "results"))
    res = analyze(args.run_dir, args.regions_dir, out, max_lag_s=args.max_lag_s,
                  remove_global=args.remove_global)
    print(f"Wrote figures to {res['out_dir']}")
    for name, ne in zip(res["names"], res["n_events"]):
        print(f"  {name}: {ne} events")


if __name__ == "__main__":
    main()
