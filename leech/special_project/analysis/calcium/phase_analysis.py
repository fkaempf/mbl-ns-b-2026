"""Characterise the dominant global oscillation and its spatial phase.

The leech-cord movies are dominated by one near-synchronous oscillation. This
module (a) finds its frequency from the global mean (power spectrum), and
(b) maps each pixel's PHASE at that frequency -- revealing the spatial phase
gradient (e.g. anterior->posterior wave) that plain correlation hides.

Works per-pixel, so it needs NO ROI annotation -- usable across a whole batch.

  python -m analysis.calcium.phase_analysis <run_dir> [regions_dir] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import signals


def power_spectrum(trace: np.ndarray, fs: float):
    """One-sided power spectrum of a 1-D trace. Returns (freqs, power)."""
    x = np.asarray(trace, float)
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return f, p


def dominant_frequency(trace: np.ndarray, fs: float, fmin: float = 0.02):
    """Frequency of the largest spectral peak above fmin (skips DC/drift)."""
    f, p = power_spectrum(trace, fs)
    band = f >= fmin
    fb, pb = f[band], p[band]
    return float(fb[np.argmax(pb)]), (f, p)


def pixel_phase_amplitude(movie: np.ndarray, freq: float, fs: float):
    """Per-pixel Fourier coefficient at `freq`: returns (amplitude, phase).

    Projects each pixel's time series onto exp(-i 2pi f t); amplitude is the
    oscillation strength, phase (radians, -pi..pi) its timing.
    """
    T = movie.shape[0]
    t = np.arange(T)
    w = np.exp(-2j * np.pi * freq * t / fs)
    coef = np.zeros(movie.shape[1:], dtype="complex128")
    mean = movie.mean(axis=0)
    for k in range(T):
        coef += (movie[k] - mean) * w[k]
    coef *= 2.0 / T
    return np.abs(coef), np.angle(coef)


# ----------------------------------------------------------------- plotting ---
def _plot_spectrum(f, p, fpk, fs, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(f[1:], p[1:], lw=1)
    ax.axvline(fpk, color="r", ls="--", label=f"peak {fpk:.3f} Hz ({1/fpk:.2f}s)")
    ax.axvline(fs / 2, color="k", ls=":", alpha=0.5, label=f"Nyquist {fs/2:.2f} Hz")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power")
    ax.set_title("global-signal power spectrum"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_maps(amp, phase, mean_img, fpk, path):
    import matplotlib.pyplot as plt
    a = amp / (np.percentile(amp, 99.5) + 1e-9)
    alpha = np.clip(a, 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    ax[0].imshow(mean_img, cmap="gray"); ax[0].set_title("mean image"); ax[0].axis("off")
    im1 = ax[1].imshow(amp, cmap="inferno", vmax=np.percentile(amp, 99.5))
    ax[1].set_title(f"amplitude @ {fpk:.3f} Hz"); ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    ax[2].imshow(mean_img, cmap="gray")
    im2 = ax[2].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, alpha=alpha)
    ax[2].set_title("phase (rad) — colour = timing of the rhythm"); ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, label="phase")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_ganglion_phase(masks, names, phase, amp, mean_img, fpk, path):
    import matplotlib.pyplot as plt
    cx, cy, ph = [], [], []
    for m in masks:
        ys, xs = np.where(m)
        cx.append(xs.mean()); cy.append(ys.mean())
        # amplitude-weighted circular mean of phase inside the region
        z = (amp[m] * np.exp(1j * phase[m])).sum()
        ph.append(np.angle(z))
    cx, cy, ph = np.array(cx), np.array(cy), np.array(ph)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(mean_img, cmap="gray")
    sc = ax[0].scatter(cx, cy, c=ph, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                       s=260, edgecolor="w")
    for x, y, nm in zip(cx, cy, names):
        ax[0].text(x, y, nm[1:], color="w", ha="center", va="center", fontsize=8)
    ax[0].set_title(f"per-ganglion phase @ {fpk:.3f} Hz"); ax[0].axis("off")
    fig.colorbar(sc, ax=ax[0], fraction=0.046, label="phase (rad)")
    # phase vs anterior-posterior position (centroid x), unwrapped about G-leftmost
    order = np.argsort(cx)
    phu = np.unwrap(ph[order])
    ax[1].plot(cx[order], phu, "o-")
    for i in order:
        ax[1].annotate(names[i][1:], (cx[i], phu[list(order).index(i)]))
    ax[1].set_xlabel("anterior  <- centroid x ->  posterior")
    ax[1].set_ylabel("unwrapped phase (rad)")
    ax[1].set_title("phase gradient along the cord")
    # second y-axis: convert phase (rad) to time lag (ms) at this frequency
    period_ms = 1000.0 / fpk
    ax[1].secondary_yaxis(
        "right", functions=(lambda r: r / (2 * np.pi) * period_ms,
                             lambda t: t * 2 * np.pi / period_ms)
    ).set_ylabel(f"time lag (ms)   [period {period_ms:.0f} ms]")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return {"names": names, "centroid_x": cx.tolist(), "phase": ph.tolist()}


def analyze(run_dir, regions_dir=None, out_dir=None) -> dict:
    run = signals.load_run(run_dir)
    gm = run.movie.reshape(run.movie.shape[0], -1).mean(axis=1)
    fpk, (f, p) = dominant_frequency(gm, run.fs)
    amp, phase = pixel_phase_amplitude(run.movie, fpk, run.fs)

    out = Path(out_dir) if out_dir else Path(run_dir) / "phase"
    out.mkdir(parents=True, exist_ok=True)
    _plot_spectrum(f, p, fpk, run.fs, out / "global_spectrum.png")
    _plot_maps(amp, phase, run.mean_img, fpk, out / "phase_amplitude_maps.png")
    np.savez(out / "phase.npz", amp=amp, phase=phase, freq=fpk, fs=run.fs,
             global_trace=gm)

    result = {"freq_hz": fpk, "period_s": 1.0 / fpk, "out": str(out)}
    if regions_dir:
        regions_dir = Path(regions_dir)
        masks = np.load(regions_dir / "masks.npy")
        names = [g["name"] for g in json.loads(
            (regions_dir / "ganglia.json").read_text())["ganglia"]]
        gp = _plot_ganglion_phase(masks, names, phase, amp, run.mean_img, fpk,
                                  out / "ganglion_phase.png")
        result["ganglion_phase"] = gp
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Global-rhythm phase analysis")
    ap.add_argument("run_dir")
    ap.add_argument("regions_dir", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    r = analyze(args.run_dir, args.regions_dir, args.out)
    print(f"peak {r['freq_hz']:.3f} Hz ({r['period_s']:.2f}s) -> {r['out']}")


if __name__ == "__main__":
    main()
