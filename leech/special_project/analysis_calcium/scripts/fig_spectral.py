#!/usr/bin/env python
"""Intricate spectral / time-frequency figures for the leech ganglion GCaMP6m
recording. Reuses derived arrays. No em dashes."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch, spectrogram, coherence


def morlet_cwt(sig, freqs, fs, w=6.0):
    """Manual Morlet continuous wavelet transform -> power (len(freqs), len(sig))."""
    n = len(sig)
    out = np.empty((len(freqs), n))
    sig = sig - sig.mean()
    for i, fr in enumerate(freqs):
        s = w * fs / (2 * np.pi * fr)          # width in samples
        half = int(np.ceil(s * 4))
        x = np.arange(-half, half + 1)
        wavelet = np.exp(1j * 2 * np.pi * fr * x / fs) * np.exp(-x ** 2 / (2 * s ** 2))
        wavelet -= wavelet.mean()
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
        conv = np.convolve(sig, wavelet, mode="same")
        out[i] = np.abs(conv) ** 2
    return out
from sklearn.cluster import KMeans
import pandas as pd

ROOT = Path("/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project")
DER = ROOT / "analysis_calcium" / "derived"
OUT = ROOT / "analysis_calcium" / "plots"

fs = 2.882
DOM = 0.37151
NYQ = fs / 2
dff = np.load(DER / "traces_dff.npy")
gang = np.load(DER / "ganglion_signal.npy")
cent = pd.read_csv(DER / "roi_centroids.csv")
T = gang.shape[0]
popmean = dff.mean(0)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def psd(x):
    f, p = welch(x, fs=fs, nperseg=min(512, len(x)), noverlap=None)
    return f, p


fg, pg = psd(gang)
fp, pp = psd(popmean)
prom = pg[np.argmin(np.abs(fg - DOM))] / np.median(pg)

# ---------- Figure 1: PSD detailed ----------
fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
for a in ax.ravel():
    style(a)
ax[0, 0].semilogy(fg, pg, color="#1b5e20", lw=1.5)
for k, lab in [(1, "fundamental"), (2, "2nd harmonic"), (3, "3rd harmonic")]:
    ax[0, 0].axvline(DOM * k, color="crimson", ls="--", lw=1)
    ax[0, 0].text(DOM * k, pg.max(), f"{DOM*k:.2f}", rotation=90, va="top", fontsize=7, color="crimson")
ax[0, 0].axvline(NYQ, color="k", ls=":", lw=1); ax[0, 0].text(NYQ, pg.min() * 2, "Nyquist", rotation=90, fontsize=7)
ax[0, 0].set_title(f"Welch PSD ganglion (prominence {prom:.0f}x median)")
ax[0, 0].set_xlabel("frequency (Hz)"); ax[0, 0].set_ylabel("PSD (dF/F^2 / Hz)")
ax[0, 1].loglog(fg[1:], pg[1:], color="#1b5e20", lw=1.5)
ax[0, 1].axvline(DOM, color="crimson", ls="--")
ax[0, 1].set_title("PSD log-log"); ax[0, 1].set_xlabel("frequency (Hz)"); ax[0, 1].set_ylabel("PSD")
ax[1, 0].semilogy(fg, pg, color="#1b5e20", lw=1.4, label="ganglion")
ax[1, 0].semilogy(fp, pp, color="#bbbbbb", lw=1.2, label="patch-pop mean")
ax[1, 0].axvline(DOM, color="crimson", ls="--")
ax[1, 0].set_title("Ganglion vs patch-population PSD"); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_xlabel("frequency (Hz)"); ax[1, 0].set_ylabel("PSD")
domf = []
for i in range(dff.shape[0]):
    f, p = psd(dff[i])
    band = (f >= 0.1) & (f <= 1.4)
    domf.append(f[band][np.argmax(p[band])])
domf = np.array(domf)
ax[1, 1].hist(domf, bins=np.arange(0.1, 1.4, 0.03), color="#00695c", edgecolor="w")
ax[1, 1].axvline(DOM, color="crimson", ls="--", label=f"{DOM:.3f} Hz")
ax[1, 1].set_title(f"Per-patch dominant frequency ({100*np.mean(np.abs(domf-DOM)<0.05):.0f}% within 0.05 Hz)")
ax[1, 1].set_xlabel("Hz"); ax[1, 1].set_ylabel("patches"); ax[1, 1].legend(fontsize=8)
fig.savefig(OUT / "fig_spec_psd_detailed.png", dpi=170)
plt.close(fig)

# ---------- Figure 2: wavelet scalogram ----------
freqs = np.geomspace(0.1, 1.4, 80)
cwtm = morlet_cwt(gang, freqs, fs, w=6.0)
t = np.arange(T) / fs
fig = plt.figure(figsize=(14, 7), constrained_layout=True)
gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 4])
a0 = fig.add_subplot(gs[0]); style(a0)
a0.plot(t, gang, color="#1b5e20", lw=0.7); a0.set_xlim(0, t[-1])
a0.set_ylabel("dF/F"); a0.set_xticklabels([])
a0.set_title("Continuous wavelet scalogram (Morlet) of ganglion signal")
a1 = fig.add_subplot(gs[1])
im = a1.pcolormesh(t, freqs, cwtm, cmap="magma", shading="auto")
a1.set_yscale("log"); a1.axhline(DOM, color="cyan", ls="--", lw=1)
a1.set_ylabel("frequency (Hz)"); a1.set_xlabel("time (s)")
plt.colorbar(im, ax=a1, label="wavelet power")
fig.savefig(OUT / "fig_spec_wavelet.png", dpi=170)
plt.close(fig)

# ---------- Figure 3: spectrogram detailed ----------
f, tt, Sxx = spectrogram(gang, fs=fs, nperseg=128, noverlap=112)
fig = plt.figure(figsize=(14, 7), constrained_layout=True)
gs = GridSpec(2, 2, figure=fig, width_ratios=[4, 1], height_ratios=[4, 1])
a0 = fig.add_subplot(gs[0, 0])
im = a0.pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), cmap="magma", shading="auto")
a0.axhline(DOM, color="cyan", ls="--"); a0.set_ylim(0, 1.4)
a0.set_ylabel("frequency (Hz)"); a0.set_title("Spectrogram of ganglion signal")
a0.set_xticklabels([])
plt.colorbar(im, ax=a0, label="power (dB)")
a1 = fig.add_subplot(gs[0, 1]); style(a1)
a1.plot(Sxx.mean(1), f, color="#1b5e20"); a1.axhline(DOM, color="crimson", ls="--")
a1.set_ylim(0, 1.4); a1.set_xlabel("mean power"); a1.set_title("time-avg")
a2 = fig.add_subplot(gs[1, 0]); style(a2)
band = np.argmin(np.abs(f - DOM))
a2.plot(tt, Sxx[band], color="#4a148c"); a2.set_xlabel("time (s)")
a2.set_ylabel("power at\n0.37 Hz")
fig.savefig(OUT / "fig_spec_spectrogram_detailed.png", dpi=170)
plt.close(fig)

# ---------- Figure 4: coherence between regions ----------
nreg = 6
km = KMeans(n_clusters=nreg, n_init=10, random_state=0).fit(cent[["x", "y"]].values)
reg_sig = np.array([dff[km.labels_ == r].mean(0) for r in range(nreg)])
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
style(ax[0])
cohmat = np.eye(nreg)
for i in range(nreg):
    for j in range(i + 1, nreg):
        fc, cxy = coherence(reg_sig[i], reg_sig[j], fs=fs, nperseg=256)
        ax[0].plot(fc, cxy, lw=1, alpha=0.7, label=f"{i}-{j}")
        cohmat[i, j] = cohmat[j, i] = cxy[np.argmin(np.abs(fc - DOM))]
ax[0].axvline(DOM, color="crimson", ls="--"); ax[0].set_xlim(0, 1.4)
ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("coherence")
ax[0].set_title("Magnitude-squared coherence between regions")
ax[0].legend(fontsize=7, ncol=2)
im = ax[1].imshow(cohmat, cmap="viridis", vmin=0, vmax=1)
ax[1].set_title(f"Coherence at {DOM:.2f} Hz (mean {cohmat[np.triu_indices(nreg,1)].mean():.2f})")
ax[1].set_xlabel("region"); ax[1].set_ylabel("region")
plt.colorbar(im, ax=ax[1], label="coherence")
fig.savefig(OUT / "fig_spec_coherence.png", dpi=170)
plt.close(fig)

# ---------- Figure 5: per-patch PSD heatmap ----------
allp = []
for i in range(dff.shape[0]):
    f, p = psd(dff[i])
    allp.append(p)
allp = np.array(allp)
band = (f >= 0.1) & (f <= 1.4)
fb = f[band]
P = np.log10(allp[:, band] + 1e-9)
power_at_dom = allp[:, np.argmin(np.abs(f - DOM))]
order = np.argsort(power_at_dom)[::-1]
fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
im = ax.imshow(P[order], aspect="auto", cmap="magma",
               extent=[fb[0], fb[-1], dff.shape[0], 0], interpolation="nearest")
ax.axvline(DOM, color="cyan", ls="--", lw=1.2)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("patch (sorted by power at 0.37 Hz)")
ax.set_title("Per-patch PSD: shared 0.37 Hz stripe across the population")
plt.colorbar(im, ax=ax, label="log10 PSD")
fig.savefig(OUT / "fig_spec_patch_spectra.png", dpi=170)
plt.close(fig)

print("OK spec prom", round(prom), "patch%within0.05", round(100 * np.mean(np.abs(domf - DOM) < 0.05)),
      "mean coh", round(cohmat[np.triu_indices(nreg, 1)].mean(), 2))
