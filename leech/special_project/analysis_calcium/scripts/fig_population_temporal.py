#!/usr/bin/env python
"""Intricate temporal / population-dynamics figures for the leech ganglion GCaMP6m
recording (helobdella_LeechNo2_elavgcamp6m-17). Reuses derived arrays. No em dashes."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks, hilbert, butter, filtfilt, welch
from scipy.cluster.hierarchy import linkage, leaves_list

ROOT = Path("/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project")
DER = ROOT / "analysis_calcium" / "derived"
OUT = ROOT / "analysis_calcium" / "plots"

fs = 2.882
DOM = 0.37151
PERIOD = 1.0 / DOM
dff = np.load(DER / "traces_dff.npy")          # (180, T)
gang = np.load(DER / "ganglion_signal.npy")    # (T,)
T = gang.shape[0]
t = np.arange(T) / fs
popmean = dff.mean(0)

# cycle peaks on the ganglion signal
mind = int(0.7 * PERIOD * fs)
pk, _ = find_peaks(gang, distance=mind, prominence=np.std(gang) * 0.4)
ipi = np.diff(pk) / fs


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---------- Figure 1: population raster ----------
z = (dff - dff.mean(1, keepdims=True)) / (dff.std(1, keepdims=True) + 1e-9)
L = linkage(z, method="ward", metric="euclidean")
order = leaves_list(L)
zz = z[order]
amp = dff.std(1)[order]

fig = plt.figure(figsize=(15, 9), constrained_layout=True)
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 5], width_ratios=[40, 1.5])
axg = fig.add_subplot(gs[0, 0]); style(axg)
axg.plot(t, gang, color="#1b5e20", lw=0.8)
axg.plot(pk / fs, gang[pk], "v", color="crimson", ms=4)
axg.set_ylabel("ganglion\ndF/F"); axg.set_xlim(0, t[-1])
axg.set_title("Population raster: 180 active patches, ward-clustered, with ganglion rhythm and cycle peaks")
axg.set_xticklabels([])
axr = fig.add_subplot(gs[1, 0])
im = axr.imshow(zz, aspect="auto", cmap="magma", vmin=-2, vmax=4,
                extent=[0, t[-1], len(zz), 0], interpolation="nearest")
axr.set_xlabel("time (s)"); axr.set_ylabel("patch (clustered order)")
axcb = fig.add_subplot(gs[1, 1]); plt.colorbar(im, cax=axcb, label="dF/F (z-scored)")
axa = fig.add_subplot(gs[0, 1]); axa.axis("off")
fig.savefig(OUT / "fig_pop_raster.png", dpi=170)
plt.close(fig)

# ---------- Figure 2: cycle average ----------
half = int(round(0.5 * PERIOD * fs))
win = np.arange(-half, half + 1)
tau = win / fs


def stack_cycles(sig):
    segs = [sig[p + win] for p in pk if p + win[0] >= 0 and p + win[-1] < T]
    return np.array(segs)


sg = stack_cycles(gang)
sp = stack_cycles(popmean)
fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
for a in ax.ravel():
    style(a)
for seg in sg:
    ax[0, 0].plot(tau, seg, color="grey", alpha=0.15, lw=0.5)
ax[0, 0].plot(tau, sg.mean(0), color="#1b5e20", lw=2.5)
ax[0, 0].fill_between(tau, sg.mean(0) - sg.std(0), sg.mean(0) + sg.std(0),
                      color="#1b5e20", alpha=0.25)
ax[0, 0].axvline(0, color="crimson", ls=":")
ax[0, 0].set_title(f"Ganglion cycle average (n={len(sg)} cycles)")
ax[0, 0].set_xlabel("time from peak (s)"); ax[0, 0].set_ylabel("dF/F")
for seg in sp:
    ax[0, 1].plot(tau, seg, color="grey", alpha=0.15, lw=0.5)
ax[0, 1].plot(tau, sp.mean(0), color="#4a148c", lw=2.5)
ax[0, 1].fill_between(tau, sp.mean(0) - sp.std(0), sp.mean(0) + sp.std(0),
                      color="#4a148c", alpha=0.25)
ax[0, 1].axvline(0, color="crimson", ls=":")
ax[0, 1].set_title("Patch-population cycle average")
ax[0, 1].set_xlabel("time from peak (s)"); ax[0, 1].set_ylabel("dF/F")
ax[1, 0].hist(ipi, bins=20, color="#00695c", edgecolor="w")
ax[1, 0].axvline(np.median(ipi), color="crimson", ls="--",
                 label=f"median {np.median(ipi):.2f} s")
ax[1, 0].set_title(f"Inter-peak intervals (CV {np.std(ipi)/np.mean(ipi):.3f})")
ax[1, 0].set_xlabel("interval (s)"); ax[1, 0].set_ylabel("count"); ax[1, 0].legend()
axp = fig.add_subplot(2, 2, 4, projection="polar")
ph = (pk / fs % PERIOD) / PERIOD * 2 * np.pi
axp.hist(ph, bins=24, color="#1b5e20", alpha=0.8)
axp.set_title("Cycle-peak phase distribution")
ax[1, 1].axis("off")
fig.savefig(OUT / "fig_pop_cycle_average.png", dpi=170)
plt.close(fig)

# ---------- Figure 3: instantaneous (Hilbert) ----------
b, a = butter(2, [0.2 / (fs / 2), 0.6 / (fs / 2)], btype="band")
filt = filtfilt(b, a, gang)
an = hilbert(filt)
env = np.abs(an)
ph_t = np.unwrap(np.angle(an))
ifreq = np.diff(ph_t) / (2 * np.pi) * fs
fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for a_ in ax.ravel():
    style(a_)
ax[0, 0].plot(t, filt, color="grey", lw=0.6, label="bandpassed")
ax[0, 0].plot(t, env, color="crimson", lw=1.3, label="envelope")
ax[0, 0].plot(t, -env, color="crimson", lw=1.3)
ax[0, 0].set_title("Ganglion signal with amplitude envelope")
ax[0, 0].set_xlabel("time (s)"); ax[0, 0].set_ylabel("dF/F (band 0.2-0.6 Hz)")
ax[0, 0].legend(fontsize=8)
ax[0, 1].plot(t[1:], ifreq, color="#1b5e20", lw=0.8)
ax[0, 1].axhline(DOM, color="crimson", ls="--", label=f"{DOM:.3f} Hz")
ax[0, 1].set_ylim(0, 0.8); ax[0, 1].set_title("Instantaneous frequency")
ax[0, 1].set_xlabel("time (s)"); ax[0, 1].set_ylabel("Hz"); ax[0, 1].legend(fontsize=8)
ax[1, 0].plot(t, env, color="#4a148c", lw=0.9)
ax[1, 0].set_title("Instantaneous amplitude")
ax[1, 0].set_xlabel("time (s)"); ax[1, 0].set_ylabel("envelope")
good = (t[1:] > 5) & (t[1:] < t[-1] - 5)
ax[1, 1].hist(ifreq[good], bins=40, color="#00695c", edgecolor="w")
ax[1, 1].axvline(DOM, color="crimson", ls="--")
ax[1, 1].set_title("Instantaneous-frequency distribution")
ax[1, 1].set_xlabel("Hz"); ax[1, 1].set_ylabel("count"); ax[1, 1].set_xlim(0, 0.8)
fig.savefig(OUT / "fig_pop_instantaneous.png", dpi=170)
plt.close(fig)

# ---------- Figure 4: interval dynamics ----------
fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for a_ in ax.ravel():
    style(a_)
ax[0, 0].plot(np.arange(len(ipi)), ipi, "o-", color="#1b5e20", ms=3, lw=0.8)
ax[0, 0].axhline(np.median(ipi), color="crimson", ls="--")
ax[0, 0].set_title("Inter-peak interval vs cycle")
ax[0, 0].set_xlabel("cycle number"); ax[0, 0].set_ylabel("interval (s)")
x1, y1 = ipi[:-1], ipi[1:]
ax[0, 1].scatter(x1, y1, s=18, color="#4a148c", alpha=0.7)
lim = [ipi.min() * 0.9, ipi.max() * 1.1]
ax[0, 1].plot(lim, lim, color="grey", ls=":")
sd1 = np.std(y1 - x1) / np.sqrt(2); sd2 = np.std(y1 + x1) / np.sqrt(2)
from matplotlib.patches import Ellipse
cx, cy = np.mean(x1), np.mean(y1)
ax[0, 1].add_patch(Ellipse((cx, cy), 2 * sd2, 2 * sd1, angle=45,
                   fill=False, color="crimson", lw=1.5))
ax[0, 1].set_title(f"Poincare map (SD1 {sd1:.3f}, SD2 {sd2:.3f} s)")
ax[0, 1].set_xlabel("IPI(n) s"); ax[0, 1].set_ylabel("IPI(n+1) s")
ax[0, 1].set_xlim(lim); ax[0, 1].set_ylim(lim)
ax[1, 0].hist(ipi, bins=20, color="#00695c", edgecolor="w")
ax[1, 0].axvline(np.median(ipi), color="crimson", ls="--",
                 label=f"median {np.median(ipi):.2f} s, CV {np.std(ipi)/np.mean(ipi):.3f}")
ax[1, 0].set_title("IPI histogram"); ax[1, 0].set_xlabel("interval (s)")
ax[1, 0].set_ylabel("count"); ax[1, 0].legend(fontsize=8)
k = 5
run = np.convolve(ipi, np.ones(k) / k, mode="valid")
ax[1, 1].plot(np.arange(len(run)), run, color="#1b5e20", lw=1.5)
ax[1, 1].set_title(f"Running mean IPI ({k}-cycle)")
ax[1, 1].set_xlabel("cycle number"); ax[1, 1].set_ylabel("interval (s)")
fig.savefig(OUT / "fig_pop_interval_dynamics.png", dpi=170)
plt.close(fig)

# ---------- Figure 5: patch small-multiples ----------
amp_all = dff.std(1)
top = np.argsort(amp_all)[::-1][:36]
fig, axes = plt.subplots(6, 6, figsize=(16, 11), sharex=True, sharey=True,
                         constrained_layout=True)
ymax = np.percentile(dff[top], 99)
for ax_, ci in zip(axes.ravel(), top):
    ax_.plot(t, dff[ci], color="#1b5e20", lw=0.5)
    ax_.set_title(f"patch {ci}", fontsize=7)
    ax_.set_ylim(-0.1, ymax)
    style(ax_)
for ax_ in axes[-1]:
    ax_.set_xlabel("s", fontsize=7)
fig.suptitle("36 most active patches: dF/F (shared scale), uniform 0.37 Hz rhythm", fontsize=13)
fig.savefig(OUT / "fig_pop_patch_overview.png", dpi=170)
plt.close(fig)

print("OK pop", len(pk), "cycles, IPI CV", round(np.std(ipi) / np.mean(ipi), 3),
      "n clusters order", len(order))
