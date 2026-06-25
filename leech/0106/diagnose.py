#!/usr/bin/env python3
"""Quick diagnostic: look at one file at several timescales + rate envelope,
to find the crawl rhythm timescale and set burst detection sensibly."""
import os, numpy as np, pyabf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

a = pyabf.ABF("1.abf")
fs = a.dataRate
a.setSweep(0, channel=0); A = a.sweepY
a.setSweep(0, channel=1); B = a.sweepY
T = len(A) / fs
print(f"1.abf: {T:.0f} s @ {fs} Hz, {len(A)} samples/ch")

def envelope(x, smooth_s):
    """rectified, smoothed firing-rate envelope (downsampled to 100 Hz)."""
    step = int(fs / 100)
    r = np.abs(x)[::step]
    return gaussian_filter1d(r, smooth_s * 100)

fig, ax = plt.subplots(4, 1, figsize=(14, 9))
# (1) raw 5 s
w = slice(0, int(5 * fs))
tt = np.arange(w.stop) / fs
ax[0].plot(tt, A[w], lw=0.3, label="C3"); ax[0].plot(tt, B[w] - 1, lw=0.3, label="C4 (offset)")
ax[0].set_title("Raw, first 5 s"); ax[0].set_xlabel("s"); ax[0].legend(fontsize=8)
# (2) raw 60 s
w = slice(0, int(60 * fs))
tt = np.arange(w.stop) / fs
ax[1].plot(tt, A[w], lw=0.2); ax[1].plot(tt, B[w] - 1, lw=0.2)
ax[1].set_title("Raw, first 60 s"); ax[1].set_xlabel("s")
# (3) rate envelope, first 300 s, a few smoothings
te = np.arange(0, T, 0.01)[:30000]
for sm in (0.2, 0.5, 1.0):
    eA = envelope(A, sm)[:30000]
    ax[2].plot(te[:len(eA)], eA, lw=0.6, label=f"C3 smooth={sm}s")
ax[2].set_title("C3 rate envelope, first 300 s (look for rhythmic peaks)")
ax[2].set_xlabel("s"); ax[2].legend(fontsize=8)
# (4) both envelopes, first 300 s
eA = envelope(A, 0.5)[:30000]; eB = envelope(B, 0.5)[:30000]
ax[3].plot(te[:len(eA)], eA, lw=0.6, label="C3")
ax[3].plot(te[:len(eB)], eB, lw=0.6, label="C4")
ax[3].set_title("C3 vs C4 envelopes, first 300 s (coordination?)")
ax[3].set_xlabel("s"); ax[3].legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/diagnose.png", dpi=120); plt.close(fig)
print("-> figures/diagnose.png")
