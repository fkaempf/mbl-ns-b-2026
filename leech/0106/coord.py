#!/usr/bin/env python3
"""Left-right (C3 vs C4) synchrony, accounting for C4 being the noisier
electrode (extra envelope smoothing). Steady windows, pre vs post DA, + file 4.
Also nails regularity (CV of C3 cycle period) in steady windows."""
import numpy as np, pyabf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from crawl_cpg import rate_envelope, detect_bursts, detect_spikes, cv

EVENT = 1200.0

def env_pair(fn, extra_smooth_c4=1.0):
    a = pyabf.ABF(fn); fs = a.dataRate
    a.setSweep(0, channel=0); C3 = a.sweepY
    a.setSweep(0, channel=1); C4 = a.sweepY
    eA, efs = rate_envelope(C3, fs)
    eB, _ = rate_envelope(C4, fs)
    eB = gaussian_filter1d(eB, extra_smooth_c4 * efs)   # tame C4 noise
    return eA, eB, efs, fs, C3

def corr_window(eA, eB, efs, t0, t1, maxlagS=10):
    s, e = int(t0 * efs), int(t1 * efs)
    a = eA[s:e] - eA[s:e].mean(); b = eB[s:e] - eB[s:e].mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan, np.nan
    xc = np.correlate(a, b, "full") / (a.std() * b.std() * len(a))
    mid = len(a) - 1; ml = int(maxlagS * efs)
    seg = xc[mid - ml: mid + ml + 1]
    k = np.argmax(np.abs(seg))
    return seg[k], (k - ml) / efs

eA, eB, efs, fs, C3 = env_pair("1.abf")
spk, _ = detect_spikes(C3, fs)
bursts, _ = detect_bursts(eA, efs, spk)
on = np.array([b["onset"] for b in bursts])

windows = [("pre-DA 600-1150", 600, 1150),
           ("post-peak 1400-2000", 1400, 2000),
           ("late 3000-3600", 3000, 3600)]
print("Left-right (C3 vs C4) synchrony + C3 rhythm regularity, file 1")
print(f"{'window':<22}{'L-R r':>8}{'lag(s)':>8}{'C3 periodCV':>13}{'n bursts':>10}")
for nm, t0, t1 in windows:
    r, lag = corr_window(eA, eB, efs, t0, t1)
    o = on[(on >= t0) & (on < t1)]; per = np.diff(o)
    print(f"{nm:<22}{r:>8.2f}{lag:>8.2f}{cv(per):>13.2f}{len(o):>10}")

# file 4 (sustained DA, hours later)
eA4, eB4, efs4, fs4, C34 = env_pair("4.abf")
r4, lag4 = corr_window(eA4, eB4, efs4, 100, 700)
print(f"{'file4 100-700 (late)':<22}{r4:>8.2f}{lag4:>8.2f}")

# figure: overlaid L/R envelopes in each window (z-scored)
fig, ax = plt.subplots(3, 1, figsize=(13, 7))
for axi, (nm, t0, t1) in zip(ax, windows):
    s, e = int(t0 * efs), int((t0 + 30) * efs)   # 30 s snapshot
    tt = np.arange(s, e) / efs
    za = (eA[s:e] - eA[s:e].mean()) / eA[s:e].std()
    zb = (eB[s:e] - eB[s:e].mean()) / eB[s:e].std()
    axi.plot(tt, za, label="C3 (clean)"); axi.plot(tt, zb, label="C4 (smoothed)", alpha=0.8)
    r, lag = corr_window(eA, eB, efs, t0, t1)
    axi.set_title(f"{nm}s   (L-R r={r:.2f}, lag={lag:.2f}s)"); axi.set_ylabel("z-env")
    axi.legend(fontsize=8, loc="upper right")
ax[-1].set_xlabel("time (s)")
fig.suptitle("Left (C3) vs right (C4) envelope, 30 s snapshots across the DA response")
fig.tight_layout(); fig.savefig("figures/10_left_right.png", dpi=130); plt.close(fig)
print("\n-> figures/10_left_right.png")
