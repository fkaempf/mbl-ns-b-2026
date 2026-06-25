#!/usr/bin/env python3
"""Figure 8 redone at 5-minute resolution: the six burst metrics computed in
5-min windows across the whole concatenated experiment (all 4 files), instead
of one point per file. DA bath-applied at 20 min into file 1."""
import numpy as np, pyabf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from crawl_cpg import rate_envelope, detect_bursts, detect_spikes

FILES = ["1.abf", "2.abf", "3.abf", "4.abf"]
WIN = 300.0           # 5-min windows
EVENT = 1200.0        # DA at 20 min into file 1

# ---- concatenate: global burst table + envelopes ----
onset, dur, nspk, ifr = [], [], [], []
eA_all, eB_all = [], []
edges_file, off, efs = [], 0.0, None
for fn in FILES:
    a = pyabf.ABF(fn); fs = a.dataRate
    a.setSweep(0, channel=0); C3 = a.sweepY
    a.setSweep(0, channel=1); C4 = a.sweepY
    spk, _ = detect_spikes(C3, fs)
    eA, efs = rate_envelope(C3, fs)
    eB, _ = rate_envelope(C4, fs)
    bursts, _ = detect_bursts(eA, efs, spk)
    for b in bursts:
        onset.append(off + b["onset"]); dur.append(b["dur"])
        nspk.append(b["n"]); ifr.append(b["ifr"])
    eA_all.append(eA); eB_all.append(eB)
    off += len(C3) / fs
    edges_file.append(off)

onset = np.array(onset); dur = np.array(dur)
nspk = np.array(nspk); ifr = np.array(ifr)
# per-burst duty cycle = burst duration / cycle period (interval to next onset)
duty_onset = onset[:-1]
duty = dur[:-1] / np.diff(onset)
duty[np.diff(onset) <= 0] = np.nan      # guard file-boundary joins
eA_all = np.concatenate(eA_all); eB_all = np.concatenate(eB_all)
T = off

def xcorr_win(t0, t1, maxlagS=10):
    s, e = int(t0 * efs), int(t1 * efs)
    a = eA_all[s:e] - np.mean(eA_all[s:e]); b = eB_all[s:e] - np.mean(eB_all[s:e])
    if len(a) < 10 or a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan, np.nan
    xc = np.correlate(a, b, "full") / (a.std() * b.std() * len(a))
    mid = len(a) - 1; ml = int(maxlagS * efs)
    seg = xc[mid - ml: mid + ml + 1]
    k = int(np.argmax(seg))
    return seg[k], (k - ml) / efs

# ---- per-window metrics ----
edges = np.arange(0, T + WIN, WIN)
centers, brate, period, dutyW, IFR, spb, R, LAG = [], [], [], [], [], [], [], []
for t0, t1 in zip(edges[:-1], edges[1:]):
    m = (onset >= t0) & (onset < t1)
    o = onset[m]
    md = (duty_onset >= t0) & (duty_onset < t1)
    centers.append((t0 + t1) / 2 / 60)                 # minutes
    brate.append(m.sum() / ((t1 - t0) / 60))           # bursts/min
    period.append(np.median(np.diff(o)) if m.sum() >= 2 else np.nan)
    dutyW.append(np.nanmedian(duty[md]) if md.sum() else np.nan)
    IFR.append(np.median(ifr[m]) if m.sum() else np.nan)
    spb.append(np.median(nspk[m]) if m.sum() else np.nan)
    r, lag = xcorr_win(t0, t1)
    R.append(r); LAG.append(lag)

centers = np.array(centers)
series = [("C3 burst rate /min", brate), ("cycle period (s)", period),
          ("duty cycle", dutyW), ("intraburst IFR (Hz)", IFR),
          ("spikes/burst", spb), ("C3-C4 xcorr peak r", R),
          ("C3-C4 lag (s)", LAG)]

# ---- plot ----
fig, ax = plt.subplots(2, 4, figsize=(16, 6.5))
for axi, (lab, y) in zip(ax.flat, series):
    axi.plot(centers, y, "-o", ms=3, lw=1)
    axi.axvline(EVENT / 60, color="C3", ls="--", lw=1.3)
    for e in edges_file[:-1]:
        axi.axvline(e / 60, color="0.6", ls=":", lw=0.8)
    axi.set_title(lab); axi.set_xlabel("experiment time (min)")
for axi in ax.flat[len(series):]:
    axi.axis("off")                       # hide unused panels
ax.flat[0].axvline(EVENT / 60, color="C3", ls="--", lw=1.3, label="DA")
ax.flat[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Burst metrics in 5-min windows across all 4 files (DA at 20 min; "
             "dotted = file boundaries)")
fig.tight_layout()
fig.savefig("figures/8b_across_5min.png", dpi=130); plt.close(fig)
print(f"{len(centers)} windows over {T/60:.0f} min -> figures/8b_across_5min.png")
