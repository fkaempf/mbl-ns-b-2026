#!/usr/bin/env python3
"""Deeper, burst-based analysis to find the story.
- File 1: dopamine (t=1200s) transient -> latency, peak, decay tau
- Steady-window comparison (avoid the transient)
- C4 response (it's tonic, so use envelope level + its own bursts)
- C3<->C4 coordination in sliding windows, pre vs post DA and across files
- Trajectory of burst metrics across all 4 files (washout / rundown?)
"""
import numpy as np, pandas as pd, pyabf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from crawl_cpg import rate_envelope, detect_bursts, detect_spikes, cv, rcv

FILES = ["1.abf", "2.abf", "3.abf", "4.abf"]
EVENT = 1200.0  # DA added in file 1

def burst_table(C3, fs):
    spk, _ = detect_spikes(C3, fs)
    env, efs = rate_envelope(C3, fs)
    bursts, _ = detect_bursts(env, efs, spk)
    on = np.array([b["onset"] for b in bursts])
    df = pd.DataFrame(dict(onset=on, dur=[b["dur"] for b in bursts],
                           n_spikes=[b["n"] for b in bursts],
                           ifr=[b["ifr"] for b in bursts]))
    df["period"] = np.r_[np.nan, np.diff(df.onset)]
    return df, env, efs, len(spk)

def windowed_xcorr(eA, eB, efs, winS=120, stepS=60, maxlagS=15):
    """median peak-lag and peak-corr of detrended envelope cross-corr."""
    w = int(winS*efs); st = int(stepS*efs); ml = int(maxlagS*efs)
    lags_pk, corr_pk = [], []
    for s in range(0, len(eA)-w, st):
        a = eA[s:s+w]-eA[s:s+w].mean(); b = eB[s:s+w]-eB[s:s+w].mean()
        if a.std()<1e-9 or b.std()<1e-9: continue
        xc = np.correlate(a, b, "full")/(a.std()*b.std()*len(a))
        mid = len(a)-1
        seg = xc[mid-ml:mid+ml+1]
        k = np.argmax(seg); lags_pk.append((k-ml)/efs); corr_pk.append(seg[k])
    return np.array(lags_pk), np.array(corr_pk)

# ---------- per-file summary ----------
print(f"{'file':<8}{'C3 brate/min':>13}{'period':>8}{'dur':>7}{'spk/brst':>9}"
      f"{'IFR':>7}{'xcorr lag':>11}{'xcorr r':>9}")
rows = []
file1 = None
for fn in FILES:
    a = pyabf.ABF(fn); fs = a.dataRate; T = a.sweepCount and a.sweepX[-1]
    a.setSweep(0, channel=0); C3 = a.sweepY
    a.setSweep(0, channel=1); C4 = a.sweepY
    df, eA, efs, nspk = burst_table(C3, fs)
    eB, _ = rate_envelope(C4, fs)
    lag, r = windowed_xcorr(eA, eB, efs)
    dur_total = len(C3)/fs
    rows.append(dict(file=fn, brate=len(df)/dur_total*60,
                     period=df.period.median(), dur=df.dur.median(),
                     spk=df.n_spikes.median(), ifr=df.ifr.median(),
                     lag=np.median(lag), r=np.median(r),
                     dfp=df))
    print(f"{fn:<8}{len(df)/dur_total*60:>13.1f}{df.period.median():>8.2f}"
          f"{df.dur.median():>7.2f}{df.n_spikes.median():>9.0f}{df.ifr.median():>7.0f}"
          f"{np.median(lag):>11.2f}{np.median(r):>9.2f}")
    if fn == "1.abf":
        file1 = (df, eA, eB, efs)

# ---------- file 1: dopamine transient ----------
df1, eA1, eB1, efs1 = file1
post = df1[df1.onset >= EVENT].copy()
trend = post.set_index("onset").ifr.rolling(20, center=True).median().dropna()
tpk = trend.idxmax(); ypk = trend.max()
pre_base = df1[df1.onset < EVENT].ifr.median()
print(f"\n--- file 1 dopamine transient (intraburst IFR) ---")
print(f"baseline IFR pre-DA: {pre_base:.0f} Hz")
print(f"peak {ypk:.0f} Hz at t={tpk:.0f}s  (latency {tpk-EVENT:.0f}s after DA)")
# exp decay fit from peak to end
dec = trend[trend.index >= tpk]
def expf(t, A, tau, C): return A*np.exp(-(t-tpk)/tau)+C
try:
    p,_ = curve_fit(expf, dec.index, dec.values, p0=[ypk-pre_base, 600, pre_base],
                    maxfev=10000, bounds=([0,30,0],[1e4,1e4,1e3]))
    print(f"decay tau ~ {p[1]:.0f}s ({p[1]/60:.1f} min), settling toward {p[2]:.0f} Hz")
except Exception as e:
    print("decay fit failed:", e); p=None

# ---------- steady-window comparison (avoid transient) ----------
def win(df, t0, t1): return df[(df.onset>=t0)&(df.onset<t1)]
wins = [("pre-DA 600-1150s", win(df1,600,1150)),
        ("post peak 1400-2000s", win(df1,1400,2000)),
        ("late 3000-3600s", win(df1,3000,3600))]
print(f"\n--- file 1 steady windows ---")
print(f"{'window':<22}{'n':>5}{'brate/min':>10}{'period':>8}{'CVper':>7}{'spk/b':>7}{'IFR':>6}")
for nm,w in wins:
    dur=(w.onset.max()-w.onset.min()) if len(w)>1 else 1
    print(f"{nm:<22}{len(w):>5}{len(w)/dur*60:>10.1f}{w.period.median():>8.2f}"
          f"{cv(w.period):>7.2f}{w.n_spikes.median():>7.0f}{w.ifr.median():>6.0f}")

# ---------- coordination pre vs post DA (file 1) ----------
i_ev = int(EVENT*efs1)
lpre,rpre = windowed_xcorr(eA1[:i_ev], eB1[:i_ev], efs1)
lpost,rpost = windowed_xcorr(eA1[i_ev:], eB1[i_ev:], efs1)
print(f"\n--- file 1 C3<->C4 coordination (envelope xcorr) ---")
print(f"pre-DA : peak r {np.median(rpre):.2f}, lag {np.median(lpre):.2f}s")
print(f"post-DA: peak r {np.median(rpost):.2f}, lag {np.median(lpost):.2f}s")

# ============================ FIGURES ============================
# A: across-file trajectory
sdf = pd.DataFrame(rows)
fig, ax = plt.subplots(2,3, figsize=(13,6))
labels = [("brate","C3 burst rate /min"),("period","cycle period (s)"),
          ("ifr","intraburst IFR (Hz)"),("spk","spikes/burst"),
          ("r","C3-C4 xcorr peak r"),("lag","C3-C4 lag (s)")]
for axi,(c,l) in zip(ax.flat, labels):
    axi.plot(range(1,5), sdf[c], "o-"); axi.set_title(l); axi.set_xticks(range(1,5))
    axi.set_xlabel("file #")
fig.suptitle("Trajectory across the 4 files (file 1 contains DA at 20 min)")
fig.tight_layout(); fig.savefig("figures/8_across_files.png", dpi=130); plt.close(fig)

# B: file-1 transient with fit
fig, ax = plt.subplots(figsize=(12,4))
ax.scatter(df1.onset, df1.ifr, s=8, c=np.where(df1.onset<EVENT,"C1","C0"))
ax.plot(trend.index, trend.values, "k", lw=1.2, label="rolling median")
if p is not None:
    tt=np.linspace(tpk, dec.index.max(),200); ax.plot(tt, expf(tt,*p),"r",lw=2,
        label=f"decay tau={p[1]/60:.1f}min")
ax.axvline(EVENT,color="C3",ls="--",label="DA"); ax.axvline(tpk,color="0.5",ls=":")
ax.set_xlabel("time (s)"); ax.set_ylabel("intraburst IFR (Hz)")
ax.set_title("1.abf: dopamine drives a transient rise then decay in C3 burst intensity")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/9_da_transient_fit.png", dpi=130); plt.close(fig)
print("\n-> figures/8_across_files.png, 9_da_transient_fit.png")
