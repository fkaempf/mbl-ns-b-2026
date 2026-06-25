#!/usr/bin/env python3
"""Build the one-slide summary from the workflow consensus.
C3 (clean left root) only. Hero IFR + regularity(CV) timecourse, burst-rate hump,
duty-cycle control, raw baseline-vs-peak insets, fold-change table."""
import numpy as np, pyabf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import gridspec
from crawl_cpg import rate_envelope, detect_bursts, detect_spikes

FILES = ["1.abf", "2.abf", "3.abf", "4.abf"]
WIN, EVENT = 300.0, 1200.0   # 5-min windows; DA at 20 min in file 1

onset, dur, nspk, ifr, edges_file, off = [], [], [], [], [], 0.0
raw_base = raw_peak = None
for fn in FILES:
    a = pyabf.ABF(fn); fs = a.dataRate
    a.setSweep(0, channel=0); C3 = a.sweepY
    spk, _ = detect_spikes(C3, fs)
    eA, efs = rate_envelope(C3, fs)
    bursts, _ = detect_bursts(eA, efs, spk)
    for b in bursts:
        onset.append(off + b["onset"]); dur.append(b["dur"])
        nspk.append(b["n"]); ifr.append(b["ifr"])
    if fn == "1.abf":
        snip = lambda t0: (np.arange(int(t0*fs), int((t0+8)*fs))/fs - t0,
                           C3[int(t0*fs):int((t0+8)*fs)])
        raw_base, raw_peak = snip(600), snip(1700)
    off += len(C3) / fs; edges_file.append(off)

onset = np.array(onset); dur = np.array(dur); nspk = np.array(nspk); ifr = np.array(ifr)
duty_onset = onset[:-1]; duty = dur[:-1] / np.diff(onset); duty[np.diff(onset) <= 0] = np.nan
T = off
edges = np.arange(0, T + WIN, WIN)
cen, IFR, RATE, DUTY, CVp = [], [], [], [], []
for t0, t1 in zip(edges[:-1], edges[1:]):
    m = (onset >= t0) & (onset < t1); o = onset[m]
    md = (duty_onset >= t0) & (duty_onset < t1)
    cen.append((t0 + t1) / 2 / 60)
    RATE.append(m.sum() / ((t1 - t0) / 60))
    IFR.append(np.median(ifr[m]) if m.sum() else np.nan)
    DUTY.append(np.nanmedian(duty[md]) if md.sum() else np.nan)
    iv = np.diff(o)
    CVp.append(np.std(iv) / np.mean(iv) if len(iv) > 2 and np.mean(iv) > 0 else np.nan)
cen = np.array(cen); IFR = np.array(IFR); CVp = np.array(CVp)
DA = EVENT / 60
base_ifr = np.nanmedian(IFR[cen < 20])

# ============================== FIGURE ==============================
plt.rcParams.update({"font.size": 11})
fig = plt.figure(figsize=(16, 9), facecolor="w")
fig.suptitle("Dopamine retunes the leech crawl rhythm — faster, more regular, more intense —\n"
             "then it runs down over hours with DA still in the bath  "
             "(single isolated ganglion, clean left root C3)",
             fontsize=15, fontweight="bold", y=0.985)
gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1],
                       hspace=0.45, wspace=0.32,
                       left=0.06, right=0.945, top=0.88, bottom=0.07)

# ---- HERO: IFR (left axis) + regularity CV (right axis) ----
axh = fig.add_subplot(gs[0:2, 0:2])
axh.plot(cen, IFR, "-o", ms=4, lw=1.8, color="C0", label="intraburst IFR (intensity)")
axh.axhline(base_ifr, color="C0", ls=":", lw=1.2, alpha=0.7)
axh.text(2, base_ifr + 3, "pre-DA baseline", color="C0", fontsize=9)
axc = axh.twinx()
axc.plot(cen, CVp, "-s", ms=3, lw=1.3, color="C3", alpha=0.8,
         label="period CV (regularity)")
axc.set_ylabel("period CV  (lower = more regular)", color="C3")
axc.tick_params(axis="y", colors="C3")
axh.axvline(DA, color="k", ls="--", lw=1.6)
for e in edges_file[:-1]:
    axh.axvline(e / 60, color="0.75", ls=":", lw=0.8)
ytop = np.nanmax(IFR) * 1.12
axh.set_ylim(0, ytop)
axh.annotate("", xy=(T/60, ytop*0.93), xytext=(DA, ytop*0.93),
             arrowprops=dict(arrowstyle="-", color="seagreen", lw=4))
axh.text((DA + T/60)/2, ytop*0.95, "dopamine in bath  (never washed out)",
         ha="center", va="bottom", color="seagreen", fontsize=10, fontweight="bold")
axh.annotate("DA on", xy=(DA, ytop*0.5), xytext=(DA+8, ytop*0.62),
             color="k", fontsize=10, arrowprops=dict(arrowstyle="->"))
axh.annotate("rundown — DA still present", xy=(160, IFR[np.argmin(np.abs(cen-160))]),
             xytext=(120, ytop*0.78), color="dimgray", fontsize=10,
             arrowprops=dict(arrowstyle="->", color="dimgray"))
axh.set_xlabel("experiment time (min)"); axh.set_ylabel("intraburst IFR (Hz)", color="C0")
axh.tick_params(axis="y", colors="C0")
axh.set_title("C3 rhythm: activation then rundown (5-min windows, all 4 files)", fontsize=12)
# combined legend
l1, lab1 = axh.get_legend_handles_labels(); l2, lab2 = axc.get_legend_handles_labels()
axh.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=9, framealpha=0.9)

# ---- raw insets ----
for ax, (tt, yy), ttl, c in [(fig.add_subplot(gs[0, 2]), raw_base, "baseline (~10 min): slow, sparse", "0.3"),
                             (fig.add_subplot(gs[1, 2]), raw_peak, "DA peak (~28 min): fast, dense", "C0")]:
    ax.plot(tt, yy, lw=0.4, color=c)
    ax.set_title(ttl, fontsize=10); ax.set_yticks([]); ax.set_xlabel("s", fontsize=8)
    ax.set_xlim(0, 8)

# ---- burst-rate hump ----
axr = fig.add_subplot(gs[2, 0])
axr.plot(cen, RATE, "-o", ms=3, color="C2"); axr.axvline(DA, color="k", ls="--", lw=1.2)
for e in edges_file[:-1]: axr.axvline(e/60, color="0.8", ls=":", lw=0.7)
axr.set_title("burst rate — builds then falls", fontsize=11)
axr.set_xlabel("time (min)"); axr.set_ylabel("bursts/min")

# ---- duty cycle control ----
axd = fig.add_subplot(gs[2, 1])
axd.plot(cen, DUTY, "-o", ms=3, color="C4"); axd.axhline(0.5, color="0.5", ls="--", lw=1)
axd.axvline(DA, color="k", ls="--", lw=1.2); axd.set_ylim(0, 1)
for e in edges_file[:-1]: axd.axvline(e/60, color="0.8", ls=":", lw=0.7)
axd.set_title("duty cycle ~0.5 — no systematic change\n(specificity control)", fontsize=11)
axd.set_xlabel("time (min)"); axd.set_ylabel("burst / cycle")

# ---- numbers table ----
axt = fig.add_subplot(gs[2, 2]); axt.axis("off")
rows = [["metric", "baseline", "DA peak", "late"],
        ["period (s)", "5.4", "2.3", "~6"],
        ["IFR (Hz)", "55", "120", "13"],
        ["spikes/burst", "90", "several×", "collapsed"],
        ["period CV", "0.83", "0.33", "—"],
        ["duty cycle", "~0.5", "~0.5", "~0.5"]]
tb = axt.table(cellText=rows, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.5)
for j in range(4): tb[0, j].set_facecolor("0.85"); tb[0, j].set_text_props(fontweight="bold")
for j in range(4): tb[4, j].set_text_props(fontweight="bold", color="C3")  # CV row
axt.set_title("fold-changes (direction, not absolute)", fontsize=11)

fig.text(0.06, 0.012,
         "n=1 isolated ganglion, single DA application, no washout/vehicle control — case study, not a population result.  "
         "C4 (noisy right root) and left-right coupling excluded.  Count metrics are detection-threshold sensitive; trust direction + dimensionless CV.",
         fontsize=8, color="0.4")
fig.savefig("figures/SLIDE.png", dpi=150, facecolor="w")
print("baseline IFR:", round(base_ifr,1), "-> figures/SLIDE.png")
