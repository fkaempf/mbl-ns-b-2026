#!/usr/bin/env python3
"""
crawl_cpg.py - Analyze two-channel extracellular nerve recordings of the leech
crawl CPG.

Core questions this answers
---------------------------
1. Concatenate multiple ABF files (1,2,3,4...) into ONE continuous experiment
   timeline, ordered by true recording time (from the ABF header), the same way
   epsp_auc_timeline.m does it.
2. Detect spikes -> bursts on each of the two channels.
3. Per-burst metrics: duration, spikes/burst, intraburst firing rate (IFR).
4. Co-activity hypothesis: are burst metrics (duration, spikes/burst, IFR) MORE
   STABLE (lower coefficient of variation) when BOTH channels are bursting
   together than when only one channel ("solo") is active?
5. How do these metrics + IFR drift across the whole experiment?

Usage
-----
    # See example plots now on synthetic crawl-like data:
    python crawl_cpg.py --demo

    # First: see what channels a file has and pick the 2 nerve channels:
    python crawl_cpg.py --info          # lists channels + plots each one

    # Run on your real recordings (drop the .abf files in this folder):
    python crawl_cpg.py                 # uses every *.abf here, time-ordered
    python crawl_cpg.py file1.abf file2.abf ...
    python crawl_cpg.py --ch 0,2        # use channels 0 and 2 as the two nerves

Tunables are CONFIG at the top. Use the QC/overview figure to set thresholds.
Figures + a per-burst CSV are written to ./figures/.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------------
# CONFIG  (tune these to your prep; crawl cycle ~ several seconds)
# ----------------------------------------------------------------------------
CFG = dict(
    spike_thresh_k   = 4.5,    # spike threshold = k * MAD-noise estimate
    refractory_ms    = 2.0,    # min spacing between spikes (ms)
    # --- envelope-based burst detection (right tool for dense nerve activity) ---
    env_ds_hz        = 200.0,  # downsample rate for the rectified envelope
    env_smooth_s     = 0.40,   # Gaussian smoothing of the rectified envelope (s)
    burst_thr_pctile = 60.0,   # burst starts above this percentile of the envelope
    burst_off_pctile = 45.0,   # ...and ends when it drops below this (hysteresis)
    burst_merge_s    = 0.40,   # merge bursts separated by less than this (s)
    burst_min_dur_s  = 0.80,   # drop bursts shorter than this (s)
    burst_max_dur_s  = 30.0,   # flag/clip implausibly long merged bursts (s)
    # --- co-activity + IFR ---
    coactive_overlap = 0.25,   # C4 burst is "co-active" with a C3 cycle if onsets are
                               # within this fraction of a cycle (or they overlap)
    ifr_bin_ms       = 100.0,  # bin size for the continuous population-rate (IFR) trace
    ifr_smooth_ms    = 500.0,  # Gaussian smoothing of the IFR trace
)


# ----------------------------------------------------------------------------
# Loading + concatenation
# ----------------------------------------------------------------------------
def inspect(path, outdir):
    """List every channel and plot a 20 s snippet of each, so you can identify
    which two channels are the spiky extracellular nerves."""
    import pyabf
    os.makedirs(outdir, exist_ok=True)
    a = pyabf.ABF(path)
    print(f"\n{os.path.basename(path)}  recorded {a.abfDateTime}")
    print(f"  channels : {a.channelCount}")
    print(f"  rate     : {a.dataRate} Hz")
    print(f"  sweeps   : {a.sweepCount},  {a.sweepPointCount} samples/sweep")
    for i in range(a.channelCount):
        print(f"   [{i}] {a.adcNames[i]:<12} ({a.adcUnits[i]})")
    n = a.channelCount
    fig, ax = plt.subplots(n, 1, figsize=(13, 1.6 * n), sharex=True)
    ax = np.atleast_1d(ax)
    for i in range(n):
        a.setSweep(0, channel=i)
        w = a.sweepX <= min(20, a.sweepX[-1])
        ax[i].plot(a.sweepX[w], a.sweepY[w], lw=0.4)
        ax[i].set_ylabel(f"[{i}] {a.adcNames[i]}\n({a.adcUnits[i]})", fontsize=8)
    ax[0].set_title(f"{os.path.basename(path)} - all channels (first 20 s). "
                    "Pick the 2 spiky/bursting nerve channels.")
    ax[-1].set_xlabel("time (s)")
    out = os.path.join(outdir, f"channels_{os.path.splitext(os.path.basename(path))[0]}.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  -> {out}")


def load_and_concat(paths, ch=(0, 1)):
    """Load ABF files, order by recording time, concat into one timeline.

    Returns dict with: t (s), data (Nsamp x 2), fs, file_bounds (list of
    (name, t_start, t_end)).
    """
    import pyabf
    recs = []
    for p in paths:
        a = pyabf.ABF(p)
        a.setSweep(0, channel=ch[0]); x0 = a.sweepY.copy(); t = a.sweepX.copy()
        a.setSweep(0, channel=ch[1]); x1 = a.sweepY.copy()
        # if multi-sweep, stack sweeps end-to-end
        for s in range(1, a.sweepCount):
            a.setSweep(s, channel=ch[0]); x0 = np.concatenate([x0, a.sweepY])
            a.setSweep(s, channel=ch[1]); x1 = np.concatenate([x1, a.sweepY])
        recs.append(dict(name=os.path.basename(p), when=a.abfDateTime,
                         fs=a.dataRate, x0=x0, x1=x1))
    recs.sort(key=lambda r: r["when"])           # true acquisition order
    fs = recs[0]["fs"]
    x0 = np.concatenate([r["x0"] for r in recs])
    x1 = np.concatenate([r["x1"] for r in recs])
    t = np.arange(len(x0)) / fs
    bounds, off = [], 0.0
    for r in recs:
        dur = len(r["x0"]) / fs
        bounds.append((r["name"], off, off + dur)); off += dur
    return dict(t=t, data=np.column_stack([x0, x1]), fs=fs, file_bounds=bounds)


# ----------------------------------------------------------------------------
# Spike + burst detection
# ----------------------------------------------------------------------------
def detect_spikes(x, fs):
    """Threshold-crossing spike times (s) on |x|, MAD-based noise floor."""
    noise = np.median(np.abs(x)) / 0.6745
    thr = CFG["spike_thresh_k"] * noise
    over = np.abs(x) > thr
    # rising edges of threshold crossing
    cross = np.where((~over[:-1]) & (over[1:]))[0] + 1
    if len(cross) == 0:
        return np.array([]), thr
    refr = int(CFG["refractory_ms"] * 1e-3 * fs)
    keep = [cross[0]]
    for c in cross[1:]:
        if c - keep[-1] >= refr:
            keep.append(c)
    return np.array(keep) / fs, thr


def rate_envelope(x, fs):
    """Rectified, smoothed, downsampled firing-rate envelope of a dense
    extracellular trace. Returns (env, efs)."""
    step = max(1, int(round(fs / CFG["env_ds_hz"])))
    efs = fs / step
    r = np.abs(x)[::step]
    env = gaussian_filter1d(r.astype(float), CFG["env_smooth_s"] * efs)
    return env, efs


def detect_bursts(env, efs, spk):
    """Detect bursts as supra-threshold excursions of the rate ENVELOPE.
    Robust for dense multi-unit nerve activity where ISI gaps don't exist.
    spk = spike times (s) used to count spikes/burst. Returns list of dicts."""
    hi = np.percentile(env, CFG["burst_thr_pctile"])
    lo = np.percentile(env, CFG["burst_off_pctile"])
    # hysteresis: enter a burst above `hi`, stay in until env drops below `lo`
    state = env > hi
    inb = False
    over = np.empty(len(env), bool)
    for i, (h, l) in enumerate(zip(env > hi, env > lo)):
        if not inb and h:
            inb = True
        elif inb and not l:
            inb = False
        over[i] = inb
    thr = hi
    d = np.diff(over.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if over[0]:
        starts = [0] + starts
    if over[-1]:
        ends = ends + [len(over)]
    segs = list(zip(starts, ends))
    # merge segments separated by a short dip
    merged = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) / efs < CFG["burst_merge_s"]:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])
    bursts = []
    for s, e in merged:
        on, off = s / efs, e / efs
        dur = off - on
        if dur < CFG["burst_min_dur_s"] or dur > CFG["burst_max_dur_s"]:
            continue   # too short = noise, too long = merged/artefact
        nsp = int(np.sum((spk >= on) & (spk < off)))
        bursts.append(dict(onset=on, offset=off, dur=dur, n=nsp,
                           ifr=nsp / dur if dur > 0 else np.nan, spikes=None))
    return bursts, thr


def label_coactivity(bA, bB, cycle):
    """Mark each burst on A as 'both' if a B-burst overlaps or its onset is
    within coactive_overlap*cycle, else 'solo'. Mutates burst dicts."""
    win = CFG["coactive_overlap"] * cycle
    onB = np.array([b["onset"] for b in bB]); offB = np.array([b["offset"] for b in bB])
    for b in bA:
        if len(bB) == 0:
            b["state"] = "solo"; continue
        overlap = (onB <= b["offset"] + win) & (offB >= b["onset"] - win)
        b["state"] = "both" if overlap.any() else "solo"
    return bA


# ----------------------------------------------------------------------------
# IFR trace (continuous instantaneous firing rate)
# ----------------------------------------------------------------------------
def ifr_trace(spk, t_end, fs):
    bin_s = CFG["ifr_bin_ms"] * 1e-3
    edges = np.arange(0, t_end + bin_s, bin_s)
    counts, _ = np.histogram(spk, bins=edges)
    rate = counts / bin_s
    sig = CFG["ifr_smooth_ms"] / CFG["ifr_bin_ms"]
    rate = gaussian_filter1d(rate, sig)
    centers = edges[:-1] + bin_s / 2
    return centers, rate


def cv(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.std(v) / np.mean(v) if len(v) > 1 and np.mean(v) != 0 else np.nan


def rcv(v):
    """Robust, outlier-resistant spread: quartile coefficient of dispersion
    (Q3-Q1)/(Q3+Q1).  Lower = more stable. Better than CV for long recordings."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 4:
        return np.nan
    q1, q3 = np.percentile(v, [25, 75])
    return (q3 - q1) / (q3 + q1) if (q3 + q1) != 0 else np.nan


# ----------------------------------------------------------------------------
# Synthetic crawl-CPG data for --demo
# ----------------------------------------------------------------------------
def make_demo():
    """4 'files' x 120 s. Crawl cycle ~6 s. Built so that during CO-ACTIVE
    epochs bursts are regular; during SOLO epochs the active channel is jittery
    -- so the stability hypothesis is visibly testable."""
    rng = np.random.default_rng(0)
    fs = 10000; file_dur = 120.0; n_files = 4
    cycle = 6.0
    paths = []
    base = dict(t=[], A=[], B=[], bounds=[]); off = 0.0
    for f in range(n_files):
        n = int(file_dur * fs); tt = np.arange(n) / fs
        xa = rng.normal(0, 1, n); xb = rng.normal(0, 1, n)
        gphase = tt + off
        # alternate co-active and solo epochs every ~30 s
        for c in range(int(file_dur / cycle)):
            t0 = c * cycle + rng.normal(0, 0.15)
            epoch_both = ((gphase[int(t0*fs)] // 30) % 2 == 0) if int(t0*fs) < n else True
            # channel A burst
            jit = 0.02 if epoch_both else 0.5
            dur = (1.4 if epoch_both else rng.uniform(0.6, 2.2))
            nsp = int((22 if epoch_both else rng.integers(6, 30)))
            _inject(xa, t0 + rng.normal(0, jit), dur, nsp, fs, rng, amp=6)
            # channel B burst: present always in both-epochs, often absent in solo
            if epoch_both or rng.random() < 0.25:
                durB = 1.3 if epoch_both else rng.uniform(0.6, 2.0)
                nspB = 20 if epoch_both else int(rng.integers(6, 26))
                _inject(xb, t0 + (0.4 if epoch_both else rng.uniform(-1, 1)),
                        durB, nspB, fs, rng, amp=6)
        base["A"].append(xa); base["B"].append(xb)
        base["bounds"].append((f"demo_{f+1}.abf", off, off + file_dur))
        off += file_dur
    A = np.concatenate(base["A"]); B = np.concatenate(base["B"])
    t = np.arange(len(A)) / fs
    return dict(t=t, data=np.column_stack([A, B]), fs=fs,
                file_bounds=base["bounds"]), cycle


def _inject(x, t0, dur, nsp, fs, rng, amp=6):
    if t0 < 0 or nsp < 2:
        return
    times = np.sort(rng.uniform(t0, t0 + dur, nsp))
    for ts in times:
        i = int(ts * fs)
        if 0 <= i < len(x) - 20:
            sp = amp * np.exp(-np.arange(-10, 10)**2 / 8.0) * np.sign(rng.normal())
            x[i-10:i+10] += sp


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def analyze_and_plot(rec, outdir, cycle_hint=None):
    os.makedirs(outdir, exist_ok=True)
    t, fs = rec["t"], rec["fs"]
    A, B = rec["data"][:, 0], rec["data"][:, 1]
    t_end = t[-1]

    spkA, thrA = detect_spikes(A, fs)
    spkB, thrB = detect_spikes(B, fs)
    envA, efs = rate_envelope(A, fs)
    envB, _ = rate_envelope(B, fs)
    te = np.arange(len(envA)) / efs
    bA, ethrA = detect_bursts(envA, efs, spkA)   # C3 = rhythmic reference channel
    bB, ethrB = detect_bursts(envB, efs, spkB)   # C4 (often more tonic)

    # cycle estimate from C3 (channel A) burst onsets
    onsetsA = np.array([b["onset"] for b in bA])
    cycle = (np.median(np.diff(onsetsA)) if len(onsetsA) > 2
             else (cycle_hint or 6.0))

    # ---- co-activity per C3 burst, via C4 ENVELOPE level (robust to C4 being
    #      tonic).  "both" = C4 above its LOCAL baseline during the C3 burst.
    #      Local baseline (rolling median, ~120 s) tracks the slow drift across
    #      the 4-hour experiment so classification stays fair throughout. ----
    win = int(120 * efs)
    c4base = pd.Series(envB).rolling(win, center=True, min_periods=win // 4).median().to_numpy()
    for b in bA:
        i0, i1 = int(b["onset"] * efs), int(b["offset"] * efs)
        b["c4"] = float(np.mean(envB[i0:i1])) if i1 > i0 else np.nan
        base = c4base[(i0 + i1) // 2] if (i0 + i1) // 2 < len(c4base) else np.nanmedian(c4base)
        b["state"] = "both" if b["c4"] > base else "solo"

    # ---- per-burst dataframe (C3 reference bursts) ----
    rows = [dict(channel="C3", onset=b["onset"], dur=b["dur"], n_spikes=b["n"],
                 ifr=b["ifr"], c4_env=b["c4"], state=b["state"]) for b in bA]
    df = pd.DataFrame(rows).sort_values("onset")
    df.to_csv(os.path.join(outdir, "bursts.csv"), index=False)
    metrics = [("dur", "burst duration (s)"), ("n_spikes", "spikes / burst"),
               ("ifr", "intraburst IFR (Hz)")]

    # =====================================================================
    # FIG 1 - overview: raw trace + rate envelope + detected bursts (60 s)
    # =====================================================================
    W = min(60, t_end)
    fig, ax = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
    for k, (x, env, bl, ethr, name) in enumerate([
            (A, envA, bA, ethrA, "C3"), (B, envB, bB, ethrB, "C4")]):
        wr = t <= W
        ax[k].plot(t[wr], x[wr], lw=0.25, color="0.6")
        we = te <= W
        scale = np.max(np.abs(x[wr])) / max(env[we].max(), 1e-9)
        ax[k].plot(te[we], env[we] * scale, color="C0", lw=1.3, label="rate envelope")
        ax[k].axhline(ethr * scale, color="k", lw=0.6, ls=":")
        for b in bl:
            if b["onset"] < W:
                # C3 spans colored by co-activity state; C4 spans neutral
                c = ("C0" if b.get("state") == "both" else "C1") if name == "C3" else "0.5"
                ax[k].axvspan(b["onset"], min(b["offset"], W), color=c, alpha=0.15)
        ax[k].set_ylabel(f"{name} (V)")
    ax[0].set_title(f"Overview (first {W:.0f} s) - envelope-detected bursts; "
                    "C3 spans blue=C4 co-active, orange=C4 quiet; C4 spans grey")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "1_overview.png"), dpi=130); plt.close(fig)

    # =====================================================================
    # FIG 2 - rhythm stats (C3 reference)
    # =====================================================================
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    periods = np.diff(onsetsA)
    periods = periods[periods < np.percentile(periods, 99)] if len(periods) else periods
    ax[0].hist(periods, bins=40, color="C0"); ax[0].set_title(f"Cycle period C3 (med {np.median(periods):.1f}s)"); ax[0].set_xlabel("s")
    durs = np.array([b["dur"] for b in bA])
    ax[1].hist(durs, bins=40, color="C2"); ax[1].set_title(f"Burst duration C3 (med {np.median(durs):.1f}s)"); ax[1].set_xlabel("s")
    duty = durs[:len(periods)] / periods[:len(durs)]
    ax[2].hist(duty, bins=40, color="C4"); ax[2].set_title(f"Duty cycle C3 (med {np.median(duty):.2f})"); ax[2].set_xlabel("fraction")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "2_rhythm.png"), dpi=130); plt.close(fig)

    # =====================================================================
    # FIG 3 - THE HYPOTHESIS: stability both-active vs solo (both channels pooled)
    # =====================================================================
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for k, (col, lab) in enumerate(metrics):
        both = df[df.state == "both"][col].values
        solo = df[df.state == "solo"][col].values
        parts = [both[np.isfinite(both)], solo[np.isfinite(solo)]]
        bp = ax[k].boxplot(parts, tick_labels=[
                f"both\nrCV={rcv(both):.2f}\nn={len(parts[0])}",
                f"solo\nrCV={rcv(solo):.2f}\nn={len(parts[1])}"],
                           patch_artist=True, showfliers=False)
        for box, c in zip(bp["boxes"], ["C0", "C1"]):
            box.set_facecolor(c); box.set_alpha(0.5)
        ax[k].set_title(lab)
    fig.suptitle("C3 burst stability by C4 co-activity  "
                 "(rCV = robust spread, lower = more stable)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "3_coactivity_stability.png"), dpi=130); plt.close(fig)

    # =====================================================================
    # FIG 4 - whole-experiment timeline + rolling CV (C3)
    # =====================================================================
    fig, ax = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    cA, rA = ifr_trace(spkA, t_end, fs); cB, rB = ifr_trace(spkB, t_end, fs)
    ax[0].plot(cA, rA, lw=0.5, label="C3"); ax[0].plot(cB, rB, lw=0.5, label="C4")
    ax[0].set_ylabel("pop. rate (Hz)"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Whole experiment (files concatenated by recording time)")

    dfa = df[df.channel == "C3"]
    cmap = (dfa.state == "both").map({True: "C0", False: "C1"})
    ax[1].scatter(dfa.onset, dfa.dur, c=cmap, s=8); ax[1].set_ylabel("C3 burst dur (s)")
    ax[2].scatter(dfa.onset, dfa.n_spikes, c=cmap, s=8); ax[2].set_ylabel("C3 spikes/burst")
    for axi, col in ((ax[1], "dur"), (ax[2], "n_spikes")):
        roll = dfa[col].rolling(15).apply(lambda v: cv(v), raw=False)
        tw = axi.twinx(); tw.plot(dfa.onset, roll, color="k", lw=0.9, alpha=0.6)
        tw.set_ylabel("rolling CV", color="k", fontsize=8)
    for name, t0, t1 in rec["file_bounds"]:
        for a in ax:
            a.axvline(t1, color="0.6", ls="--", lw=0.7)
        ax[0].text(t0, ax[0].get_ylim()[1] * 0.9, name, fontsize=7, color="0.4")
    ax[2].set_xlabel("experiment time (s)   [blue=both active, orange=solo]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "4_experiment_timeline.png"), dpi=130); plt.close(fig)

    # =====================================================================
    # FIG 5 - C3<->C4 coordination: phase of C4 burst within C3 cycle
    # =====================================================================
    phases = []
    onA = np.array([b["onset"] for b in bA])
    for b in bB:
        prev = onA[onA <= b["onset"]]
        nxt = onA[onA > b["onset"]]
        if len(prev) and len(nxt):
            p0, p1 = prev[-1], nxt[0]
            if p1 > p0:
                phases.append((b["onset"] - p0) / (p1 - p0))
    phases = np.array(phases)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if len(phases):
        ax[0].hist(phases, bins=24, range=(0, 1), color="C3")
        ax[0].set_title(f"Phase of C4 burst onset within C3 cycle\n"
                        f"(mean {np.mean(phases):.2f}, vector strength "
                        f"{np.abs(np.mean(np.exp(2j*np.pi*phases))):.2f})")
        ax[0].set_xlabel("phase (0-1)"); ax[0].set_ylabel("# C4 bursts")
    # windowed envelope cross-correlation (detrended) on first 600 s
    n = int(min(600, t_end) * efs)
    a0 = envA[:n] - envA[:n].mean(); b0 = envB[:n] - envB[:n].mean()
    xc = np.correlate(a0, b0, "full"); xc /= (np.std(a0) * np.std(b0) * len(a0))
    lags = (np.arange(len(xc)) - (n - 1)) / efs
    m = np.abs(lags) <= 20
    ax[1].plot(lags[m], xc[m]); ax[1].axvline(0, color="0.6", lw=0.6)
    ax[1].set_title("C3-C4 envelope cross-correlation (first 600 s)")
    ax[1].set_xlabel("lag (s)  [C4 leads <0 > C3 leads]"); ax[1].set_ylabel("corr")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "5_coordination.png"), dpi=130); plt.close(fig)

    # ---- console summary ----
    print(f"\nC3: {len(spkA)} spikes / {len(bA)} bursts;  "
          f"C4: {len(spkB)} spikes / {len(bB)} bursts")
    print(f"Median C3 cycle period: {cycle:.2f} s;  median duty cycle: {np.median(duty):.2f}")
    if len(phases):
        print(f"C4-in-C3 phase: mean {np.mean(phases):.2f}, "
              f"vector strength {np.abs(np.mean(np.exp(2j*np.pi*phases))):.2f}")
    print("\nC3 burst stability (lower = more stable). rCV = robust spread:")
    print(f"{'metric':<20}{'both rCV':>10}{'solo rCV':>10}   {'(both CV':>9}{'solo CV)':>9}")
    for col, lab in metrics:
        b = df[df.state == 'both'][col].values; s = df[df.state == 'solo'][col].values
        print(f"{lab:<20}{rcv(b):>10.3f}{rcv(s):>10.3f}   {cv(b):>9.3f}{cv(s):>9.2f}")
    print(f"\nFigures + bursts.csv -> {outdir}/")
    return df


# ----------------------------------------------------------------------------
def main():
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    demo = "--demo" in argv
    info = "--info" in argv
    # --ch 0,2  -> channels (0, 2)
    ch = (0, 1)
    if "--ch" in argv:
        ch = tuple(int(x) for x in argv[argv.index("--ch") + 1].split(","))
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "figures")
    if demo:
        rec, cyc = make_demo()
        analyze_and_plot(rec, outdir, cycle_hint=cyc)
        return
    paths = args or sorted(glob.glob(os.path.join(here, "*.abf")))
    if not paths:
        print("No .abf files found. Drop recordings here, or run with --demo.")
        return
    if info:
        for p in paths:
            inspect(p, outdir)
        return
    print(f"Loading channels {ch}:", [os.path.basename(p) for p in paths])
    rec = load_and_concat(paths, ch=ch)
    analyze_and_plot(rec, outdir)


if __name__ == "__main__":
    main()
