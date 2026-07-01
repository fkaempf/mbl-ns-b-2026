"""Does the fly keep/recover a consistent heading vector across inter-trial intervals?

Within a trial the collision randomises the heading (heading_recovery.py). Here we ask
the long-term question: between successive inter-wall intervals (the free walking from
wall-off to the next wall-on), does the fly return to the same walking-direction vector?

Per run we take the mean walking direction (net displacement, speed-gated) in each
inter-wall interval, then:
  * autocorrelation across interval lag - <cos(dir_{n+L} - dir_n)>: 1 = the vector
    persists from interval to interval, 0 = it is re-randomised each interval,
  * per-run resultant R of all its interval directions - a single stable long-term
    vector (R~1) vs no preferred direction (R~0).

Walking direction (vrh is decoupled from the path here). Run:  python intertrial_persistence.py
"""

import numpy as np
import matplotlib.pyplot as plt

import bounces as bo
from utils import experiment_id, load_combined, save_fig

EXPERIMENTS = bo.barrier_experiments("good")
MIN_SPEED = 5.0
MAX_LAG = 6
SUBDIR = "exploratory"


def iti_dirs(folder):
    """Mean walking direction (deg) in each inter-wall interval; NaN if barely walking."""
    walls = bo.wall_trials(folder)
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, x, y, sp = (df.t_unix.to_numpy(), df.vrx.to_numpy(),
                   df.vry.to_numpy(), df.speed.to_numpy())
    out = []
    for i in range(len(walls) - 1):
        a, b = walls[i][1], walls[i + 1][0]                  # wall-off_i -> wall-on_{i+1}
        m = (t >= a) & (t <= b) & (sp >= MIN_SPEED)
        if m.sum() < 10:
            out.append(np.nan); continue
        xs, ys = x[m], y[m]
        out.append(np.degrees(np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0])))
    return np.array(out)


def plot_consecutive(pa, pb):
    """Heading in interval n vs n+1: scatter (on the diagonal = same direction) and the
    distribution of the heading change between consecutive inter-wall intervals."""
    pa, pb = np.array(pa), np.array(pb)
    change = (pb - pa + 180) % 360 - 180
    same = np.mean(np.abs(change) < 45)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot([-180, 180], [-180, 180], color="0.7", ls="--", lw=1)
    ax1.scatter(pa, pb, s=14, alpha=0.5, color="#4477aa")
    ax1.set_xlim(-180, 180); ax1.set_ylim(-180, 180)
    ax1.set_xticks([-180, -90, 0, 90, 180]); ax1.set_yticks([-180, -90, 0, 90, 180])
    ax1.set_aspect("equal")
    ax1.set_xlabel("heading in interval n (deg)")
    ax1.set_ylabel("heading in interval n+1 (deg)")
    ax1.set_title("consecutive inter-wall headings\n(on the diagonal = same direction; +/-180 wraps)",
                  fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax2.hist(change, bins=np.linspace(-180, 180, 37), color="#4477aa", edgecolor="white")
    ax2.axvspan(-45, 45, color="#228833", alpha=0.10)
    ax2.axvline(0, color="red", ls="--", lw=1)
    ax2.set_xlim(-180, 180); ax2.set_xticks([-180, -90, 0, 90, 180])
    ax2.set_xlabel("heading change between consecutive intervals (deg)")
    ax2.set_ylabel("count")
    ax2.set_title(f"n={len(change)} pairs;  {100*same:.0f}% within +/-45 deg (same direction);  "
                  f"median |change| {np.median(np.abs(change)):.0f} deg", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "intertrial_consecutive.png", subdir=SUBDIR)
    plt.close(fig)


def main():
    perlag = {L: [] for L in range(1, MAX_LAG + 1)}
    Rs, labels = [], []
    pa, pb = [], []
    for f in EXPERIMENTS:
        d = iti_dirs(f)
        dd = d[~np.isnan(d)]
        if len(dd) >= 3:
            c, s = np.cos(np.radians(dd)).mean(), np.sin(np.radians(dd)).mean()
            Rs.append(np.hypot(c, s)); labels.append(experiment_id(f).split("_")[-1])
        for L in range(1, MAX_LAG + 1):
            for n in range(len(d) - L):
                if not np.isnan(d[n]) and not np.isnan(d[n + L]):
                    perlag[L].append(np.cos(np.radians(d[n + L] - d[n])))
        for n in range(len(d) - 1):                          # consecutive (n, n+1) pairs
            if not np.isnan(d[n]) and not np.isnan(d[n + 1]):
                pa.append(d[n]); pb.append(d[n + 1])
    lags = np.arange(1, MAX_LAG + 1)
    ac = np.array([np.mean(perlag[L]) if perlag[L] else np.nan for L in lags])
    acn = np.array([len(perlag[L]) for L in lags])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.axhline(0, color="0.6", ls="--", lw=1, label="0 = re-randomised each interval")
    ax1.axhline(1, color="0.6", ls=":", lw=1, label="1 = vector fully persists")
    ax1.plot(lags, ac, "o-", color="#cc3311", lw=2)
    for L, a, nn in zip(lags, ac, acn):
        ax1.annotate(f"n={nn}", (L, a), textcoords="offset points", xytext=(0, 8), fontsize=6, ha="center")
    ax1.set_xlabel("interval lag (number of inter-trial intervals apart)")
    ax1.set_ylabel("heading autocorrelation  <cos d>")
    ax1.set_ylim(-0.3, 1.0)
    ax1.set_title("does the walking vector persist across inter-trial intervals?", fontsize=10)
    ax1.legend(fontsize=7); ax1.spines[["top", "right"]].set_visible(False)

    order = np.argsort(Rs)[::-1]
    ax2.bar(range(len(Rs)), np.array(Rs)[order], color="#4477aa")
    ax2.set_xticks(range(len(Rs))); ax2.set_xticklabels([labels[i] for i in order], fontsize=7)
    ax2.axhline(np.mean(Rs), color="k", ls="--", lw=1, label=f"mean R={np.mean(Rs):.2f}")
    ax2.set_ylabel("resultant R of inter-trial directions"); ax2.set_ylim(0, 1)
    ax2.set_title("per-run concentration of the long-term heading\n(1 = one stable vector, 0 = no preferred direction)",
                  fontsize=9)
    ax2.legend(fontsize=8); ax2.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "intertrial_persistence.png", subdir=SUBDIR)
    plt.close(fig)
    plot_consecutive(pa, pb)
    print(f"autocorr vs lag: {dict(zip(lags.tolist(), np.round(ac, 2).tolist()))}")
    print(f"per-run R: median {np.median(Rs):.2f}  range {min(Rs):.2f}-{max(Rs):.2f}")


if __name__ == "__main__":
    main()
