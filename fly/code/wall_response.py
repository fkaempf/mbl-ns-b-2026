"""The fly's response to a wall, in the wall-on/off framework.

Each bounce (laser-on) is matched to the wall presentation it occurred in
(`bo.wall_trials`), put in that wall's frame (origin = wall centre, wall horizontal,
fly side up), and tagged as the **first contact** with a fresh wall or a **re-bounce**.
Three questions:

  1. speed_after_shock - do flies walk faster after being zapped (ran into a wall)?
     speed just before vs just after each bounce, paired, + the aligned trace.
  2. reflect_or_around - after the hit does the fly **reflect** (retreat, never crosses),
     **walk around** the wall end (crosses beyond +/- width/2), or **push through** the
     middle? Classified per bounce, with the trajectories of each class.
  3. first_vs_rebounce - does the first contact with a new wall differ from re-bounces
     (speed surge and reflect/around/through mix)?

Run from this directory:  python wall_response.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon, ttest_ind, mannwhitneyu

import bounces as bo
from utils import experiment_id, load_combined, save_fig

MODE = "good"      # "good" = the curated runs, "all" = every barrier run
EXPERIMENTS = bo.barrier_experiments(MODE)
SUBDIR = "wall_response" + ("" if MODE == "good" else "_all")
BEF, AFT, DT = 5.0, 30.0, 0.5      # window around the bounce (s)
PRE, POST = (-3.0, -1.0), (1.0, 3.0)   # speed before / after the shock
CROSS = 5.0                        # mm past the wall plane = the fly crossed
CLASS_COL = {"reflect": "#4477aa", "around": "#228833", "through": "#cc3311"}
TORT_BIN_S, TORT_NBINS = 10.0, 4   # tortuosity in 10 s bins, 4 each side of the shock


def cmean(deg):
    r = np.radians(deg)
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


def wrap(d):
    return (d + 180) % 360 - 180


def path_tort(uu, vv):
    """Path length / net displacement (>= 1; NaN if barely moving)."""
    ok = ~np.isnan(uu) & ~np.isnan(vv)
    uu, vv = uu[ok], vv[ok]
    if len(uu) < 3:
        return np.nan
    plen = np.hypot(np.diff(uu), np.diff(vv)).sum()
    disp = np.hypot(uu[-1] - uu[0], vv[-1] - vv[0])
    return plen / disp if disp >= 1 else np.nan


def extract(folder):
    walls = bo.wall_trials(folder)
    bt = bo.laser_on_times(folder)
    if not walls or len(bt) == 0:
        return []
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, x, y, sp, h = (df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy(),
                      df.speed.to_numpy(), df.vrh.to_numpy())
    eid = experiment_id(folder)
    first, owner = set(), {}
    for ton, toff, cx, cy, rot, w, th in walls:
        inb = bt[(bt >= ton) & (bt <= toff)]
        if len(inb):
            first.add(inb[0])
        for tb in inb:
            owner[tb] = (ton, cx, cy, rot, w)
    tau = np.arange(-BEF, AFT + DT, DT)
    out = []
    for tb in bt:
        if tb not in owner:
            continue
        ton, cx, cy, rot, w = owner[tb]
        a = np.radians(rot); c, s = np.cos(a), np.sin(a)
        win = (t >= tb - BEF) & (t <= tb + AFT)
        tw, spw = t[win] - tb, sp[win]
        if win.sum() < 5:
            continue
        dx, dy = x[win] - cx, y[win] - cy
        u, v = dx * c + dy * s, -dx * s + dy * c
        pre = (tw >= -3) & (tw < 0)
        if pre.sum() < 2:
            continue
        if v[pre].mean() < 0:                                   # fly side up
            v = -v
        U = np.interp(tau, tw, u, left=np.nan, right=np.nan)
        V = np.interp(tau, tw, v, left=np.nan, right=np.nan)
        SP = np.interp(tau, tw, spw, left=np.nan, right=np.nan)
        spre = spw[(tw >= PRE[0]) & (tw <= PRE[1])]
        spost = spw[(tw >= POST[0]) & (tw <= POST[1])]
        if not len(spre) or not len(spost):
            continue
        hw = h[win]                                             # heading change after the hit
        hp = (tw >= PRE[0]) & (tw <= PRE[1]) & (spw >= 5)
        ho = (tw >= POST[0]) & (tw <= POST[1]) & (spw >= 5)
        dh = wrap(cmean(hw[ho]) - cmean(hw[hp])) if hp.sum() >= 3 and ho.sum() >= 3 else np.nan
        post = tau > 0
        Vp, Up = V[post], U[post]
        if np.nanmin(Vp) >= -CROSS:                             # never crossed the wall
            cls = "reflect"
        else:
            k = np.where(Vp < -CROSS)[0][0]
            cls = "around" if abs(Up[k]) > w / 2 else "through"
        # tortuosity from the raw path over equal 10 s windows (full resolution, frame-
        # invariant) so it matches the binned shock-aligned plot 07.
        tort_pre = path_tort(x[(t >= tb - 10) & (t < tb)], y[(t >= tb - 10) & (t < tb)])
        tort_post = path_tort(x[(t >= tb) & (t < tb + 10)], y[(t >= tb) & (t < tb + 10)])
        out.append(dict(eid=eid, tau=tau, U=U, V=V, SP=SP, w=w, cls=cls,
                        first=tb in first, spre=float(np.mean(spre)),
                        spost=float(np.mean(spost)), dheading=dh,
                        tort=tort_post, tort_pre=tort_pre, tort_post=tort_post))
    return out


def msem(A):
    A = np.asarray(A, float)
    n = np.sum(~np.isnan(A), 0)
    return np.nanmean(A, 0), np.nanstd(A, 0, ddof=1) / np.sqrt(np.maximum(n, 1))


def plot_speed(G):
    tau = G[0]["tau"]
    m = (tau >= -BEF) & (tau <= 5)
    sp_m, sp_s = msem([b["SP"] for b in G])
    pre = np.array([b["spre"] for b in G]); post = np.array([b["spost"] for b in G])
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cols = {e: cyc[i % len(cyc)] for i, e in enumerate(sorted({b["eid"] for b in G}))}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.fill_between(tau[m], (sp_m - sp_s)[m], (sp_m + sp_s)[m], color="#cc3311", alpha=0.25)
    ax1.plot(tau[m], sp_m[m], color="#cc3311", lw=2)
    ax1.axvline(0, color="0.4", ls="--", lw=1)
    ax1.set_xlabel("time from shock (s)"); ax1.set_ylabel("speed (mm/s)")
    ax1.set_title("walking speed around the shock", fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)

    for b in G:
        ax2.plot([0, 1], [b["spre"], b["spost"]], color=cols[b["eid"]], alpha=0.25, lw=0.6)
    ax2.scatter(np.zeros(len(pre)), pre, color="0.5", s=10, zorder=3)
    ax2.scatter(np.ones(len(post)), post, color="0.5", s=10, zorder=3)
    ax2.plot([0, 1], [np.median(pre), np.median(post)], "k-o", lw=2.5, zorder=4)
    p = wilcoxon(pre, post).pvalue
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["before\n(-3..-1 s)", "after\n(+1..+3 s)"])
    ax2.set_ylabel("speed (mm/s)")
    ax2.set_title(f"before vs after shock: {np.median(pre):.1f} -> {np.median(post):.1f} mm/s "
                  f"({100*(np.median(post)/np.median(pre)-1):+.0f}%, p={p:.1e})", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "01_speed_after_shock.png", subdir=SUBDIR)
    plt.close(fig)


def plot_reflect(G):
    tau = G[0]["tau"]
    m = (tau >= -BEF) & (tau <= AFT)
    counts = {k: [b for b in G if b["cls"] == k] for k in CLASS_COL}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    fr = axes[0]
    ks = list(CLASS_COL)
    fr.bar(range(3), [len(counts[k]) / len(G) for k in ks], color=[CLASS_COL[k] for k in ks])
    fr.set_xticks(range(3)); fr.set_xticklabels([f"{k}\n(n={len(counts[k])})" for k in ks])
    fr.set_ylabel("fraction of bounces"); fr.set_title("reflect vs walk-around vs through", fontsize=10)
    fr.spines[["top", "right"]].set_visible(False)
    for ax, k in zip(axes[1:], ks):
        g = counts[k]
        w = g[0]["w"] if g else 100
        ax.add_patch(Rectangle((-w / 2, -bo.WALL_TH / 2), w, bo.WALL_TH, facecolor="0.4",
                               edgecolor="k", lw=0.6, zorder=2))
        for b in g:
            ax.plot(b["U"][m], b["V"][m], color=CLASS_COL[k], alpha=0.3, lw=0.6, zorder=3)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_aspect("equal"); ax.set_xlabel("along wall (mm)")
        ax.set_title(f"{k}  ({100*len(g)/len(G):.0f}%)", color=CLASS_COL[k], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylabel("perpendicular to wall (mm)")
    save_fig(fig, "02_reflect_or_around.png", subdir=SUBDIR)
    plt.close(fig)


def plot_first_vs_re(G):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for mask, lab, col in [([b for b in G if b["first"]], "first contact", "#1f77b4"),
                           ([b for b in G if not b["first"]], "re-bounce", "#d62728")]:
        surge = np.array([b["spost"] - b["spre"] for b in mask])
        ax1.hist(surge, bins=np.linspace(-15, 25, 21), histtype="step", lw=2, color=col,
                 label=f"{lab} (n={len(mask)}, med {np.median(surge):+.1f})")
    ax1.axvline(0, color="0.5", ls="--", lw=1)
    ax1.set_xlabel("speed surge after shock (mm/s)"); ax1.set_ylabel("count")
    ax1.set_title("speed surge: first contact vs re-bounce", fontsize=10)
    ax1.legend(fontsize=7); ax1.spines[["top", "right"]].set_visible(False)

    ks = list(CLASS_COL)
    fcat = [b["cls"] for b in G if b["first"]]
    rcat = [b["cls"] for b in G if not b["first"]]
    xf = [fcat.count(k) / max(len(fcat), 1) for k in ks]
    xr = [rcat.count(k) / max(len(rcat), 1) for k in ks]
    x = np.arange(3)
    ax2.bar(x - 0.2, xf, 0.4, label=f"first (n={len(fcat)})", color="#1f77b4")
    ax2.bar(x + 0.2, xr, 0.4, label=f"re-bounce (n={len(rcat)})", color="#d62728")
    ax2.set_xticks(x); ax2.set_xticklabels(ks)
    ax2.set_ylabel("fraction"); ax2.set_title("reflect/around/through: first vs re-bounce", fontsize=10)
    ax2.legend(fontsize=7); ax2.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "03_first_vs_rebounce.png", subdir=SUBDIR)
    plt.close(fig)


def plot_heading_change(G):
    """How much the fly turns after the hit: the heading change, as a circular rose
    and a linear histogram (0 = kept its heading, +/-180 = reversed)."""
    d = np.array([b["dheading"] for b in G])
    d = d[~np.isnan(d)]
    rev = np.mean(np.abs(d) > 90)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    edges = np.linspace(-np.pi, np.pi, 37)
    counts, _ = np.histogram(np.radians(d), bins=edges)
    ax1.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges),
            color="#4477aa", edgecolor="white")
    ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
    ax1.set_title("heading change after the hit (rose)\n0 = kept heading,  +/-180 = reversed",
                  fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(d, bins=np.linspace(-180, 180, 37), color="#4477aa", edgecolor="white")
    ax2.axvline(0, color="0.5", ls="--", lw=1)
    ax2.set_xlabel("heading change after the hit (deg; <0 = right, >0 = left)")
    ax2.set_ylabel("bounces")
    ax2.set_title(f"n={len(d)}, median |turn| {np.median(np.abs(d)):.0f} deg, "
                  f"{100*rev:.0f}% reverse", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "04_heading_change.png", subdir=SUBDIR)
    plt.close(fig)


def plot_tortuosity_by_class(G):
    """Post-bounce path tortuosity (path / net displacement) split by wall outcome,
    with pairwise Welch t-tests (Mann-Whitney in brackets, robust to the skew)."""
    ks = list(CLASS_COL)
    data = [[b["tort"] for b in G if b["cls"] == k and np.isfinite(b["tort"])] for k in ks]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.boxplot(data, showfliers=False, widths=0.5, medianprops=dict(color="k", lw=2))
    for i, (k, d) in enumerate(zip(ks, data), 1):
        ax.scatter(np.random.normal(i, 0.05, len(d)), d, s=10, color=CLASS_COL[k],
                   alpha=0.4, zorder=3)
    txt = []
    for i, j in [(0, 2), (0, 1), (1, 2)]:
        pt = ttest_ind(data[i], data[j], equal_var=False).pvalue
        pm = mannwhitneyu(data[i], data[j]).pvalue
        txt.append(f"{ks[i]} vs {ks[j]}:  t p={pt:.3f}  (MWU {pm:.3f})")
    ax.text(0.98, 0.97, "\n".join(txt) + "\npooled bounces (pseudoreplicated)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xticks(range(1, len(ks) + 1))
    ax.set_xticklabels([f"{k}\n(n={len(d)}, med {np.median(d):.1f})" for k, d in zip(ks, data)])
    ax.set_ylabel("post-bounce path tortuosity (path / net displacement)")
    ax.set_title("path tortuosity by wall outcome (reflect / around / through)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "05_tortuosity_by_class.png", subdir=SUBDIR)
    plt.close(fig)


def tortuosity_aligned(experiments):
    """Per-bounce path tortuosity in 10 s bins around the laser-on (rows = bounces)."""
    edges = np.arange(-TORT_NBINS, TORT_NBINS + 1) * TORT_BIN_S
    rows = []
    for f in experiments:
        bt = bo.laser_on_times(f)
        df = load_combined(f, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
        t, x, y = df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy()
        for tb in bt:
            rows.append([path_tort(x[(t >= tb + lo) & (t < tb + hi)],
                                   y[(t >= tb + lo) & (t < tb + hi)])
                         for lo, hi in zip(edges[:-1], edges[1:])])
    return np.array(rows), edges


def plot_tortuosity_aligned(experiments):
    """Tortuosity binned and aligned to the shock (the laser-on analog of interwall 04)."""
    T, edges = tortuosity_aligned(experiments)
    centers = (edges[:-1] + edges[1:]) / 2
    m, sem = msem(T)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, c in enumerate(centers):
        yj = T[:, j][np.isfinite(T[:, j])]
        ax.scatter(np.full(yj.size, c), yj, color="0.7", alpha=0.15, s=8, zorder=1)
    ax.errorbar(centers, m, yerr=sem, fmt="o-", color="#cc3311", capsize=3, lw=2,
                zorder=3, label="mean +/- SEM")
    ax.axvline(0, color="red", ls="--", lw=1, label="shock (laser on)")
    ax.set_xlabel("time relative to shock (s)")
    ax.set_ylabel("path tortuosity (path / net displacement)")
    ax.set_ylim(1, np.nanpercentile(T, 95))
    ax.set_title(f"tortuosity around the shock, 10 s bins (n={T.shape[0]} bounces)", fontsize=10)
    ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "06_tortuosity_aligned.png", subdir=SUBDIR)
    plt.close(fig)


def main():
    G = []
    for f in EXPERIMENTS:
        try:
            b = extract(f)
        except Exception as e:
            print(f"  {f}: skip ({type(e).__name__})"); continue
        G += b
        print(f"  {experiment_id(f)}: {len(b)} bounces")
    print(f"total: {len(G)} bounces  | first {sum(b['first'] for b in G)}, "
          f"re {sum(not b['first'] for b in G)}")
    cc = {k: sum(b['cls'] == k for b in G) for k in CLASS_COL}
    print(f"  reflect/around/through: {cc}")
    plot_speed(G)
    plot_reflect(G)
    plot_first_vs_re(G)
    plot_heading_change(G)
    plot_tortuosity_by_class(G)
    plot_tortuosity_aligned(EXPERIMENTS)
    print(f"done -> plots/{SUBDIR}/")


if __name__ == "__main__":
    main()
