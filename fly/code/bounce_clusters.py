"""Per-bounce gallery: one figure per bounce, the trajectory 30 s before / 60 s
after the laser hit, in the bounce's own wall frame with the **true finite wall**
drawn from the barrier's config geometry. Coloured by time from the bounce; titled
with the bounce's order and time in the experiment. (k-means clustering of bounce
types is added in a later step.)

Run from this directory:  python bounce_clusters.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import bounces as bo
from utils import add_scalebar, emptiest_corner, experiment_id, load_combined, save_fig

MODE = "good"      # "good" = the curated runs, "all" = every barrier run
EXPERIMENTS = bo.barrier_experiments(MODE)
BEFORE_S, AFTER_S = 30.0, 60.0   # seconds before / after the bounce (adjustable)
DT = 0.5
N_CLUSTERS = 5
CLUST_WIN = (-10.0, 30.0)        # window the bounce-type clustering looks at
USE_GOOD = False                 # unbiased clustering: no per-bounce preselection
SUBDIR = (f"bounce_gallery{'' if MODE == 'good' else '_all'}/"
          f"{int(BEFORE_S)}pre_{int(AFTER_S)}post")   # window-named folder
DPI = 720         # 7in x 720dpi ~ 5000 px per side
CLUSTER_COLORS = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#aa3377", "#66ccee"]


def current_barrier(walls, tb):
    """The barrier in play at bounce ``tb``: the most recently CREATEd one before it.
    Barriers are spawned and DESTROYed one at a time (one per bounce), so this is
    unambiguous - the collision log is not needed and its normal is unreliable."""
    prior = [w for w in walls if w[5] <= tb]
    return max(prior, key=lambda w: w[5]) if prior else None


def extract(folder):
    bt = bo.laser_on_times(folder)
    walls = bo.rectmaze_walls(folder)
    df = load_combined(folder, min_speed_mm_s=0, max_speed_mm_s=bo.MAX_SPEED)
    t, x, y, sp = df.t_unix.to_numpy(), df.vrx.to_numpy(), df.vry.to_numpy(), df.speed.to_numpy()
    t0, eid, n = t[0], experiment_id(folder), len(bt)
    tau = np.arange(-BEFORE_S, AFTER_S + DT, DT)
    out = []
    for i, tb in enumerate(bt, 1):
        bar = current_barrier(walls, tb)
        if bar is None:
            continue
        cx, cy, rot, bw = bar[0], bar[1], bar[2], bar[3]
        # wall frame straight from the config barrier geometry (the ground truth):
        # origin at the wall centre, normal = rotation_z, oriented toward the fly.
        a = np.radians(rot)
        e_n0 = np.array([-np.sin(a), np.cos(a)])
        w = (t >= tb - BEFORE_S) & (t <= tb + AFTER_S)
        if w.sum() < 10:
            continue
        tw, xw, yw, spw = t[w] - tb, x[w] - cx, y[w] - cy, sp[w]
        pre = (tw >= -bo.INCID_S) & (tw < 0)
        if pre.sum() < 3:
            continue
        e_n = e_n0 if np.array([xw[pre].mean(), yw[pre].mean()]) @ e_n0 >= 0 else -e_n0
        e_t = np.array([e_n[1], -e_n[0]])                        # right-handed tangent
        u = xw * e_t[0] + yw * e_t[1]
        v = xw * e_n[0] + yw * e_n[1]
        mov = pre & (spw >= bo.MIN_SPEED)
        if mov.sum() < 3:
            continue
        du = u[mov][-1] - u[mov][0]
        flip = -1.0 if du < 0 else 1.0                           # approach from the left
        u = u * flip
        out.append(dict(
            eid=eid, i=i, n=n, t_min=(tb - t0) / 60.0, tau=tau,
            U=np.interp(tau, tw, u, left=np.nan, right=np.nan),
            V=np.interp(tau, tw, v, left=np.nan, right=np.nan),
            width=bw))
    return out


def draw_wall(ax, b):
    """The true finite wall (config width x thickness 2), centred at the wall centre,
    plus the laser heat zone (wall inflated by the laser margin)."""
    hw, hth, m = b["width"] / 2, bo.WALL_TH / 2, bo.LASER_MARGIN
    ax.add_patch(Rectangle((-hw, -hth - m), b["width"], bo.WALL_TH + 2 * m,
                           facecolor="red", alpha=0.2, edgecolor="red", lw=0.5, zorder=1))
    ax.add_patch(Rectangle((-hw, -hth), b["width"], bo.WALL_TH,
                           facecolor="0.4", edgecolor="black", lw=0.8, zorder=2))


def cluster_bounces(allb, k=N_CLUSTERS):
    """k-means the bounce trajectories (resampled U,V over CLUST_WIN) into ``k`` types.
    Writes ``b['cluster']`` in place; clusters are relabelled 0..k-1 by size (0 =
    biggest) so colours are stable across runs."""
    tau = allb[0]["tau"]
    m = (tau >= CLUST_WIN[0]) & (tau <= CLUST_WIN[1])
    feats = []
    for b in allb:
        u = pd.Series(b["U"][m]).ffill().bfill().to_numpy()
        v = pd.Series(b["V"][m]).ffill().bfill().to_numpy()
        feats.append(np.concatenate([u, v]))
    X = StandardScaler().fit_transform(np.array(feats))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    order = {c: r for r, c in enumerate(pd.Series(labels).value_counts().index)}
    for b, c in zip(allb, labels):
        b["cluster"] = order[c]


def plot_bounce(b, sub):
    U, V, tau = b["U"], b["V"], b["tau"]
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_wall(ax, b)
    pts = np.column_stack([U, V]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", array=tau[:-1], lw=1.3, zorder=3)
    ax.add_collection(lc)
    i0 = int(np.argmin(np.abs(tau)))
    ax.scatter(U[i0], V[i0], color="black", s=45, zorder=5)      # fly at the bounce
    ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    add_scalebar(ax, loc=emptiest_corner(ax, U, V))
    fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02).set_label("time from bounce (s)")
    c = b["cluster"]
    ax.set_title(f"{b['eid']} · bounce {b['i']}/{b['n']} · t = {b['t_min']:.1f} min · "
                 f"type {c}", fontsize=10, color=CLUSTER_COLORS[c])
    save_fig(fig, f"{b['eid']}_bounce_{b['i']:03d}_type{c}.png", subdir=sub, dpi=DPI)
    plt.close(fig)


def plot_cluster_overview(allb, sub, k=N_CLUSTERS):
    """One panel per bounce type: its members (faint, time window) + mean trajectory."""
    tau = allb[0]["tau"]
    m = (tau >= CLUST_WIN[0]) & (tau <= CLUST_WIN[1])
    fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 6.0), sharex=True, sharey=True)
    for c, ax in enumerate(axes):
        members = [b for b in allb if b["cluster"] == c]
        col = CLUSTER_COLORS[c]
        for b in members:
            ax.plot(b["U"][m], b["V"][m], color=col, alpha=0.18, lw=0.6)
        if members:
            U = np.nanmean([b["U"][m] for b in members], axis=0)
            V = np.nanmean([b["V"][m] for b in members], axis=0)
            ax.plot(U, V, color="black", lw=2)
        ax.axhspan(-bo.WALL_TH / 2, bo.WALL_TH / 2, color="0.4", zorder=0)
        ax.set_title(f"type {c}  (n={len(members)})", color=col, fontsize=10)
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle(f"bounce types (k-means, {CLUST_WIN[0]:.0f}..{CLUST_WIN[1]:.0f}s)", fontsize=11)
    save_fig(fig, "cluster_overview.png", subdir=sub, dpi=200)
    save_fig(fig, "cluster_overview.pdf", subdir=sub)          # vector, for zooming
    plt.close(fig)


def main():
    allb = []
    for f in EXPERIMENTS:
        b = extract(f)
        allb += b
        print(f"  {experiment_id(f)}: {len(b)} bounces")
    print(f"total: {len(allb)} bounces")
    good = bo.good_set() if USE_GOOD else None
    if good is not None:
        before = len(allb)
        allb = [b for b in allb if (b["eid"], b["i"]) in good]
        print(f"restricted to curated good bounces: {len(allb)}/{before}")
    sub = SUBDIR if good is None else SUBDIR.replace("bounce_gallery/", "bounce_gallery_good/")
    cluster_bounces(allb)
    sizes = pd.Series([b["cluster"] for b in allb]).value_counts().sort_index()
    print("cluster sizes:", dict(sizes))
    plot_cluster_overview(allb, sub)
    if len(allb) <= 150:                   # per-bounce gallery only for small sets
        for b in allb:
            plot_bounce(b, sub)
    else:
        print(f"  {len(allb)} bounces: rendered cluster_overview only (skipped per-bounce gallery)")
    print(f"done -> plots/{sub}/")


if __name__ == "__main__":
    main()
