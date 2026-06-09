"""
Spatial ecology / arena maps for a leech food-orientation foraging assay.

Theme: where are leeches relative to food and the dish wall, and which way
do they point. All positions are normalized so the dish is the unit circle
(center 0,0, wall at radius 1).

Coordinate convention: input is IMAGE coordinates (y grows downward). We
negate y everywhere (ny = -(mid_y - cy)/r) so that maps look natural
(y grows upward, math convention). heading_am_rad is already in math
convention (image-y negated) per the dataset spec, so heading arrows are
drawn directly with cos/sin of heading_am_rad.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/food_orientation"
CSV_IN = os.path.join(BASE, "food_orientation.csv")
PLOTS = os.path.join(BASE, "plots")
METRICS = os.path.join(BASE, "metrics")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(METRICS, exist_ok=True)

LABELS = {
    "IMG_2855_dish0.mp4": "2855",
    "IMG_2857_dish0.mp4": "2857",
    "PXL_20260602_183730315.TS_dish0.mp4": "PXL1-d0",
    "PXL_20260602_183730315.TS_dish1.mp4": "PXL1-d1",
    "PXL_20260602_210739662.TS_dish1.mp4": "PXL2-d1",
}
ORDER = ["IMG_2855_dish0.mp4", "IMG_2857_dish0.mp4",
         "PXL_20260602_183730315.TS_dish0.mp4",
         "PXL_20260602_183730315.TS_dish1.mp4",
         "PXL_20260602_210739662.TS_dish1.mp4"]

ALIGN_CMAP = plt.cm.RdYlGn_r  # 0 deg (aimed at food) -> green, 180 -> red


def norm_xy(df):
    """Add normalized coords with image-y negated (natural orientation)."""
    df = df.copy()
    df["nx_mid"] = (df["mid_x"] - df["arena_cx"]) / df["arena_r"]
    df["ny_mid"] = -(df["mid_y"] - df["arena_cy"]) / df["arena_r"]
    df["nx_ant"] = (df["ant_x"] - df["arena_cx"]) / df["arena_r"]
    df["ny_ant"] = -(df["ant_y"] - df["arena_cy"]) / df["arena_r"]
    df["nx_post"] = (df["post_x"] - df["arena_cx"]) / df["arena_r"]
    df["ny_post"] = -(df["post_y"] - df["arena_cy"]) / df["arena_r"]
    df["nx_food"] = (df["food_x"] - df["arena_cx"]) / df["arena_r"]
    df["ny_food"] = -(df["food_y"] - df["arena_cy"]) / df["arena_r"]
    df["rho_leech"] = np.hypot(df["nx_mid"], df["ny_mid"])
    df["rho_food"] = np.hypot(df["nx_food"], df["ny_food"])
    return df


def draw_circle(ax, lw=1.5):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="0.3", lw=lw, zorder=1)


def fig1_arena_maps(df):
    n = len(ORDER)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    norm = plt.Normalize(0, 180)

    for i, vid in enumerate(ORDER):
        ax = axes[i]
        g = df[df["video"] == vid]
        draw_circle(ax)
        # all food points (faint)
        gf = g.dropna(subset=["nx_food", "ny_food"])
        ax.scatter(gf["nx_food"], gf["ny_food"], marker="*", s=80,
                   facecolor="gold", edgecolor="0.4", alpha=0.35, zorder=2)
        # leech segments post->ant colored by food_align_deg
        gl = g.dropna(subset=["nx_ant", "nx_post"])
        for _, r in gl.iterrows():
            al = r["food_align_deg"]
            col = ALIGN_CMAP(norm(al)) if np.isfinite(al) else "0.6"
            ax.plot([r["nx_post"], r["nx_ant"]],
                    [r["ny_post"], r["ny_ant"]],
                    color=col, lw=1.4, alpha=0.85, zorder=3,
                    solid_capstyle="round")
            # small dot at anterior to show pointing direction
            ax.scatter(r["nx_ant"], r["ny_ant"], s=8, color=col,
                       zorder=4)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.grid(True, color="0.9", lw=0.5)
        ax.set_title("%s  (n=%d frames-leeches)" % (LABELS[vid], len(gl)),
                     fontsize=11)
        ax.set_xlabel("normalized x (dish radius=1)")
        ax.set_ylabel("normalized y")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    sm = plt.cm.ScalarMappable(cmap=ALIGN_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("food_align_deg (0=aimed at food, 180=away)")
    # legend for food marker on an unused/first axis
    leg = [Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                  markeredgecolor="0.4", markersize=12, label="food (all frames)")]
    axes[0].legend(handles=leg, loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle("Arena maps: leech body segments (post->ant) colored by "
                 "alignment to food", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.9, top=0.92, bottom=0.06,
                        wspace=0.25, hspace=0.3)
    out = os.path.join(PLOTS, "spatial_arena_maps.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig2_radial(df):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(15, 6))

    # (a) per-video rho distribution + food rho line
    data = []
    foodrho = []
    labs = []
    for vid in ORDER:
        g = df[df["video"] == vid]
        rl = g["rho_leech"].dropna().values
        data.append(rl)
        foodrho.append(np.nanmedian(g["rho_food"].values))
        labs.append(LABELS[vid])

    parts = axa.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4c72b0")
        pc.set_alpha(0.6)
    for i, fr in enumerate(foodrho):
        if np.isfinite(fr):
            axa.hlines(fr, i + 1 - 0.35, i + 1 + 0.35, color="darkorange",
                       lw=2.5, zorder=5)
    axa.axhline(0.7, color="0.5", ls="--", lw=1, label="rho=0.7 (outer ring)")
    axa.axhline(1.0, color="0.3", ls=":", lw=1, label="wall (rho=1)")
    axa.set_xticks(range(1, len(labs) + 1))
    axa.set_xticklabels(labs)
    axa.set_ylabel("radial position rho (0=center, 1=wall)")
    axa.set_title("(a) Leech radial position per video\n"
                  "(orange bar = food median rho)")
    axa.set_ylim(0, 1.2)
    axa.grid(True, axis="y", color="0.9", lw=0.5)
    leg = [Line2D([0], [0], color="darkorange", lw=2.5, label="food median rho"),
           Line2D([0], [0], color="0.5", ls="--", label="rho=0.7"),
           Line2D([0], [0], color="0.3", ls=":", label="wall rho=1")]
    axa.legend(handles=leg, fontsize=8, loc="lower right")

    # (b) bar of mean leech rho vs food rho
    x = np.arange(len(ORDER))
    w = 0.38
    mean_leech = [np.nanmean(df[df.video == v]["rho_leech"]) for v in ORDER]
    med_food = foodrho
    axb.bar(x - w / 2, mean_leech, w, color="#4c72b0", label="mean leech rho")
    axb.bar(x + w / 2, med_food, w, color="darkorange", label="food median rho")
    axb.axhline(0.7, color="0.5", ls="--", lw=1)
    axb.set_xticks(x)
    axb.set_xticklabels(labs)
    axb.set_ylabel("radial position rho")
    axb.set_title("(b) Mean leech rho vs food rho per video")
    axb.set_ylim(0, 1.1)
    axb.grid(True, axis="y", color="0.9", lw=0.5)
    axb.legend(fontsize=9)

    fig.suptitle("Radial position / thigmotaxis: are leeches wall-hugging "
                 "while food is elsewhere", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(PLOTS, "spatial_radial_thigmotaxis.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig3_foodcentric(df):
    d = df.dropna(subset=["nx_food", "ny_food", "nx_mid", "ny_mid"]).copy()
    d["dx"] = d["nx_mid"] - d["nx_food"]
    d["dy"] = d["ny_mid"] - d["ny_food"]

    panels = ORDER + ["ALL"]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    ext = 2.0

    for i, key in enumerate(panels):
        ax = axes[i]
        if key == "ALL":
            sub = d
            title = "ALL videos pooled (n=%d)" % len(sub)
        else:
            sub = d[d["video"] == key]
            title = "%s (n=%d)" % (LABELS[key], len(sub))
        if len(sub) > 0:
            hb = ax.hexbin(sub["dx"], sub["dy"], gridsize=22,
                           extent=(-ext, ext, -ext, ext),
                           cmap="viridis", mincnt=1)
            fig.colorbar(hb, ax=ax, shrink=0.8, label="count")
        # food at origin
        ax.scatter(0, 0, marker="*", s=260, color="red",
                   edgecolor="white", linewidth=1.2, zorder=5)
        ax.axhline(0, color="white", lw=0.6, alpha=0.5)
        ax.axvline(0, color="white", lw=0.6, alpha=0.5)
        # 0.3 ref ring
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(0.3 * np.cos(th), 0.3 * np.sin(th), color="red", ls="--",
                lw=1, alpha=0.7)
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("dx (leech - food, dish radii)")
        ax.set_ylabel("dy (leech - food, dish radii)")

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Food-centric density: leech midpoints relative to food "
                 "(red star = food, dashed ring = 0.3 radii)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(PLOTS, "spatial_foodcentric_density.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def metrics_csv(df):
    d = df.copy()
    d["fc_dist"] = np.hypot(d["nx_mid"] - d["nx_food"], d["ny_mid"] - d["ny_food"])
    rows = []
    for vid in ORDER:
        g = d[d["video"] == vid]
        rl = g["rho_leech"].dropna()
        fc = g["fc_dist"].dropna()
        rows.append({
            "video": vid,
            "label": LABELS[vid],
            "n": int(g["mid_x"].notna().sum()),
            "mean_leech_rho": round(float(rl.mean()), 4),
            "median_leech_rho": round(float(rl.median()), 4),
            "food_rho(median)": round(float(np.nanmedian(g["rho_food"])), 4)
                if g["rho_food"].notna().any() else np.nan,
            "frac_leeches_outer(rho>0.7)": round(float((rl > 0.7).mean()), 4)
                if len(rl) else np.nan,
            "mean_foodcentric_dist(normalized)": round(float(fc.mean()), 4)
                if len(fc) else np.nan,
            "frac_within_0p3_of_food": round(float((fc < 0.3).mean()), 4)
                if len(fc) else np.nan,
        })
    out = os.path.join(METRICS, "spatial_metrics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out, pd.DataFrame(rows)


def main():
    df = pd.read_csv(CSV_IN)
    df = norm_xy(df)
    f1 = fig1_arena_maps(df)
    f2 = fig2_radial(df)
    f3 = fig3_foodcentric(df)
    mout, mdf = metrics_csv(df)
    print("WROTE:")
    for p in (f1, f2, f3, mout):
        print(" ", p, "exists=", os.path.exists(p))
    print()
    print(mdf.to_string(index=False))


if __name__ == "__main__":
    main()
