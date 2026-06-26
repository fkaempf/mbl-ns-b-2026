"""VR heading before vs after the confinement, gated by walking speed.

For each experiment the fulltrack and vrpos logs are merged (utils.load_combined):
the VR heading is kept only while the fly walks faster than MIN_SPEED. Heading is
sampled in a window just before vs just after the confinement. Per experiment we
draw the before/after heading histogram; across experiments we draw a scatter of
mean heading before (x) vs after (y).

Edit the config block. Run:  python vrpos_heading_hist.py
"""

import numpy as np
import matplotlib.pyplot as plt

import scalloping as sc
from utils import experiment_id, load_combined, save_panel

# --- config (edit) ---------------------------------------------------------
EXPERIMENTS = [
    "20260624/rig1_experiment_05",
    "20260624/rig1_experiment_06",
    "20260624/rig1_experiment_16",
    "20260624/rig1_experiment_18",
    "20260624/rig1_experiment_20",
]
BEFORE_MIN = (10, 15)        # heading window (min from start) before the confinement
AFTER_MIN = (30, 35)         # ... and after it
MIN_SPEED = 5.0              # mm/s; headings slower than this are discarded
VR_HEADING = "proper_rotation_z"

SUBDIR = "heading"
BLUE, GREEN = "#4477aa", "#228833"
CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def minimal(ax):
    """Drop the top/right spines for a cleaner look."""
    ax.spines[["top", "right"]].set_visible(False)


def circ(deg):
    """Circular mean and SEM (deg) of a heading sample (~1 independent sample/s)."""
    if len(deg) == 0:
        return np.nan, np.nan
    mu, _ = sc.circ_mean_R(np.radians(deg))
    sem = sc.circ_sem(np.radians(deg), n_eff=max(1, len(deg) // 120))
    return np.degrees(mu), np.degrees(sem)


def before_after_heading(folder):
    """Speed-gated VR heading (deg) in the before and after windows."""
    df = load_combined(folder, vr_heading=VR_HEADING, min_speed_mm_s=MIN_SPEED)

    def win(lo, hi):
        return df.vrh[(df.t >= lo * 60) & (df.t < EXPERIMENTS * 60)].to_numpy()

    return win(*BEFORE_MIN), win(*AFTER_MIN)


def plot_histogram(eid, before, after):
    def draw(ax):
        bins = np.linspace(-180, 180, 37)
        ax.hist(before, bins=bins, density=True, color=BLUE, alpha=0.55, label="before")
        ax.hist(after, bins=bins, density=True, color=GREEN, alpha=0.55, label="after")
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, 0, 180])
        ax.set_xlabel("heading (deg)")
        ax.set_ylabel("density")
        ax.set_title(eid)
        ax.legend(frameon=False)
        minimal(ax)

    save_panel(SUBDIR, f"{eid}_heading_hist.png", draw, figsize=(6, 3.5))



def plot_scatter(records):
    def draw(ax):
        ax.axline((0, 0), slope=1, color="0.8", ls="--", lw=1, zorder=0)
        for i, r in enumerate(records):
            ax.errorbar(r["mub"], r["mua"], xerr=r["semb"], yerr=r["sema"], fmt="o",
                        ms=6, capsize=2, color=CYCLE[i % len(CYCLE)], label=r["eid"])
        ax.set(xlim=(-180, 180), ylim=(-180, 180), xticks=[-180, 0, 180],
               yticks=[-180, 0, 180], xlabel="heading before (deg)",
               ylabel="heading after (deg)")
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=7)
        minimal(ax)
    
    save_panel(SUBDIR, "_summary_mean_heading.png", draw, figsize=(5, 5),
               title="mean heading before vs after")


def main():
    print(f"=== VR heading (speed > {MIN_SPEED:g} mm/s) · "
          f"before {BEFORE_MIN} vs after {AFTER_MIN} min ===")
    records = []
    for folder in EXPERIMENTS:
        eid = experiment_id(folder)
        before, after = before_after_heading(folder)
        mub, semb = circ(before)
        mua, sema = circ(after)
        records.append(dict(eid=eid, mub=mub, semb=semb, mua=mua, sema=sema))
        plot_histogram(eid, before, after)
        print(f"  {eid}: before {mub:6.1f}±{semb:.1f}  after {mua:6.1f}±{sema:.1f}  "
              f"n={len(before):,}/{len(after):,}")
    plot_scatter(records)
    plt.show()


if __name__ == "__main__":
    main()
