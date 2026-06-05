#!/usr/bin/env python
"""2x2 factorial summary for the juveniles: dopamine x food.

The juvenile set is a complete 2x2 (one video per cell, 2 tracked leeches each):
  DA+NoFood = Video1_dish0    DA+Food  = Video1_dish1
  Veh+NoFood= Video2_dish0    Veh+Food = Video2_dish1
Each cell bar is the MEAN of that cell's 2 leeches. Only Veh+NoFood (Video2_dish0)
is high-confidence tracking (PCK 90); the other 3 cells are low-conf (hatched).
Mirrors the adult analysis_fed_leech/plots/design_2x2_effects.png for comparison.
No em dashes."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/leech/special_project/analysis_juvenile"
MET = f"{BASE}/metrics"
OUT = f"{BASE}/plots"

# base treatment (DISPLAY without the trailing " L0"/" L1") -> (drug, food)
CELL = {
    "Dopamine":   ("DA",  "NoFood"),
    "DA+Food":    ("DA",  "Food"),
    "Veh+NoFood": ("Veh", "NoFood"),
    "Veh+Food":   ("Veh", "Food"),
}
LOWCONF_CELLS = {"Dopamine", "DA+Food", "Veh+Food"}   # all but Veh+NoFood
VEH_C, DA_C = "0.78", "#d62728"


def base_treat(s):
    return str(s).rsplit(" L", 1)[0]


def cell_mean(csv, value_col, ratio_den=None):
    """Return {base_treatment: mean over its leeches} for value_col (or a ratio)."""
    df = pd.read_csv(f"{MET}/{csv}")
    tcol = "treatment"
    df["cell"] = df[tcol].map(base_treat)
    if ratio_den is not None:
        df["_v"] = df[value_col] / df[ratio_den]
    else:
        df["_v"] = df[value_col]
    return df.groupby("cell")["_v"].mean().to_dict()


pct_moving = cell_mean("kinematics_summary.csv", "pct_time_moving")
headtail = cell_mean("kinematics_summary.csv", "total_head_path_cm", ratio_den="total_tail_path_cm")
area = cell_mean("spaceuse_metrics.csv", "area_grid_cm2")
turning = cell_mean("turning_metrics.csv", "turning_per_min_deg")

PANELS = [
    (pct_moving, "% time moving", "% time moving"),
    (headtail, "head:tail path (foraging probe)", "head / tail path"),
    (area, "area explored (cm2)", "occupancy area (cm2)"),
    (turning, "turning (deg/min)", "deg / min"),
]

# style matched to the adult analysis_fed_leech 2x2: two groups (NoFood, Food),
# within each a touching Veh/DA pair, no bar edges/hatching, two-level x labels,
# full box spines, no y-axis label.
foods = ["NoFood", "Food"]
centers = np.array([0.0, 1.0])
w = 0.4

fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, (data, title, ylab) in zip(axes, PANELS):
    for di, (drug, col) in enumerate([("Veh", VEH_C), ("DA", DA_C)]):
        vals = []
        for food in foods:
            name = next(n for n, c in CELL.items() if c == (drug, food))
            vals.append(data.get(name, np.nan))
        ax.bar(centers + (di - 0.5) * w, vals, w, color=col,
               label="Vehicle" if drug == "Veh" else "Dopamine")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([centers[0] - w / 2, centers[0] + w / 2,
                   centers[1] - w / 2, centers[1] + w / 2])
    ax.set_xticklabels(["Veh", "DA", "Veh", "DA"])
    ax.tick_params(axis="x", length=0)
    ax.text(0.25, -0.16, "NoFood", transform=ax.transAxes, ha="center", va="top", fontsize=11)
    ax.text(0.75, -0.16, "Food", transform=ax.transAxes, ha="center", va="top", fontsize=11)
    ax.set_title(title, fontsize=11)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", frameon=True, ncol=2)
fig.suptitle("2x2 design: dopamine x food effects (juveniles; each cell = mean of 2 leeches, "
             "low-conf except Veh+NoFood)", fontsize=14)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig(f"{OUT}/design_2x2_effects.png", dpi=130)
plt.close(fig)
print("wrote design_2x2_effects.png")
for data, title, _ in PANELS:
    print(title, {k: round(v, 2) for k, v in data.items()})
