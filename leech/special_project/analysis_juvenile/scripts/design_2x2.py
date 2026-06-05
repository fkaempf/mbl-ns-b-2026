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
VEH_C, DA_C = "0.7", "#d62728"


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

foods = ["NoFood", "Food"]
x = np.arange(len(foods))
w = 0.38

fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, (data, title, ylab) in zip(axes, PANELS):
    for di, (drug, col) in enumerate([("Veh", VEH_C), ("DA", DA_C)]):
        vals, hatches = [], []
        for food in foods:
            # find the base treatment matching this (drug, food) cell
            name = next(n for n, c in CELL.items() if c == (drug, food))
            vals.append(data.get(name, np.nan))
            hatches.append("//" if name in LOWCONF_CELLS else "")
        bars = ax.bar(x + (di - 0.5) * w, vals, w,
                      color=col, label="Vehicle" if drug == "Veh" else "Dopamine",
                      edgecolor="k", linewidth=0.6)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(foods)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylab, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", frameon=True, ncol=2)
fig.suptitle("2x2 design: dopamine x food effects (juveniles; each bar = mean of 2 leeches; "
             "hatched = low-conf, only Veh+NoFood is high-conf)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/design_2x2_effects.png", dpi=130)
plt.close(fig)
print("wrote design_2x2_effects.png")
for data, title, _ in PANELS:
    print(title, {k: round(v, 2) for k, v in data.items()})
