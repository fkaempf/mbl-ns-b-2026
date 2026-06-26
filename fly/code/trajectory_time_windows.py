"""Trajectory of one experiment, split into consecutive time windows.

Saves plots/<experiment>_trajectory_<window>.png for each window.
Run from this directory:  python trajectory_time_windows.py
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import draw_trajectory, experiment_id, experiment_label, load_trajectory, save_fig

# Experiment folder (relative to fly/data, or absolute). Edit this.
EXPERIMENT = "260624/2026_06_24/rig1_experiment_06"

_, x, y, time_s = load_trajectory(EXPERIMENT)
label, eid = experiment_label(EXPERIMENT), experiment_id(EXPERIMENT)


def plot_window(mask, name, *, mark_start=False):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    draw_trajectory(ax, x[mask], y[mask], mark_start=mark_start)
    save_fig(fig, f"{eid}_trajectory_{name}.png", title=f"{label} · {name}")


# Three consecutive 15-minute windows, plus the full track.
plot_window(time_s < 900, "0_900s", mark_start=True)
plot_window((900 < time_s) & (time_s < 1800), "900_1800s")
plot_window(1800 < time_s, "1800s_end")
plot_window(np.ones_like(time_s, dtype=bool), "full")

plt.show()
