"""Trajectory and heading-over-time figures for one experiment.

Saves plots/<experiment>_trajectory*.png and plots/<experiment>_heading*.png.
Run from this directory:  python heading_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import experiment_id, experiment_label, load_trajectory, save_fig

# Experiment folder (relative to fly/data, or absolute). Edit this.
EXPERIMENT = "260624/2026_06_24/rig1_experiment_16"

HEADING_FIGSIZE = (24, 3)  # heading vs time spans ~45 min, so make it wide

data, x, y, time_s = load_trajectory(EXPERIMENT)
label, eid = experiment_label(EXPERIMENT), experiment_id(EXPERIMENT)

# --- Trajectory: default aspect, then equal aspect ---
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(x, y)
save_fig(fig, f"{eid}_trajectory.png", title=label)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(x, y)
ax.axis('equal')
save_fig(fig, f"{eid}_trajectory_equal.png", title=label)

# --- Heading over time ---
heading = data.integrated_heading_lab.copy()

# raw
fig, ax = plt.subplots(1, 1, figsize=HEADING_FIGSIZE)
ax.plot(time_s, heading)
save_fig(fig, f"{eid}_heading.png", title=f"{label} · heading")

# mask the wrap-around discontinuities (|Δ| > π) so the line isn't bridged
heading[np.abs(np.diff(heading, prepend=0)) > np.pi] = np.nan
fig, ax = plt.subplots(1, 1, figsize=HEADING_FIGSIZE)
ax.plot(time_s, heading)
save_fig(fig, f"{eid}_heading_wrapped.png", title=f"{label} · heading (wrap-masked)")

# re-wrap into [-180, 180) degrees (keeps the NaN gaps from the step above)
heading_corrected = (heading * 180 / np.pi + 180) % 360 - 180
fig, ax = plt.subplots(1, 1, figsize=HEADING_FIGSIZE)
ax.plot(time_s, heading_corrected)
save_fig(fig, f"{eid}_heading_corrected.png", title=f"{label} · heading (deg, [-180, 180))")

plt.show()
