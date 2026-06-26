"""Side-by-side trajectory comparison of three experiments -> plots/trajectories_3x.png.

Run from this directory:  python trajectories_3panel.py
"""

import matplotlib.pyplot as plt

from utils import draw_trajectory, experiment_label, load_trajectory, save_fig

# Experiment folders to compare (relative to fly/data, or absolute). Edit these.
EXPERIMENTS = [
    "260624/2026_06_24/rig1_experiment_05",
    "260624/2026_06_24/rig1_experiment_06",
    "260624/2026_06_24/rig1_experiment_12",
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, folder in zip(axes, EXPERIMENTS):
    _, x, y, _ = load_trajectory(folder)
    draw_trajectory(ax, x, y, mark_start=True, title=experiment_label(folder))

save_fig(fig, "trajectories_3x.png")
plt.show()
