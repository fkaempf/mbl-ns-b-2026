


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# path = r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_04\rig1_experiment_04_20260623_135929_fictrac_node_fulltrack.csv"
# path = r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_05\rig1_experiment_05_20260623_141611_fictrac_node_fulltrack.csv"
# path = r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_07\rig1_experiment_07_20260623_144733_fictrac_node_fulltrack.csv"
path0 = "/Users/fkampf/Documents/mbl/code/mbl-ns-b-2026/fly/data/260624/2026_06_24/rig1_experiment_06/rig1_experiment_06_20260624_115842_fictrac_node_fulltrack.csv"



fig,ax = plt.subplots(1,1,figsize=(12,4))


data = pd.read_csv(path0)
x = -np.asarray(data.integrated_position_lab_1.copy()) * 3.065
y = np.asarray(data.integrated_position_lab_0.copy()) * 3.065
time_unix_nano =np.asarray(data.timestamp.copy())
time_s = (time_unix_nano - time_unix_nano[0])/1e9


fig,ax = plt.subplots(1,1,figsize=(12,4))
ax.scatter(x[0], y[0], c='#0FFF50', zorder=3)
ax.plot(x[time_s<900], y[time_s<900], zorder=2)
ax.axis('equal')
ax.axis('off')
scalebar = AnchoredSizeBar(
    ax.transData,
    100, '10 cm', 
    loc='lower right',               # Acts as the alignment origin point
    bbox_to_anchor=(0.95, 0.05),     # (x, y) precisely 5% away from the edges
    bbox_transform=ax.transAxes,     # Tells it to use 0-1 axes coordinates
    pad=0.5,
    color='black',       
    frameon=False,
    size_vertical=1
)
ax.add_artist(scalebar)
fig.savefig(os.path.join(plots_dir, "trajectory_0_900s.png"))
plt.show()


fig,ax = plt.subplots(1,1,figsize=(12,4))
#ax.scatter(x[0], y[0], c='#0FFF50', zorder=3)
ax.plot(x[(900<time_s)&(time_s<1800)], y[(900<time_s)&(time_s<1800)], zorder=2)
ax.axis('equal')
ax.axis('off')
scalebar = AnchoredSizeBar(
    ax.transData,
    100, '10 cm', 
    loc='lower right',               # Acts as the alignment origin point
    bbox_to_anchor=(0.95, 0.05),     # (x, y) precisely 5% away from the edges
    bbox_transform=ax.transAxes,     # Tells it to use 0-1 axes coordinates
    pad=0.5,
    color='black',       
    frameon=False,
    size_vertical=1
)
ax.add_artist(scalebar)
fig.savefig(os.path.join(plots_dir, "trajectory_900_1800s.png"))
plt.show()

fig,ax = plt.subplots(1,1,figsize=(12,4))
#ax.scatter(x[0], y[0], c='#0FFF50', zorder=3)
ax.plot(x[1800<time_s], y[1800<time_s], zorder=2)
ax.axis('equal')
ax.axis('off')
scalebar = AnchoredSizeBar(
    ax.transData,
    100, '10 cm', 
    loc='lower right',               # Acts as the alignment origin point
    bbox_to_anchor=(0.95, 0.05),     # (x, y) precisely 5% away from the edges
    bbox_transform=ax.transAxes,     # Tells it to use 0-1 axes coordinates
    pad=0.5,
    color='black',       
    frameon=False,
    size_vertical=1
)
ax.add_artist(scalebar)
fig.savefig(os.path.join(plots_dir, "trajectory_1800s_end.png"))
plt.show()



fig,ax = plt.subplots(1,1)
#ax.scatter(x[0], y[0], c='#0FFF50', zorder=3)
ax.plot(x, y, zorder=2)
ax.axis('equal')
ax.axis('off')
scalebar = AnchoredSizeBar(
    ax.transData,
    100, '10 cm', 
    loc='lower right',               # Acts as the alignment origin point
    bbox_to_anchor=(0.95, 0.05),     # (x, y) precisely 5% away from the edges
    bbox_transform=ax.transAxes,     # Tells it to use 0-1 axes coordinates
    pad=0.5,
    color='black',       
    frameon=False,
    size_vertical=1
)
ax.add_artist(scalebar)
fig.savefig(os.path.join(plots_dir, "trajectory_full.png"))
plt.show()


