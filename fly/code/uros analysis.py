%matplotlib qt

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# path = r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_04\rig1_experiment_04_20260623_135929_fictrac_node_fulltrack.csv"
# path = r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_05\rig1_experiment_05_20260623_141611_fictrac_node_fulltrack.csv"
# path = r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_07\rig1_experiment_07_20260623_144733_fictrac_node_fulltrack.csv"
path= r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_08\rig1_experiment_08_20260623_151614_fictrac_node_fulltrack.csv"

data = pd.read_csv(path)

# %%

# x = data.integrated_position_lab_0

x = -np.asarray(data.integrated_position_lab_1.copy()) * 3.065

y = np.asarray(data.integrated_position_lab_0.copy()) * 3.065

#%%

fig,ax = plt.subplots(1,1)
ax.plot(x,y)
fig.savefig(os.path.join(plots_dir, "trajectory.png"))

fig2,ax2 = plt.subplots(1,1)
ax2.plot(x,y)
ax2.axis('equal')
fig2.savefig(os.path.join(plots_dir, "trajectory_equal.png"))

# %%

time_unix_nano =np.asarray(data.timestamp.copy())

time_s = (time_unix_nano - time_unix_nano[0])/1e9

# %%

heading = ((data.integrated_heading_lab.copy()))
                     
fig,ax = plt.subplots(1,1, figsize=(25,1.5))

ax.plot(time_s,heading)
fig.savefig(os.path.join(plots_dir, "heading.png"))

# %%

fig,ax = plt.subplots(1,1, figsize=(25,1.5))

delta = np.diff(heading,prepend=0)

idx = abs(delta) > np.pi

heading[idx] = np.nan

plt.plot(time_s,heading)
fig.savefig(os.path.join(plots_dir, "heading_wrapped.png"))

# %%
heading_degree = heading*180/np.pi
delta = np.diff(heading_degree,prepend=0)

# idx = abs(delta) > 90

# heading[idx] = np.nan

heading_corrected = (heading_degree + 180)%360 - 180

fig,ax = plt.subplots(1,1, figsize=(25,1.5))
plt.plot(time_s,heading_corrected)
fig.savefig(os.path.join(plots_dir, "heading_corrected.png"))
# plt.hist(heading,50)