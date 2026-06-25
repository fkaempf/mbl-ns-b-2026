%matplotlib qt


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
path0 = r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_08\rig1_experiment_08_20260623_151614_fictrac_node_fulltrack.csv"
path1 = r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_14\rig1_experiment_14_20260623_162101_fictrac_node_fulltrack.csv"
path2 = r"C:\Users\user\Desktop\2026_06_23\rig1_experiment_23\rig1_experiment_23_20260623_171356_fictrac_node_fulltrack.csv"


fig,ax = plt.subplots(1,3,figsize=(12,4))
for i,x in enumerate([path0,path1,path2]):

    data = pd.read_csv(x)
    
    
    
    x = -np.asarray(data.integrated_position_lab_1.copy()) * 3.065
    
    y = np.asarray(data.integrated_position_lab_0.copy()) * 3.065
    
    time_unix_nano =np.asarray(data.timestamp.copy())
    
    time_s = (time_unix_nano - time_unix_nano[0])/1e9
    
    
    ax[i].scatter(x[0], y[0], c='#0FFF50', zorder=3)
    ax[i].plot(x, y, zorder=2)
    
    
    ax[i].axis('equal')
    ax[i].axis('off')
    
    
    scalebar = AnchoredSizeBar(
        ax[i].transData,
        100, '10 cm', 
        loc='lower right',               # Acts as the alignment origin point
        bbox_to_anchor=(0.95, 0.05),     # (x, y) precisely 5% away from the edges
        bbox_transform=ax[i].transAxes,     # Tells it to use 0-1 axes coordinates
        pad=0.5,
        color='black',       
        frameon=False,
        size_vertical=1
    )
    
    ax[0].add_artist(scalebar)

fig.savefig(os.path.join(plots_dir, "trajectories_3x.png"))
plt.show()


