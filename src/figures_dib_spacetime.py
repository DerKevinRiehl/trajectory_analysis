import os
import sys
import pickle

import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from matplotlib import gridspec
from matplotlib.patches import Rectangle
from _constants import VEHICLE_INFO_PATH, FILTERING_SAMPLING_FREQUENCY
from tools_trajectory_plotting import plot_accelerations
from tools_trajectory_plotting import plot_time_space_diagram


# Root Directories
DATA_ROOT = "../data_trajectories/6_final_trajectories/"
RCSN_ROOT = "../data_trajectories/7_final_trajectories_reconstructed/"

RELEVANT_VIDEO = "DJI_0943.MOV"


CIRCLE_RADIUS = 32.8
CIRCLE_CIRCUMFERENCE = 2*np.pi*CIRCLE_RADIUS - 35


###################################################################################
# Main: Figure 10
###################################################################################
# Load
df = pd.read_csv(RCSN_ROOT + RELEVANT_VIDEO + "_norelax.txt", sep=",")
unique_vehicles = set(df["Vehicle_ID"].tolist())

plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
cmap = plt.cm.rainbow  # Choose colormap (e.g., plasma, magma, inferno)
norm = plt.Normalize(vmin=df["v_Vel"].min(), vmax=df["v_Vel"].max())  # Global color scaling

for vehicle in unique_vehicles:
    df_sub = df[df["Vehicle_ID"]==vehicle]
    global_time = df_sub["Global_Time"].tolist()
    lane_x = df_sub["Lane_X"].tolist()
    velocity = df_sub["v_Vel"].tolist()
    # Split to parts
    plots = []
    current_x = []
    current_time = []
    current_vel = []
    ctr = 0
    for t, x, vel in zip(global_time, lane_x, velocity):
        if x - ctr*CIRCLE_CIRCUMFERENCE > CIRCLE_CIRCUMFERENCE:
            if len(current_x)>0:
                plots.append([current_time, current_x, current_vel])
            current_x = []
            current_time = []
            current_vel = []
            ctr += 1
        current_x.append(x - ctr*CIRCLE_CIRCUMFERENCE)
        current_vel.append(vel)
        current_time.append(t)
    # Last segment
    if current_x:
        plots.append([current_time, current_x, current_vel])
    # Plot
    for part in plots:
        x = np.array(part[0])
        y = np.array(part[1]) 
        velocities = np.array(part[2]) 
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, array=velocities)
        line = ax.add_collection(lc)

ax.autoscale_view()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Vehicle Position [m]\n(Resetted every circumference)')

cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    label='Velocity [m/s]'
)

fig.tight_layout()
plt.show()
