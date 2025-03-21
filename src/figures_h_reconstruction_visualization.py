import os
import sys
import pickle

import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.patches import Rectangle, FancyArrowPatch

from _constants import VEHICLE_INFO_PATH
from tools_trajectory_plotting import plot_accelerations


# Root Directories
DATA_ROOT = "../data_trajectories/6_final_trajectories/"
RCSN_ROOT = "../data_trajectories/7_final_trajectories_reconstructed/"
RELEVANT_VIDEO = "DJI_0933.MOV"

# Plot settings
# plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
color_before = "burlywood"
color_proposed = "blueviolet"
color_pibutter = "forestgreen"

###################################################################################
# Main
###################################################################################
df = pd.read_csv(DATA_ROOT + RELEVANT_VIDEO + ".txt", sep=",")
df = plot_accelerations(df)
plt.close()
vehicle_id = "VEHICLE_1"
vehicle_df = df[df["Vehicle_ID"] == vehicle_id]
if not vehicle_df["Frame_ID"].is_monotonic_increasing:
    vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
vehicle_df = vehicle_df.reset_index().drop(columns=["index"])

df_info = pd.read_csv(VEHICLE_INFO_PATH + RELEVANT_VIDEO + ".txt", sep="\t")
mfc_car_id = df_info.loc[df_info["Vehicle_ID"] == vehicle_id, "MFC_CarID"].item()
with open(os.path.join(VEHICLE_INFO_PATH, f"ID{mfc_car_id}_AccelCapInterp.pkl"), 'rb') as f:
    accel_max_spl = pickle.load(f)
with open(os.path.join(VEHICLE_INFO_PATH, f"ID{mfc_car_id}_DecelCapInterp.pkl"), 'rb') as f:
    decel_min_spl = pickle.load(f)

recon_window, recon_step = 10.0, 8.0
t_start = 8

fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=100)
fc = colors.to_rgba('darkgray')
fc = fc[:-1] + (0.3,)
axs[0].add_patch(
    Rectangle(xy=(t_start, 0), width=recon_window, height=75, edgecolor = 'black', facecolor=fc, fill=True, lw=1, label="Reconstruction Window $i$")
)
fc = colors.to_rgba('lightsteelblue')
fc = fc[:-1] + (0.3,)
axs[0].add_patch(
    Rectangle(xy=(t_start+recon_step, 0), width=recon_window, height=75, edgecolor = 'darkblue', facecolor=fc, fill=True, lw=1, label="Reconstruction Window $i+1$")
)
arr = FancyArrowPatch(posA=(t_start, 45), posB=(t_start+recon_window, 45), arrowstyle='<|-|>,head_width=.15', mutation_scale=20, facecolor='black', fill=True, lw=1)
axs[0].add_patch(arr)
axs[0].annotate(r'$w_{recon}$', (.5, .5), xycoords=arr, ha='center', va='bottom')
arr = FancyArrowPatch(posA=(t_start, 36), posB=(t_start+recon_step, 36), arrowstyle='<|-|>,head_width=.15', mutation_scale=20, facecolor='black', fill=True, lw=1)
axs[0].add_patch(arr)
axs[0].annotate(r'$w_{step}$', (.5, .5), xycoords=arr, ha='center', va='bottom')
arr = FancyArrowPatch(posA=(t_start+recon_step, 25), posB=(t_start+recon_step+recon_window, 25), arrowstyle='<|-|>,head_width=.15', mutation_scale=20, facecolor='darkblue', fill=True, lw=1)
axs[0].add_patch(arr)
axs[0].annotate(r'$w_{recon}$', (.5, .5), xycoords=arr, ha='center', va='bottom', color='darkblue')

axs[0].plot(vehicle_df["Global_Time"], vehicle_df["Lane_X"]-vehicle_df["Lane_X"].iloc[0], label="Trajectory Before Reconstruction", color=color_before, linewidth=2)
axs[0].set_xlim([t_start-2.0, t_start + 2*recon_window ])
axs[0].set_ylim([10, 70])
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Position [m]")
axs[0].grid()
axs[0].legend(loc='upper left', fontsize="small")
axs[0].set_title("(a) Reconstruction Sliding Window", fontweight="bold", fontsize="medium")

v_min, v_max = 0, 40
speeds = np.linspace(v_min, v_max, 100)
accel_max = accel_max_spl(speeds)
accel_min = decel_min_spl(speeds)
axs[1].set_title("(b) Vehicle and Driver Dynamics Constraints", fontweight="bold", fontsize="medium")
axs[1].plot(speeds, accel_max, label="Vehicle Acceleration Capacity", color="black", linestyle="--")
axs[1].plot(speeds, 0.5*accel_max, label="Common Driver Acceleration Capacity", color="black")
axs[1].plot(speeds, accel_min, label="Vehicle Deceleration Capacity", color="red")
axs[1].set_ylim([-5, 15])
axs[1].grid()
axs[1].set_xlabel("Speed [m/s]")
axs[1].set_ylabel("Acceleration [m/s$^2$]")
axs[1].legend(fontsize="small", loc="upper right")

fig.tight_layout()
fig.savefig(f"./Reconstruction_Visualization.pdf", bbox_inches='tight', dpi=100)
plt.show()

