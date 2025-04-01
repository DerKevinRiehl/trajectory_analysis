import os
import sys
import pickle

import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.patches import Rectangle
from _constants import VEHICLE_INFO_PATH, FILTERING_SAMPLING_FREQUENCY
from tools_trajectory_plotting import plot_accelerations


# Root Directories
DATA_ROOT = "../data_trajectories/6_final_trajectories/"
RCSN_ROOT = "../data_trajectories/7_final_trajectories_reconstructed/"
RELEVANT_VIDEO = "DJI_0933.MOV"

# Plot settings
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 


###################################################################################
# Functions
###################################################################################
def add_subplot_axes(ax, rect, facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def setup_broken_axis(gs_cell, title, data, ylim_top, ylim_bottom, xlabels):
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell, height_ratios=[1, 3], hspace=0.05)
    ax_top = fig.add_subplot(gs_sub[0])
    ax_bottom = fig.add_subplot(gs_sub[1])
    
    plt.sca(ax_top)
    plt.title(title)
    data.plot.bar(ax=ax_top, edgecolor="black", color=["green","aquamarine"], width=0.75, legend=False, zorder=3)
    ax_top.set_ylim(ylim_top)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.grid(zorder=0)
    
    data.plot.bar(ax=ax_bottom, edgecolor="black", color=["green","aquamarine"], width=0.75, legend=False, zorder=3)
    ax_bottom.set_ylim(ylim_bottom)
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.grid(zorder=0)
    ax_bottom.set_xticklabels(xlabels, rotation=45, ha='right')
    
    # Add diagonal lines
    d = .015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    return ax_bottom

###################################################################################
# Main: Figure 7
###################################################################################
# Load
df = pd.read_csv(DATA_ROOT + RELEVANT_VIDEO + ".txt", sep=",")
df = plot_accelerations(df)
plt.close()
df_info = pd.read_csv(VEHICLE_INFO_PATH + RELEVANT_VIDEO + ".txt", sep="\t")
df_CvxOpt = pd.read_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_CvxOptNoRelax.txt", sep=",")
df_PIButterworth = pd.read_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_PIButterworth.txt", sep=",")


vehicle_id = "VEHICLE_1"
vehicle_df = df[df["Vehicle_ID"] == vehicle_id]
if not vehicle_df["Frame_ID"].is_monotonic_increasing:
    vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
vehicle_df = vehicle_df.reset_index().drop(columns=["index"])
vehicle_df["v_Vel"] = vehicle_df["Lane_X"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY

vehicle_df_CvxOpt = df_CvxOpt[df_CvxOpt["Vehicle_ID"] == vehicle_id]
if not vehicle_df_CvxOpt["Frame_ID"].is_monotonic_increasing:
    vehicle_df_CvxOpt = vehicle_df_CvxOpt.sort_values(by=["Frame_ID"], ascending=True)
vehicle_df_CvxOpt = vehicle_df_CvxOpt.reset_index().drop(columns=["index"])

vehicle_df_PIButterworth = df_PIButterworth[df_PIButterworth["Vehicle_ID"] == vehicle_id]
if not vehicle_df_PIButterworth["Frame_ID"].is_monotonic_increasing:
    vehicle_df_PIButterworth = vehicle_df_PIButterworth.sort_values(by=["Frame_ID"], ascending=True)
vehicle_df_PIButterworth = vehicle_df_PIButterworth.reset_index().drop(columns=["index"])

mfc_car_id = df_info.loc[df_info["Vehicle_ID"] == vehicle_id, "MFC_CarID"].item()
with open(VEHICLE_INFO_PATH + f"ID{mfc_car_id}_AccelCapInterp.pkl", 'rb') as f:
    accel_max_spl = pickle.load(f)
with open(VEHICLE_INFO_PATH + f"ID{mfc_car_id}_DecelCapInterp.pkl", 'rb') as f:
    decel_min_spl = pickle.load(f)

# Plot
color_before = "burlywood"
color_proposed = "blueviolet"
color_pibutter = "forestgreen"

fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=100)
axs[0][0].plot(vehicle_df["Global_Time"], vehicle_df["v_Vel"], label="Before Reconstruction", color=color_before)
axs[0][0].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["v_Vel"], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[0][0].plot(vehicle_df_PIButterworth["Global_Time"], vehicle_df_PIButterworth["v_Vel"], label="PI-Butterworth Reconstruction", linestyle="-.", alpha=0.75, color=color_pibutter)
axs[0][0].set_xlabel("Time [s]")
axs[0][0].set_ylabel("Speed [m/s]")
axs[0][0].grid()
#axs[0][0].set_xlim(0, 150)
axs[0][0].set_ylim(-1.5, 8)
axs[0][0].legend(fontsize="x-small", loc="upper left")
time_arr = vehicle_df["Global_Time"].to_numpy()
t1, t2 = 230, 240
t1_idx, t2_idx = np.argwhere(time_arr == t1)[0][0], np.argwhere(time_arr == t2)[0][0]
v1 = np.amin([vehicle_df["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_CvxOpt["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_PIButterworth["v_Vel"].iloc[t1_idx:t2_idx]])-0.15
v2 = np.amax([vehicle_df["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_CvxOpt["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_PIButterworth["v_Vel"].iloc[t1_idx:t2_idx]])+0.15
axs[0][0].add_patch(
    Rectangle(xy=(t1, v1), width=t2-t1, height=v2-v1, edgecolor = 'black', fill=False, lw=1)
)
subpos = [0.45, 0.24, 0.32, 0.32]
subax = add_subplot_axes(axs[0][0], subpos)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df["v_Vel"].iloc[t1_idx:t2_idx], label="Before Reconstruction", color=color_before)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_CvxOpt["v_Vel"].iloc[t1_idx:t2_idx], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_PIButterworth["v_Vel"].iloc[t1_idx:t2_idx], label="PI-Butterworth Reconstruction", linestyle="-.", alpha=0.75, color=color_pibutter)
subax.grid()
subax.set_xlim([t1, t2])

axs[0][1].set_title("(a) Trajectory Variables", fontweight="bold")
axs[0][1].plot(vehicle_df["Global_Time"], vehicle_df["Space_Hdwy"], label="Before Reconstruction", color=color_before)
axs[0][1].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["Space_Hdwy"], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[0][1].plot(vehicle_df_PIButterworth["Global_Time"], vehicle_df_PIButterworth["Space_Hdwy"], label="PI-Butterworth Reconstruction", linestyle="-.", alpha=0.75, color=color_pibutter)
axs[0][1].set_xlabel("Time [s]")
axs[0][1].set_ylabel("Space Headway [m]")
axs[0][1].grid()
#axs[0][1].set_xlim(0,150)
#axs[0][1].set_ylim(7.5,25)
axs[0][1].yaxis.set_major_locator(plt.MultipleLocator(4))

axs[0][2].plot(vehicle_df["Global_Time"], vehicle_df["v_Accel"], label="Before Reconstruction", color=color_before)
axs[0][2].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["v_Accel"], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[0][2].plot(vehicle_df_PIButterworth["Global_Time"], vehicle_df_PIButterworth["v_Accel"], label="PI-Butterworth Reconstruction", linestyle="-.", alpha=0.75, color=color_pibutter)
axs[0][2].set_xlabel("Time [s]")
axs[0][2].set_ylabel("Acceleration [m/s$^2$]")
axs[0][2].grid()
#axs[0][2].set_xlim(0,150)
axs[0][2].set_ylim(-5,10)

axs[1][0].set_title("                                                            (b) Lane-Coordinate Positions", fontweight="bold")
axs[1][0].plot(vehicle_df["Global_Time"], vehicle_df["Lane_X"]-vehicle_df["Lane_X"].iloc[0], label="Before Reconstruction", color=color_before)
axs[1][0].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["Lane_X"]-vehicle_df_CvxOpt["Lane_X"].iloc[0], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[1][0].plot(vehicle_df_PIButterworth["Global_Time"], vehicle_df_PIButterworth["Lane_X"]-vehicle_df_PIButterworth["Lane_X"].iloc[0], label="PI-Butterworth Reconstruction", linestyle="-.", alpha=0.75, color=color_pibutter)
axs[1][0].set_xlabel("Time [s]")
axs[1][0].set_ylabel("Position [m]")
axs[1][0].grid()
#axs[1][0].set_xlim(0,150)
#axs[1][0].set_ylim(0,500)
time_arr = vehicle_df["Global_Time"].to_numpy()
t1, t2 = 280, time_arr[-1]
t1_idx, t2_idx = np.argwhere(time_arr == t1)[0][0], np.argwhere(time_arr == t2)[0][0]
d1 = np.amin([vehicle_df["Lane_X"].iloc[t1_idx], vehicle_df_CvxOpt["Lane_X"].iloc[t1_idx], vehicle_df_PIButterworth["Lane_X"].iloc[t1_idx]]) - vehicle_df["Lane_X"].iloc[0]
d2 = np.amax([vehicle_df["Lane_X"].iloc[t2_idx], vehicle_df_CvxOpt["Lane_X"].iloc[t2_idx], vehicle_df_PIButterworth["Lane_X"].iloc[t2_idx]]) - vehicle_df["Lane_X"].iloc[0]
axs[1][0].add_patch(
    Rectangle(xy=(t1, d1), width=t2-t1, height=d2-d1, edgecolor = 'black', fill=False, lw=1)
)
subpos = [0.51, 0.08, 0.36, 0.36]
subax = add_subplot_axes(axs[1][0], subpos)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df["Lane_X"].iloc[t1_idx:t2_idx]-vehicle_df["Lane_X"].iloc[0], label="Before Reconstruction", color=color_before)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_CvxOpt["Lane_X"].iloc[t1_idx:t2_idx]-vehicle_df_CvxOpt["Lane_X"].iloc[0], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_PIButterworth["Lane_X"].iloc[t1_idx:t2_idx]-vehicle_df_PIButterworth["Lane_X"].iloc[0], label="PI-Butterworth Reconstruction", linestyle="-.", alpha=0.75, color=color_pibutter)
subax.grid()
subax.set_xlim([t1, t2])

err_lanex_cvxopt = abs(vehicle_df_CvxOpt["Lane_X"].to_numpy() - vehicle_df["Lane_X"].to_numpy())
err_lanex_pibutterworth = abs(vehicle_df_PIButterworth["Lane_X"].to_numpy() - vehicle_df["Lane_X"].to_numpy())
axs[1][1].plot(vehicle_df_CvxOpt["Global_Time"], err_lanex_cvxopt, label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[1][1].plot(vehicle_df_PIButterworth["Global_Time"], err_lanex_pibutterworth, label="PI-Butterworth Reconstruction", linestyle="-.", alpha=0.75, color=color_pibutter)
axs[1][1].set_xlabel("Time [s]")
axs[1][1].set_ylabel("Absolute Position Error [m]")
axs[1][1].grid()
#axs[1][1].set_xlim(0,150)
#axs[1][1].set_ylim(0,6)

v_min = min(vehicle_df_CvxOpt["v_Vel"].min(), vehicle_df["v_Vel"].min())
v_max = max(vehicle_df_CvxOpt["v_Vel"].max(), vehicle_df["v_Vel"].max())
speeds = np.linspace(v_min, v_max, 100)
accel_max = accel_max_spl(speeds)
accel_min = decel_min_spl(speeds)
axs[1][2].set_title("(c) Speed-Acceleration Curve", fontweight="bold")
axs[1][2].scatter(vehicle_df["v_Vel"], vehicle_df["v_Accel"], alpha=0.75, color=color_before)
axs[1][2].scatter(vehicle_df_CvxOpt["v_Vel"], vehicle_df_CvxOpt["v_Accel"], linestyle="--", alpha=0.75, color=color_proposed)
axs[1][2].scatter(vehicle_df_PIButterworth["v_Vel"], vehicle_df_PIButterworth["v_Accel"], linestyle="-.", alpha=0.75, color=color_pibutter)
axs[1][2].plot(speeds, accel_max, label="Vehicle Acceleration Capacity", color="black", linestyle="--")
axs[1][2].plot(speeds, 0.5*accel_max, label="Driver Acceleration Capacity", color="black")
axs[1][2].plot(speeds, accel_min, label="Vehicle Deceleration Capacity", color="red")
axs[1][2].grid()
axs[1][2].set_xlabel("Speed [m/s]")
axs[1][2].set_ylabel("Acceleration [m/s$^2$]")
axs[1][2].legend(fontsize="x-small", loc="upper left")
axs[1][2].set_ylim(-6, 14)
axs[1][2].set_xlim(v_min, v_max)
axs[1][2].yaxis.set_major_locator(plt.MultipleLocator(2))

fig.tight_layout()
plt.show()
#fig.savefig(f"./ReconstructionComparison_{RELEVANT_VIDEO}_{vehicle_id}.pdf", bbox_inches='tight', dpi=100)
sys.exit(1)


###################################################################################
# Main: Figure 6
###################################################################################
maxAbsErr_CvxOpt = [
    0.3434, 0.8784, 0.3176, 0.3244, 0.3038, 0.3861, 0.3369, 0.3376, 0.3574, 0.3794, 0.4621, 0.3848, 0.4310, 0.6722,
]
meanAbsErr_CvxOpt = [
    0.0748, 0.0634, 0.0639, 0.0724, 0.0573, 0.0621, 0.0573, 0.0533, 0.0525, 0.0666, 0.0506, 0.0550, 0.0573, 0.0679,
]

maxAbsErr_PIButterworth = [
    10.0965, 4.7498, 17.2330, 4.7039, 2.4352, 9.4504, 14.4192, 4.6432, 9.8411, 5.0093, 13.9010, 7.5365, 6.1789, 11.3832,
]
meanAbsErr_PIButterworth = [
    4.7032, 2.0734, 8.4069, 2.9474, 0.8117, 4.3275, 6.8550, 2.3106, 4.5463, 2.2924, 5.8906, 2.2039, 2.8186, 5.8588,
]

unique_vehicle_ids = [f"V_{i}" for i in range(1, 15)]
fig = plt.figure(figsize=(12, 3), dpi=100)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
df1 = pd.DataFrame(np.asarray([maxAbsErr_CvxOpt, maxAbsErr_PIButterworth]).transpose(), columns=["Proposed Reconstruction", "PI-Butterworth Reconstruction"])
ax1 = setup_broken_axis(gs[0], "Max. Absolute Position Error [m]", df1, (1, 20), (0, 1), xlabels=unique_vehicle_ids)
ax1.set_ylabel("")
df2 = pd.DataFrame(np.asarray([meanAbsErr_CvxOpt, meanAbsErr_PIButterworth]).transpose(), columns=["Proposed Reconstruction", "PI-Butterworth Reconstruction"])
ax2 = setup_broken_axis(gs[1], "Mean Absolute Position Error [m]", df2, (0.1, 10), (0, 0.1), xlabels=unique_vehicle_ids)
ax2.set_ylabel("")
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -0.05), ncol=2)
fig.tight_layout()
#fig.savefig(f"./ReconstructionComparison_{RELEVANT_VIDEO}_LaneXErrors.pdf", bbox_inches='tight', dpi=100)

plt.show()


###################################################################################
# Main: Figure 8
###################################################################################
from tools_trajectory_plotting import plot_oblique_trajectories

fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=100)

axs[0] = plot_oblique_trajectories(df, start_frame=0, end_frame=None, vehicle_labels_in_plot=False, axis=axs[0])
handles, labels = axs[0].get_legend_handles_labels()
axs[0].get_legend().remove()
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Oblique Lane-Coordinate Position")
axs[0].set_title("(a) Before Reconstruction", fontweight="bold")

axs[1] = plot_oblique_trajectories(df_CvxOpt, start_frame=0, end_frame=None, vehicle_labels_in_plot=False, axis=axs[1])
axs[1].get_legend().remove()
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("")
axs[1].set_title("(b) Proposed Reconstruction", fontweight="bold")

axs[2] = plot_oblique_trajectories(df_PIButterworth, start_frame=0, end_frame=None, vehicle_labels_in_plot=False, axis=axs[2])
axs[2].get_legend().remove()
axs[2].set_xlabel("Time [s]")
axs[2].set_ylabel("")
axs[2].set_title("(c) PI-Butterworth Reconstruction", fontweight="bold")

#labels = [f"V_{l}" for l in labels]
#fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -0.05), ncol=14, fontsize="small")
fig.tight_layout()
#fig.savefig(f"./ReconstructionComparison_{RELEVANT_VIDEO}_ObliqueTrajectories.pdf", bbox_inches='tight', dpi=100)
plt.show()