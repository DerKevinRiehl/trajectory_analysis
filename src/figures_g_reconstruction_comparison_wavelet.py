"""
Consistent Vehicle Trajectory Extraction From Aerial Recordings Using Oriented Object Detection
-------------------------------------------
Authors:        Kevin Riehl, Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024 - 2025
Submitted to:   Scientific Reports
-------------------------------------------
"""

# Imports
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


def setup_broken_axis_barplot(gs_cell, title, data, ylim_top, ylim_bottom, xlabels, colors):
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell, height_ratios=[1, 3], hspace=0.05)
    ax_top = fig.add_subplot(gs_sub[0])
    ax_bottom = fig.add_subplot(gs_sub[1])
    
    plt.sca(ax_top)
    plt.title(title)
    data.plot.bar(ax=ax_top, edgecolor="black", color=colors, width=0.75, legend=False, zorder=3)
    ax_top.set_ylim(ylim_top)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.grid(zorder=0)
    
    data.plot.bar(ax=ax_bottom, edgecolor="black", color=colors, width=0.75, legend=False, zorder=3)
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


def setup_broken_axis_boxplot(gs_cell, title, df_Pc, df_info, ylim_top, ylim_bottom, xlabels, colors):
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell, height_ratios=[1, 3], hspace=0.05)
    ax_top = fig.add_subplot(gs_sub[0])
    ax_bottom = fig.add_subplot(gs_sub[1])

    plt.sca(ax_top)
    plt.title(title)
    plot_df = df_Pc[(df_Pc['Pt'] >= 1e-01) & (df_Pc['Pt'] - df_Pc['Pmax'] >= 1e-02)]
    extra_df = []
    for v_id in plot_df['Vehicle_ID'].unique():
        extra_df.append([v_id, np.nan, np.nan, "Proposed Reconstruction"])
    extra_df = pd.DataFrame(extra_df, columns=['Vehicle_ID', 'Pt', 'Pmax', 'Source'])
    plot_df = pd.concat((plot_df, extra_df)).reset_index().drop(columns='index')
    plot_df['Source'] = pd.Categorical(plot_df['Source'], categories=colors.keys(), ordered=True)
    plot_df['Vehicle_ID'] = pd.Categorical(plot_df['Vehicle_ID'], categories=df_Pc['Vehicle_ID'].unique(), ordered=True)
    plot_df = plot_df.sort_values(by=['Source', 'Vehicle_ID'])
    sns.stripplot(data=plot_df, x='Vehicle_ID', y='Pmax', hue='Source', ax=ax_top, 
                  palette=colors, dodge=True, jitter=False, legend=False, marker='o', size=5)
    sns.stripplot(data=plot_df, x='Vehicle_ID', y='Pmax', hue='Source', ax=ax_bottom, 
                    palette=colors, dodge=True, jitter=False, legend=False, marker='o', size=5)
    
    plot_df = df_Pc[df_Pc['Pt'] > 1e-02]
    sns.boxplot(data=plot_df, x='Vehicle_ID', y='Pt', hue='Source', ax=ax_top, showfliers=False, legend=False, 
                palette=colors, flierprops=dict(alpha=0.05), showmeans=True, meanprops=dict(markerfacecolor='white', markeredgecolor='black'), boxprops=dict(alpha=0.6))
    sns.scatterplot(data=df_info, x='Vehicle_ID', y='Max_Power_KW', ax=ax_top, c='red', marker='D', s=50, linewidth=2, legend=False)
    ax_top.set(xlabel=None, ylabel=None)
    ax_top.set_ylim(ylim_top)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.grid(zorder=0)
    
    sns.boxplot(data=plot_df, x='Vehicle_ID', y='Pt', hue='Source', ax=ax_bottom, showfliers=False, legend=False, 
                palette=colors, flierprops=dict(alpha=0.05), showmeans=True, meanprops=dict(markerfacecolor='white', markeredgecolor='black'), boxprops=dict(alpha=0.6))
    sns.scatterplot(data=df_info, x='Vehicle_ID', y='Max_Power_KW', ax=ax_bottom, c='red', marker='D', s=50, linewidth=2, legend=False)
    ax_bottom.set(xlabel=None, ylabel=None)
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
df = plot_accelerations(df, recompute_speeds=True)
plt.close()
df_info = pd.read_csv(VEHICLE_INFO_PATH + RELEVANT_VIDEO + ".txt", sep="\t")
df_CvxOpt = pd.read_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_CvxOptNoRelax.txt", sep=",")
df_Butterworth = pd.read_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_Butterworth.txt", sep=",")
df_Wavelet = pd.read_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_Wavelet.txt", sep=",")


vehicle_id = "VEHICLE_1"
vehicle_df = df[df["Vehicle_ID"] == vehicle_id]
if not vehicle_df["Frame_ID"].is_monotonic_increasing:
    vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
vehicle_df = vehicle_df.reset_index().drop(columns=["index"])

vehicle_df_CvxOpt = df_CvxOpt[df_CvxOpt["Vehicle_ID"] == vehicle_id]
if not vehicle_df_CvxOpt["Frame_ID"].is_monotonic_increasing:
    vehicle_df_CvxOpt = vehicle_df_CvxOpt.sort_values(by=["Frame_ID"], ascending=True)
vehicle_df_CvxOpt = vehicle_df_CvxOpt.reset_index().drop(columns=["index"])

vehicle_df_Wavelet = df_Wavelet[df_Wavelet["Vehicle_ID"] == vehicle_id]
if not vehicle_df_Wavelet["Frame_ID"].is_monotonic_increasing:
    vehicle_df_Wavelet = vehicle_df_Wavelet.sort_values(by=["Frame_ID"], ascending=True)
vehicle_df_Wavelet = vehicle_df_Wavelet.reset_index().drop(columns=["index"])

vehicle_df_Butterworth = df_Butterworth[df_Butterworth["Vehicle_ID"] == vehicle_id]
if not vehicle_df_Butterworth["Frame_ID"].is_monotonic_increasing:
    vehicle_df_Butterworth = vehicle_df_Butterworth.sort_values(by=["Frame_ID"], ascending=True)
vehicle_df_Butterworth = vehicle_df_Butterworth.reset_index().drop(columns=["index"])

mfc_car_id = df_info.loc[df_info["Vehicle_ID"] == vehicle_id, "MFC_CarID"].item()
with open(VEHICLE_INFO_PATH + f"ID{mfc_car_id}_AccelCapInterp.pkl", 'rb') as f:
    accel_max_spl = pickle.load(f)
with open(VEHICLE_INFO_PATH + f"ID{mfc_car_id}_DecelCapInterp.pkl", 'rb') as f:
    decel_min_spl = pickle.load(f)

# Plot
color_before = "burlywood"
color_proposed = "blueviolet"
color_butter = "black"
color_wavelet = "forestgreen"

fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=100)
axs[0][0].plot(vehicle_df["Global_Time"], vehicle_df["v_Vel"], label="Before Reconstruction", color=color_before)
axs[0][0].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["v_Vel"], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[0][0].plot(vehicle_df_Butterworth["Global_Time"], vehicle_df_Butterworth["v_Vel"], label="Butterworth Reconstruction", linestyle=":", alpha=0.75, color=color_butter)
axs[0][0].plot(vehicle_df_Wavelet["Global_Time"], vehicle_df_Wavelet["v_Vel"], label="Wavelet-Based Reconstruction", linestyle="-.", alpha=0.75, color=color_wavelet)
axs[0][0].set_xlabel("Time [s]")
axs[0][0].set_ylabel("Speed [m/s]")
axs[0][0].grid()
#axs[0][0].set_xlim(0, 150)
axs[0][0].set_ylim(-2, 7.5)
time_arr = vehicle_df["Global_Time"].to_numpy()
t1, t2 = 230, 240
t1_idx, t2_idx = np.argwhere(time_arr == t1)[0][0], np.argwhere(time_arr == t2)[0][0]
v1 = np.amin([vehicle_df["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_CvxOpt["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_Wavelet["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_Butterworth["v_Vel"].iloc[t1_idx:t2_idx]])-0.15
v2 = np.amax([vehicle_df["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_CvxOpt["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_Wavelet["v_Vel"].iloc[t1_idx:t2_idx], vehicle_df_Butterworth["v_Vel"].iloc[t1_idx:t2_idx]])+0.15
axs[0][0].add_patch(
    Rectangle(xy=(t1, v1), width=t2-t1, height=v2-v1, edgecolor = 'black', fill=False, lw=1)
)
subpos = [0.42, 0.24, 0.35, 0.35]
subax = add_subplot_axes(axs[0][0], subpos)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df["v_Vel"].iloc[t1_idx:t2_idx], label="Before Reconstruction", color=color_before)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_CvxOpt["v_Vel"].iloc[t1_idx:t2_idx], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_Butterworth["v_Vel"].iloc[t1_idx:t2_idx], label="Butterworth Reconstruction", linestyle=":", alpha=0.75, color=color_butter)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_Wavelet["v_Vel"].iloc[t1_idx:t2_idx], label="Wavelet-Based Reconstruction", linestyle="-.", alpha=0.75, color=color_wavelet)
subax.grid()
subax.set_xlim([t1, t2])

axs[0][1].set_title("(a) Trajectory Variables", fontweight="bold")
axs[0][1].plot(vehicle_df["Global_Time"], vehicle_df["Space_Hdwy"], label="Before Reconstruction", color=color_before)
axs[0][1].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["Space_Hdwy"], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[0][1].plot(vehicle_df_Butterworth["Global_Time"], vehicle_df_Butterworth["Space_Hdwy"], label="Butterworth Reconstruction", linestyle=":", alpha=0.75, color=color_butter)
axs[0][1].plot(vehicle_df_Wavelet["Global_Time"], vehicle_df_Wavelet["Space_Hdwy"], label="Wavelet-Based Reconstruction", linestyle="-.", alpha=0.75, color=color_wavelet)
axs[0][1].set_xlabel("Time [s]")
axs[0][1].set_ylabel("Space Headway [m]")
axs[0][1].grid()
#axs[0][1].set_xlim(0,150)
#axs[0][1].set_ylim(7.5,25)
axs[0][1].legend(fontsize="x-small", loc="upper left")
axs[0][1].yaxis.set_major_locator(plt.MultipleLocator(4))

axs[0][2].plot(vehicle_df["Global_Time"], vehicle_df["v_Accel"], label="Before Reconstruction", color=color_before)
axs[0][2].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["v_Accel"], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[0][2].plot(vehicle_df_Butterworth["Global_Time"], vehicle_df_Butterworth["v_Accel"], label="Butterworth Reconstruction", linestyle=":", alpha=0.75, color=color_butter)
axs[0][2].plot(vehicle_df_Wavelet["Global_Time"], vehicle_df_Wavelet["v_Accel"], label="Wavelet-Based Reconstruction", linestyle="-.", alpha=0.75, color=color_wavelet)
axs[0][2].set_xlabel("Time [s]")
axs[0][2].set_ylabel("Acceleration [m/s$^2$]")
axs[0][2].grid()
#axs[0][2].set_xlim(0,150)
axs[0][2].set_ylim(-4,6)

axs[1][0].set_title("                                                            (b) Lane-Coordinate Positions", fontweight="bold")
axs[1][0].plot(vehicle_df["Global_Time"], vehicle_df["Lane_X"]-vehicle_df["Lane_X"].iloc[0], label="Before Reconstruction", color=color_before)
axs[1][0].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["Lane_X"]-vehicle_df_CvxOpt["Lane_X"].iloc[0], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[1][0].plot(vehicle_df_Butterworth["Global_Time"], vehicle_df_Butterworth["Lane_X"]-vehicle_df_Butterworth["Lane_X"].iloc[0], label="Butterworth Reconstruction", linestyle=":", alpha=0.75, color=color_butter)
axs[1][0].plot(vehicle_df_Wavelet["Global_Time"], vehicle_df_Wavelet["Lane_X"]-vehicle_df_Wavelet["Lane_X"].iloc[0], label="Wavelet-Based Reconstruction", linestyle="-.", alpha=0.75, color=color_wavelet)
axs[1][0].set_xlabel("Time [s]")
axs[1][0].set_ylabel("Position [m]")
axs[1][0].grid()
#axs[1][0].set_xlim(0,150)
#axs[1][0].set_ylim(0,500)
time_arr = vehicle_df["Global_Time"].to_numpy()
t1, t2 = 280, time_arr[-1]
t1_idx, t2_idx = np.argwhere(time_arr == t1)[0][0], np.argwhere(time_arr == t2)[0][0]
d1 = np.amin([vehicle_df["Lane_X"].iloc[t1_idx], vehicle_df_CvxOpt["Lane_X"].iloc[t1_idx], vehicle_df_Wavelet["Lane_X"].iloc[t1_idx], vehicle_df_Butterworth["Lane_X"].iloc[t1_idx]]) - vehicle_df["Lane_X"].iloc[0]
d2 = np.amax([vehicle_df["Lane_X"].iloc[t2_idx], vehicle_df_CvxOpt["Lane_X"].iloc[t2_idx], vehicle_df_Wavelet["Lane_X"].iloc[t2_idx], vehicle_df_Butterworth["Lane_X"].iloc[t2_idx]]) - vehicle_df["Lane_X"].iloc[0]
axs[1][0].add_patch(
    Rectangle(xy=(t1, d1), width=t2-t1, height=d2-d1, edgecolor = 'black', fill=False, lw=1)
)
subpos = [0.51, 0.08, 0.36, 0.36]
subax = add_subplot_axes(axs[1][0], subpos)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df["Lane_X"].iloc[t1_idx:t2_idx]-vehicle_df["Lane_X"].iloc[0], label="Before Reconstruction", color=color_before)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_CvxOpt["Lane_X"].iloc[t1_idx:t2_idx]-vehicle_df_CvxOpt["Lane_X"].iloc[0], label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_Butterworth["Lane_X"].iloc[t1_idx:t2_idx]-vehicle_df_Butterworth["Lane_X"].iloc[0], label="Butterworth Reconstruction", linestyle=":", alpha=0.75, color=color_butter)
subax.plot(time_arr[t1_idx:t2_idx], vehicle_df_Wavelet["Lane_X"].iloc[t1_idx:t2_idx]-vehicle_df_Wavelet["Lane_X"].iloc[0], label="Wavelet-Based Reconstruction", linestyle="-.", alpha=0.75, color=color_wavelet)
subax.grid()
subax.set_xlim([t1, t2])

err_lanex_cvxopt = abs(vehicle_df_CvxOpt["Lane_X"].to_numpy() - vehicle_df["Lane_X"].to_numpy())
err_lanex_pibutter = abs(vehicle_df_Butterworth["Lane_X"].to_numpy() - vehicle_df["Lane_X"].to_numpy())
err_lanex_wavelet = abs(vehicle_df_Wavelet["Lane_X"].to_numpy() - vehicle_df["Lane_X"].to_numpy())
axs[1][1].plot(vehicle_df_CvxOpt["Global_Time"], err_lanex_cvxopt, label="Proposed Reconstruction", linestyle="--", alpha=0.75, color=color_proposed)
axs[1][1].plot(vehicle_df_Butterworth["Global_Time"], err_lanex_pibutter, label="Butterworth Reconstruction", linestyle=":", alpha=0.75, color=color_butter)
axs[1][1].plot(vehicle_df_Wavelet["Global_Time"], err_lanex_wavelet, label="Wavelet-Based Reconstruction", linestyle="-.", alpha=0.75, color=color_wavelet)
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
axs[1][2].scatter(vehicle_df_Butterworth["v_Vel"], vehicle_df_Butterworth["v_Accel"], linestyle=":", alpha=0.75, color=color_butter)
axs[1][2].scatter(vehicle_df_Wavelet["v_Vel"], vehicle_df_Wavelet["v_Accel"], linestyle="-.", alpha=0.75, color=color_wavelet)
axs[1][2].plot(speeds, accel_max, label="Vehicle Acceleration Capacity", color="blue", linestyle="--")
axs[1][2].plot(speeds, 0.5*accel_max, label="Driver Acceleration Capacity", color="blue")
axs[1][2].plot(speeds, accel_min, label="Vehicle Deceleration Capacity", color="red")
axs[1][2].grid()
axs[1][2].set_xlabel("Speed [m/s]")
axs[1][2].set_ylabel("Acceleration [m/s$^2$]")
axs[1][2].legend(fontsize="x-small", loc="upper left")
axs[1][2].set_ylim(-6, 14)
axs[1][2].set_xlim(v_min, v_max)
axs[1][2].yaxis.set_major_locator(plt.MultipleLocator(2))

fig.tight_layout()
#plt.show()
#fig.savefig(f"./ReconstructionComparison_{RELEVANT_VIDEO}_{vehicle_id}.pdf", bbox_inches='tight', dpi=100)
#sys.exit(1)

###################################################################################
# Main: Figure 6
###################################################################################
color_butter = "gray"
maxAbsErr_CvxOpt = [
    0.3434, 0.8784, 0.3176, 0.3244, 0.3038, 0.3861, 0.3369, 0.3376, 0.3574, 0.3794, 0.4621, 0.3848, 0.4310, 0.6722,
]
meanAbsErr_CvxOpt = [
    0.0748, 0.0634, 0.0639, 0.0724, 0.0573, 0.0621, 0.0573, 0.0533, 0.0525, 0.0666, 0.0506, 0.0550, 0.0573, 0.0679,
]

maxAbsErr_Butterworth = [
    10.0954, 4.7606, 17.2423, 4.719, 2.4304, 9.4722, 14.4385, 4.6368, 9.832, 5.022, 13.8849, 7.5431, 6.1834, 11.3887
]
meanAbsErr_Butterworth = [
    4.7025, 2.0734, 8.4061, 2.9472, 0.812, 4.3274, 6.8546, 2.3104, 4.5459, 2.2924, 5.8896, 2.2028, 2.8183, 5.8583
]

maxAbsErr_Wavelet = [
    0.1259, 0.5627, 0.2092, 0.1683, 0.2219, 0.2241, 0.1163, 0.3033, 0.1696, 0.2411, 0.3233, 0.2694, 0.2783, 0.3738,
]
meanAbsErr_Wavelet = [
    0.0258, 0.0270, 0.0287, 0.0300, 0.0249, 0.0230, 0.0227, 0.0242, 0.0274, 0.0435, 0.0268, 0.0303, 0.0275, 0.0224
]

unique_vehicle_ids = [f"V_{i}" for i in range(1, 15)]
fig = plt.figure(figsize=(12, 3), dpi=100)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
df1 = pd.DataFrame(np.asarray([maxAbsErr_CvxOpt, maxAbsErr_Butterworth, maxAbsErr_Wavelet]).transpose(), columns=["Proposed Reconstruction", "Butterworth Reconstruction", "Wavelet-Based Reconstruction"])
ax1 = setup_broken_axis_barplot(
    gs[0], "Max. Absolute Position Error [m]", df1, (1, 20), (0, 1), xlabels=unique_vehicle_ids, colors=[color_proposed, color_butter, color_wavelet] #["forestgreen","aquamarine","darkseagreen"]
)
ax1.set_ylabel("")
df2 = pd.DataFrame(np.asarray([meanAbsErr_CvxOpt, meanAbsErr_Butterworth, meanAbsErr_Wavelet]).transpose(), columns=["Proposed Reconstruction", "Butterworth Reconstruction", "Wavelet-Based Reconstruction"])
ax2 = setup_broken_axis_barplot(
    gs[1], "Mean Absolute Position Error [m]", df2, (0.1, 10), (0, 0.1), xlabels=unique_vehicle_ids, colors=[color_proposed, color_butter, color_wavelet] #["forestgreen","aquamarine","darkseagreen"]
)
ax2.set_ylabel("")
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -0.05), ncol=3)
fig.tight_layout()
#fig.savefig(f"./ReconstructionComparison_{RELEVANT_VIDEO}_LaneXErrors.pdf", bbox_inches='tight', dpi=100)

#plt.show()

###################################################################################
# Main: Figure 8
###################################################################################
from tools_trajectory_plotting import plot_oblique_trajectories

fig, axs = plt.subplots(2, 2, figsize=(12, 6), dpi=100)

axs[0, 0] = plot_oblique_trajectories(df, start_frame=0, end_frame=None, vehicle_labels_in_plot=False, axis=axs[0, 0])
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[0, 0].get_legend().remove()
axs[0, 0].set_xlabel("")
axs[0, 0].set_ylabel("Oblique Lane-Coordinate Position")
axs[0, 0].set_title("(a) Before Reconstruction", fontweight="bold")

axs[0, 1] = plot_oblique_trajectories(df_CvxOpt, start_frame=0, end_frame=None, vehicle_labels_in_plot=False, axis=axs[0, 1])
axs[0, 1].get_legend().remove()
axs[0, 1].set_xlabel("")
axs[0, 1].set_ylabel("")
axs[0, 1].set_title("(b) Proposed Reconstruction", fontweight="bold")

axs[1, 0] = plot_oblique_trajectories(df_Butterworth, start_frame=0, end_frame=None, vehicle_labels_in_plot=False, axis=axs[1, 0])
axs[1, 0].get_legend().remove()
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Oblique Lane-Coordinate Position")
axs[1, 0].set_title("(c) Butterworth Reconstruction", fontweight="bold")

axs[1, 1] = plot_oblique_trajectories(df_Wavelet, start_frame=0, end_frame=None, vehicle_labels_in_plot=False, axis=axs[1, 1])
axs[1, 1].get_legend().remove()
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("")
axs[1, 1].set_title("(d) Wavelet-Based Reconstruction", fontweight="bold")

#labels = [f"V_{l}" for l in labels]
#fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -0.05), ncol=14, fontsize="small")
fig.tight_layout()
#fig.savefig(f"./ReconstructionComparison_{RELEVANT_VIDEO}_ObliqueTrajectories.pdf", bbox_inches='tight', dpi=100)
#plt.show()


###################################################################################
# Main: Figure 10
###################################################################################
from tools_trajectory_evaluation import calculateEnergyConsumption

electric_vehicles = df_info.loc[df_info['Powertrain']=='Electric', 'Vehicle_ID'].unique()
electric_vehicles_num = [int(v.split('_')[1]) for v in electric_vehicles]
print(electric_vehicles)

df_Ec, df = calculateEnergyConsumption(df, df_info)
df_Ec['Source'] = "Before Reconstruction"
energy_df, df_CvxOpt = calculateEnergyConsumption(df_CvxOpt, df_info)
energy_df['Source'] = "Proposed Reconstruction"
df_Ec = pd.concat((df_Ec, energy_df))
energy_df, df_Butterworth = calculateEnergyConsumption(df_Butterworth, df_info)
energy_df['Source'] = "Butterworth Reconstruction"
df_Ec = pd.concat((df_Ec, energy_df))
energy_df, df_Wavelet = calculateEnergyConsumption(df_Wavelet, df_info)
energy_df['Source'] = "Wavelet-Based Reconstruction"
df_Ec = pd.concat((df_Ec, energy_df))
del energy_df
df_Ec = df_Ec[~df_Ec['Vehicle_ID'].isin(electric_vehicles)]
df_Ec[["vehicleMeaningless","Vehicle_Num"]] = df_Ec["Vehicle_ID"].str.split("_", n=1, expand=True)
df_Ec["Vehicle_Num"] = df_Ec["Vehicle_Num"].astype(int)
df_Ec = df_Ec.sort_values(by=["Vehicle_Num"]).reset_index().drop(columns='index')
df_Ec = df_Ec.drop(columns=['vehicleMeaningless', 'Vehicle_ID'])
df_Ec = df_Ec.groupby(by=['Vehicle_Num', 'Source']).mean().unstack().rename_axis(None).rename_axis([None, None], axis=1).droplevel(0, axis=1)
df_Ec = df_Ec[['Before Reconstruction', 'Proposed Reconstruction', 'Butterworth Reconstruction', 'Wavelet-Based Reconstruction']]

df_Pc = df[['Vehicle_ID', 'Pt', 'Pmax']].copy()
df_Pc['Source'] = "Before Reconstruction"
power_df = df_CvxOpt[['Vehicle_ID', 'Pt', 'Pmax']].copy()
power_df['Source'] = "Proposed Reconstruction"
df_Pc = pd.concat((df_Pc, power_df))
power_df = df_Butterworth[['Vehicle_ID', 'Pt', 'Pmax']].copy()
power_df['Source'] = "Butterworth Reconstruction"
df_Pc = pd.concat((df_Pc, power_df))
power_df = df_Wavelet[['Vehicle_ID', 'Pt', 'Pmax']].copy()
power_df['Source'] = "Wavelet-Based Reconstruction"
df_Pc = pd.concat((df_Pc, power_df))
del power_df
df_Pc = df_Pc[~df_Pc['Vehicle_ID'].isin(electric_vehicles)]

unique_vehicle_ids = [f"V_{i}" for i in range(1, 15) if i not in electric_vehicles_num]
fig = plt.figure(figsize=(12, 3), dpi=100)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
ax1 = setup_broken_axis_barplot(
    gs[0], "", df_Ec, (10, 65), (0, 10), xlabels=unique_vehicle_ids,
    colors={"Before Reconstruction": color_before, "Proposed Reconstruction": color_proposed, "Butterworth Reconstruction": color_butter, "Wavelet-Based Reconstruction": color_wavelet}
)
ax1.set_ylabel("$E_c$ [kWh/100km]")
ax2 = setup_broken_axis_boxplot(
    gs[1], "", df_Pc, df_info[~df_info['Vehicle_ID'].isin(electric_vehicles)], (25, 225), (-1, 25), xlabels=unique_vehicle_ids, 
    colors={"Before Reconstruction": color_before, "Proposed Reconstruction": color_proposed, "Butterworth Reconstruction": color_butter, "Wavelet-Based Reconstruction": color_wavelet}
)
ax2.set_ylabel("$P_t$ [kW]")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -0.05), ncol=4)
fig.tight_layout()
plt.show()

"""
fig, axs = plt.subplots(1, 2, figsize=(12, 3), dpi=100)
df_Ec.plot.bar(
    ax=axs[0], edgecolor="black", width=0.75, legend=False, zorder=3,
    color={"Before Reconstruction": color_before, "Proposed Reconstruction": color_proposed, "Butterworth Reconstruction": color_butter, "Wavelet-Based Reconstruction": color_wavelet}
)
axs[0].grid(zorder=0)
axs[0].set_xticklabels(unique_vehicle_ids, rotation=45, ha='right')
axs[0].set_ylabel("$E_c$ [kWh/100km]")

plot_df = df_Pc[df_Pc['Pt'] > 1e-02]
sns.boxplot(data=plot_df, x='Vehicle_ID', y='Pt', hue='Source', ax=axs[1], showfliers=False, legend=False, 
            palette={"Before Reconstruction": color_before, "Proposed Reconstruction": color_proposed, "Butterworth Reconstruction": color_butter, "Wavelet-Based Reconstruction": color_wavelet}, 
            flierprops=dict(alpha=0.05), showmeans=True, meanprops=dict(markerfacecolor='white', markeredgecolor='black'))
plot_df = plot_df[plot_df['Pt'] - plot_df['Pmax'] >= 1e-02]
sns.stripplot(data=plot_df, x='Vehicle_ID', y='Pmax', hue='Source', ax=axs[1], 
                palette={"Before Reconstruction": color_before, "Proposed Reconstruction": color_proposed, "Butterworth Reconstruction": color_butter, "Wavelet-Based Reconstruction": color_wavelet},
                dodge=True, legend=False, marker='o') #s=50, linewidth=2)
sns.scatterplot(data=df_info[~df_info['Vehicle_ID'].isin(electric_vehicles)], x='Vehicle_ID', y='Max_Power_KW', ax=axs[1], c='red', marker='D', s=50, linewidth=2, legend=False)
axs[1].set_xticklabels(unique_vehicle_ids, rotation=45, ha='right')
axs[1].set_ylabel("$P_t$ [kW]")
axs[1].set_yscale('log')
fig.tight_layout()
plt.show()
"""

sys.exit(1)