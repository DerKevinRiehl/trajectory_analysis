# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec



data_obb_x = [ "CFA", "REDET", "ROI_TRANS", "S2A", 
                "RetinaNet",  "RCNN", 
                "Yolo N", "Yolo S", "Yolo M", "Yolo L", "Yolo X"]


int_cons_hbb_exl = [3.04194, 3.7234, 3.95143, 3.55138,
                  4.75584, #3.84232,
                  13.5118,           3.52314, ]

int_cons_obb_exl = [3.00461, 3.7071, 3.92376, 3.54148,
                  3.77445, #21.1894,
                  13.5133,           3.47693, ]

plt_cons_hbb_exl = [12.8165, 7.50931, 4.19663, 5.5009,
                  9.31187, #11.5932,
                  7.39326,           5.34387, ]

plt_cons_obb_exl = [11.747, 7.58478, 4.27343, 5.57772,
                  8.7466, #11.5928,
                  7.37106,            4.85116, ]






int_cons_hbb = [3.04194, 3.7234, 3.95143, 3.55138,
                  4.75584, 3.84232,
                  13.5118, 82.6114, 3.52314, 222.249, 208.613]

int_cons_obb = [3.00461, 3.7071, 3.92376, 3.54148,
                  3.77445, 21.1894,
                  13.5133, 82.3276, 3.47693, 208.613, 79.2266]

plt_cons_hbb = [12.8165, 7.50931, 4.19663, 5.5009,
                  9.31187, 11.5932,
                  7.39326, 14.226, 5.34387, 147.276, 19.4735]

plt_cons_obb = [11.747, 7.58478, 4.27343, 5.57772,
                  8.7466, 11.5928,
                  7.37106, 13.7542, 4.85116, 401.998, 14.4182]


folder = "../data_benchmark/6_final_trajectories/"
df_obb = pd.read_csv(folder+"Inference_s2anet_r50_fpn_fp16_1x_dota_le135_OBB.txt")



# PLOT
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
fig = plt.figure(figsize=(12, 5), dpi=100)
gs = gridspec.GridSpec(2, 3, width_ratios=[0.5, 0.5, 1])


plt.subplot(gs[0, 0])
plt.title("Cartesian Space")
plt.grid(zorder=0)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Coordinate X [m]")
plt.ylabel("Coordinate Y [m]")
df_vehicle = df_obb[df_obb["Vehicle_ID"]=="VEHICLE_9"]
plt.plot(df_vehicle["Cartesian_X"], df_vehicle["Cartesian_Y"], color="Grey")

plt.subplot(gs[0, 1])
plt.title("Lane Space")
plt.grid(zorder=0)
plt.xlabel("Time [s]")
plt.ylabel("Coordinate X [m]")
plt.plot(df_vehicle["Global_Time"], df_vehicle["Lane_X"], color="Grey")
plt.xlim(0,150)
plt.ylim(0, 500)

plt.subplot(gs[1, 0])
plt.grid(zorder=0)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Coordinate X [m]")
plt.ylabel("Coordinate Y [m]")
for vehID in df_obb["Vehicle_ID"].unique():
    df_vehicle = df_obb[df_obb["Vehicle_ID"]==vehID]
    n = 1000
    plt.plot(df_vehicle["Cartesian_X"].iloc[0:n]+np.random.random()*2, df_vehicle["Cartesian_Y"].iloc[0:n]+np.random.random()*2)

plt.subplot(gs[1, 1])
plt.grid(zorder=0)
plt.xlabel("Time [s]")
plt.ylabel("Lane Coordinate Y [m]")
for vehID in df_obb["Vehicle_ID"].unique():
    df_vehicle = df_obb[df_obb["Vehicle_ID"]==vehID]
    plt.plot(df_vehicle["Global_Time"], df_vehicle["Lane_X"])
plt.xlim(0,150)
plt.ylim(0, 500)

gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], height_ratios=[1, 3], hspace=0.05)
ax1 = fig.add_subplot(gs_right[0])
plt.title("Internal Consistency [m] (Single Vehicle)")
df = pd.DataFrame(np.asarray([int_cons_hbb, int_cons_obb]).transpose(), columns=["HBB", "OBB"])
df.plot.bar(ax=ax1, edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
ax1.set_ylim(10, df.values.max() + 10)  # Adjust as needed
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(labelbottom=False, bottom=False)
ax1.grid(zorder=0)
ax2 = fig.add_subplot(gs_right[1])
df.plot.bar(ax=ax2, edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
ax2.set_ylim(0, 5)  # Adjust as needed
ax2.spines['top'].set_visible(False)
ax2.grid(zorder=0)
ax2.set_xticklabels([])
d = .015  # size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
ax2.set_ylabel("Consistency [m]\n")


gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 2], height_ratios=[1, 3], hspace=0.05)
ax1 = fig.add_subplot(gs_right[0])
plt.title("Platoon Consistency [m] (All Vehicles)")
df = pd.DataFrame(np.asarray([plt_cons_hbb, plt_cons_obb]).transpose(), columns=["HBB", "OBB"])
df.plot.bar(ax=ax1, edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
ax1.set_ylim(10, df.values.max() + 10)  # Adjust as needed
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(labelbottom=False, bottom=False)
ax1.grid(zorder=0)
ax2 = fig.add_subplot(gs_right[1])
df.plot.bar(ax=ax2, edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
ax2.set_ylim(0, 10)  # Adjust as needed
ax2.spines['top'].set_visible(False)
ax2.grid(zorder=0)
ax2.set_xticklabels([])
d = .015  # size of diagonal lines
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
ax2.set_ylabel("Consistency [m]\n")
# ax2.set_xticklabels(data_obb_x)
# ax2.set_xticks(rotation=45)
ax2.set_xticklabels(data_obb_x, rotation=45, ha='right')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()


