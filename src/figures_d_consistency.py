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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib.patches import Rectangle


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
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("Coordinate X [m]")
ax.set_ylabel("Coordinate Y [m]")
en = np.random.random()*2
ax.add_patch(
    Rectangle(xy=(-20, 22.5), width=5, height=5, edgecolor = 'black', fill=False, lw=1)
)
subpos = [-0.15, 0.45, 0.4, 0.4]
subax = add_subplot_axes(ax, subpos)
for vehID in df_obb["Vehicle_ID"].unique():
    df_vehicle = df_obb[df_obb["Vehicle_ID"]==vehID]
    n = 1000
    cart_x, cart_y = df_vehicle["Cartesian_X"].to_numpy()[0:n]+en, df_vehicle["Cartesian_Y"].to_numpy()[0:n]+en
    ax.plot(cart_x, cart_y)
    idxs = np.argwhere((cart_x >=-20) & (cart_x <= -15))
    if len(idxs) > 0:
        subax.plot(cart_x[idxs], cart_y[idxs])
subax.set_ylim([22.5, 27.5])
subax.grid()


plt.subplot(gs[1, 1])
plt.grid(zorder=0)
plt.xlabel("Time [s]")
plt.ylabel("Coordinate X [m]")
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
plt.show()

