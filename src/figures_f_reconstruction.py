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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


###############################################################################
###############################################################################
########### ALL EVALUATION DONE ON DJI_0933.MOV ###############################
###############################################################################
###############################################################################


# Models
data_hbb_x = [ "CFA", "REDET", "ROI_TRANS", "S2A", 
                "RetinaNet",  "RCNN", 
                "Yolo N", "Yolo S", "Yolo M", "Yolo L", "Yolo X"]
data_obb_x = [  "CFA", "REDET", "ROI_TRANS", "S2A", 
                "RetinaNet",  "RCNN", 
                "Yolo N", "Yolo S", "Yolo M", "Yolo L", "Yolo X"]

# =========== UNPROCESSED ===========
# %     # number of annotations
data_hbb_y_rlx1 = [  5.61, 2.92, 2.88, 2.6,
                        2.41, 375.36,
                        0, 0, 9.78, 0, 2.67]
data_obb_y_rlx1 = [  5.59, 2.94, 2.86, 2.58,
                        2.440, 0,
                        204.92, 244.56, 9.93, 0, 2.65]

# %     # avg confidence of annotation
data_hbb_y_rlx2 = [  7.07, 0.81, 0.34, 0.14,
                        0.48, 471.13,
                        0, 0, 10.32, 0, 0.36]
data_obb_y_rlx2 = [  7.24, 0.51, 0.44, 0.24,
                        0.48, 0,
                        249.11, 313.73, 10.35, 0, 0.35]





plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
fig = plt.figure(figsize=(12, 3), dpi=100)

# Create GridSpec
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# Function to set up broken axis
def setup_broken_axis(gs_cell, title, data, ylim_top, ylim_bottom):
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell, height_ratios=[1, 3], hspace=0.05)
    ax_top = fig.add_subplot(gs_sub[0])
    ax_bottom = fig.add_subplot(gs_sub[1])
    
    plt.sca(ax_top)
    plt.title(title)
    data.plot.bar(ax=ax_top, edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
    ax_top.set_ylim(ylim_top)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.grid(zorder=0)
    
    data.plot.bar(ax=ax_bottom, edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
    ax_bottom.set_ylim(ylim_bottom)
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.grid(zorder=0)
    ax_bottom.set_xticklabels(data_obb_x, rotation=45, ha='right')
    
    # Add diagonal lines
    d = .015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    return ax_bottom

# Subplot 1: Relaxation Variable 1
df1 = pd.DataFrame(np.asarray([data_hbb_y_rlx1, data_obb_y_rlx1]).transpose(), columns=["HBB", "OBB"])
ax1 = setup_broken_axis(gs[0], "Relaxation Variable 1", df1, (11, 500), (0, 11))
ax1.set_ylabel("")

# Subplot 2: Relaxation Variable 2
df2 = pd.DataFrame(np.asarray([data_hbb_y_rlx2, data_obb_y_rlx2]).transpose(), columns=["HBB", "OBB"])
ax2 = setup_broken_axis(gs[1], "Relaxation Variable 2", df2, (11, 500), (0, 11))
ax2.set_ylabel("")

plt.tight_layout()
plt.show()