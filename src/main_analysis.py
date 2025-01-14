"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish, Kevin Riehl
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tools_trajectory_filtering import reconstruct_trajectories_cvxopt
from tools_trajectory_filtering import apply_physics_informed_butterworth_filter

from tools_trajectory_plotting import plot_time_space_diagram
from tools_trajectory_plotting import plot_oblique_trajectories
from tools_trajectory_plotting import plot_velocities_and_headways
from tools_trajectory_plotting import plot_accelerations

from tools_stability_analysis import estimate_L2gain_CTHpolicy
from tools_stability_analysis import speed_standard_deviation, speed_dft

from _constants import default_plotting_settings

# #############################################################################
# CONSTANTS
# #############################################################################
DATA_ROOT = "C:/Users/selbaklish/Desktop/Python_Workspace/OOBB/analysis/data/6_final_trajectories/"
INFO_ROOT = "C:/Users/selbaklish/Desktop/Python_Workspace/OOBB/analysis/data/7_vehicle_information/"

ALL_VIDEOS = [
    "DJI_0933.MOV", "DJI_0934.MOV", #"DJI_0939.MOV", "DJI_0940.MOV", "DJI_0943.MOV", "DJI_0944.MOV"
]

RELEVANT_VIDEO = "DJI_0933.MOV"
# RELEVANT_VIDEO = "DJI_0934.MOV"
# RELEVANT_VIDEO = "DJI_0939.MOV"
# RELEVANT_VIDEO = "DJI_0940.MOV"
# RELEVANT_VIDEO = "DJI_0943.MOV"
# RELEVANT_VIDEO = "DJI_0944.MOV"


# #############################################################################
# 1. Filtering
# #############################################################################

with open('../data/7_vehicle_information/accel_capacity_interpolator.pkl', 'rb') as f:
    accel_max_spl = pickle.load(f)
with open('../data/7_vehicle_information/decel_capacity_interpolator.pkl', 'rb') as f:
    decel_min_spl = pickle.load(f)

df = pd.read_csv(DATA_ROOT + RELEVANT_VIDEO + ".txt", sep=",")
df = plot_accelerations(df)
plt.show()

target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+".txt"
df_filtered = pd.read_csv(target_output_file, sep=",")
plot_time_space_diagram(df_filtered, start_frame=200*25, end_frame=300*25, vehicle_labels_in_plot=True)
plot_velocities_and_headways(df_filtered)
plot_accelerations(df_filtered)
plot_oblique_trajectories(df_filtered, start_frame=200*25, end_frame=300*25, vehicle_labels_in_plot=True, active_cursor=False)
plt.show()

unique_vehicles = df["Vehicle_ID"].unique()
for vehicle_id in unique_vehicles:
    vehicle_df = df[df["Vehicle_ID"] == vehicle_id]
    vehicle_df_filt = df_filtered[df_filtered["Vehicle_ID"] == vehicle_id]

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(vehicle_df["Global_Time"], vehicle_df["v_Vel"], label="Before")
    axs[0, 0].plot(vehicle_df_filt["Global_Time"], vehicle_df_filt["v_Vel"], label="After", linestyle="--")
    axs[0, 0].set_ylabel("Speed (m/s)")
    axs[0, 0].legend()

    axs[0, 1].plot(vehicle_df["Global_Time"], vehicle_df["Space_Hdwy"], label="Before")
    axs[0, 1].plot(vehicle_df_filt["Global_Time"], vehicle_df_filt["Space_Hdwy"], label="After", linestyle="--")
    axs[0, 1].set_ylabel("Space_Headway (m)")
    axs[0, 1].legend()

    axs[1, 0].plot(vehicle_df["Global_Time"], vehicle_df["v_Accel"], label="Before")
    axs[1, 0].plot(vehicle_df_filt["Global_Time"], vehicle_df_filt["v_Accel"], label="After", linestyle="--")
    axs[1, 0].set_ylabel("Acceleration (m/s^2)")
    axs[1, 0].legend()

    axs[1, 1].plot(vehicle_df["Global_Time"], vehicle_df["Lane_X"], label="Before")
    axs[1, 1].plot(vehicle_df_filt["Global_Time"], vehicle_df_filt["Lane_X"], label="After", linestyle="--")
    axs[1, 1].set_ylabel("Position (m)")
    axs[1, 1].legend()

    plt.figure()
    v_min = min(vehicle_df_filt["v_Vel"].min(), vehicle_df["v_Vel"].min())
    v_max = max(vehicle_df_filt["v_Vel"].max(), vehicle_df["v_Vel"].max())
    speeds = np.linspace(v_min, v_max, 100)
    accel_max = accel_max_spl(speeds)
    accel_min = decel_min_spl(speeds)
    plt.plot(vehicle_df["v_Vel"], vehicle_df["v_Accel"], label="Before", alpha=0.75)
    plt.plot(vehicle_df_filt["v_Vel"], vehicle_df_filt["v_Accel"], label="After", linestyle="--")
    plt.plot(speeds, accel_max, label="Max. Accel. Capacity", color="black", linestyle="--")
    plt.plot(speeds, 0.5*accel_max, label="Driver Accel. Capacity", color="black")
    plt.plot(speeds, accel_min, label="Decel. Capacity", color="red")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.show()
sys.exit(1)


"""
df = pd.read_csv(DATA_ROOT + RELEVANT_VIDEO + ".txt", sep=",")
plot_velocities_and_headways(df)
df = plot_accelerations(df)
plot_time_space_diagram(df)
#plot_oblique_trajectories(df)
#plot_oblique_trajectories(df, start_frame=0, end_frame=2500, vehicle_labels_in_plot=True, active_cursor=True)
plt.show()

#df_info = pd.read_csv(INFO_ROOT + RELEVANT_VIDEO + ".txt", sep="\t")
#print(df_info)

df = apply_physics_informed_butterworth_filter(df)
#df = reconstruct_trajectories_cvxopt(df)
plot_time_space_diagram(df)
plot_velocities_and_headways(df)
plot_accelerations(df)
plt.show()
print(df.columns)

target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_V2.txt"
df.to_csv(target_output_file, index=False)

sys.exit(1)
"""

# #############################################################################
# 2. L2-gain Estimation
# #############################################################################
Analysis_df = None
for video in ALL_VIDEOS:
    target_output_file = "../data/6_final_trajectories_filtered/"+video+".txt"
    df = pd.read_csv(target_output_file, sep=",")

    df_info = pd.read_csv(INFO_ROOT + video + ".txt", sep="\t")
    
    L2gains_df = estimate_L2gain_CTHpolicy(df, start_frame=250, end_frame=df["Frame_ID"].max()-250)
    speed_std_df = speed_standard_deviation(df)
    dft_df = speed_dft(df)

    exp_df = L2gains_df.merge(speed_std_df, on=["Vehicle_ID"], how="left")
    exp_df = exp_df.merge(dft_df, on=["Vehicle_ID"], how="left")
    exp_df = exp_df.merge(df_info, on=["Vehicle_ID"], how="left")
    exp_df["Video"] = video
    exp_df = exp_df.sort_values(by=["Vehicle_Rank"]).reset_index()

    if Analysis_df is None:
        Analysis_df = exp_df.copy()
    else:
        Analysis_df = pd.concat((Analysis_df, exp_df))

del L2gains_df, speed_std_df, exp_df, df, df_info

print(Analysis_df)

Analysis_df["Vehicle_Rank_Label"] = Analysis_df["Vehicle_Rank"].astype(str)

"""
fig, axs = plt.subplots(
    1, len(ALL_VIDEOS), figsize=[default_plotting_settings["figure_size"][0]*len(ALL_VIDEOS), default_plotting_settings["figure_size"][1]]
)
for (ax, video) in zip(axs, ALL_VIDEOS):
    ax.plot(Analysis_df["Vehicle_Rank"], Analysis_df["L2gain"], linewidth=1.0, linestyle="--")
    ax.scatter(Analysis_df["Vehicle_Rank"], Analysis_df["L2gain"], marker="^")
    ax.set_xlabel("Vehicle Rank")
    ax.set_ylabel(r'\gamma')
    ax.set_title(video)

plt.show()
sys.exit(1)
"""

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="L2gain", hue="Vehicle_ID",
            palette=default_plotting_settings["color_palette"])
xmin, xmax = plt.gca().get_xlim()
plt.hlines(1, xmin-1, xmax, color="red", linestyles="--")
plt.xlim([xmin, xmax])
plt.show()

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="Period", hue="Vehicle_ID",
            palette=default_plotting_settings["color_palette"])
plt.show()

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="Max_Amplitude", hue="Vehicle_ID",
            palette=default_plotting_settings["color_palette"])
plt.show()

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="Speed_Std_Dev", hue="Powertrain",
            palette=default_plotting_settings["color_palette"])
plt.show()