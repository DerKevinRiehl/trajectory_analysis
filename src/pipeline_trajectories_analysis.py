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

import scipy.interpolate
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

from tools_stability_analysis import estimate_L2gain_CTHpolicy, estimate_LinfinityGain_CTHpolicy
from tools_stability_analysis import speed_standard_deviation, speed_dft

from _constants import default_plotting_settings
import _constants as cs

# #############################################################################
# CONSTANTS
# #############################################################################
COMPARE_FILTERING = True

DATA_ROOT = "../data_trajectories/6_final_trajectories/"
RCSN_ROOT = "../data_trajectories/7_final_trajectories_reconstructed/"

ALL_VIDEOS = [
    #"DJI_0933.MOV", "DJI_0934.MOV", 
    #"DJI_0939.MOV", "DJI_0940.MOV", 
    "DJI_0943.MOV", "DJI_0944.MOV"
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
if COMPARE_FILTERING:
    df = pd.read_csv(DATA_ROOT + RELEVANT_VIDEO + ".txt", sep=",")
    df = plot_accelerations(df)
    plt.close()
    #plot_velocities_and_headways(df)
    #plot_time_space_diagram(df)
    #plot_oblique_trajectories(df)
    #plot_oblique_trajectories(df, start_frame=0, end_frame=2500, vehicle_labels_in_plot=True, active_cursor=True)
    #plt.show()

    df_info = pd.read_csv(cs.VEHICLE_INFO_PATH + RELEVANT_VIDEO + ".txt", sep="\t")
    #print(df_info)
    
    df_CvxOpt = reconstruct_trajectories_cvxopt(df, vehicle_info_df=df_info, end_frame=None, relax_accel_cnst=False, weight_speed_noise=10.0)
    df_PIButterworth = apply_physics_informed_butterworth_filter(df, vehicle_info_df=df_info)
    df_CvxOpt.to_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_CvxOptNoRelax.txt", index=False)
    df_PIButterworth.to_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_PIButterworth.txt", index=False)
    

    from tools_trajectory_evaluation import calculateInternalConsistency
    from tools_trajectory_evaluation import calculatePlatoonConsistency_Headway
    from tools_trajectory_evaluation import calculatePlatoonConsistency_PhysicalValidHeadway

    print("*********************** BEFORE RECONSTRUCTION ***********************")
    e_i_max, e_i_min, e_i_avg, e_i_std, distance_travelled = calculateInternalConsistency(df, cs.FILTERING_SAMPLING_FREQUENCY)
    e_p_max, e_p_min, e_p_avg, e_p_std = calculatePlatoonConsistency_Headway(df, cs.FILTERING_SAMPLING_FREQUENCY)
    vals_violation, total_vehicle_frames = calculatePlatoonConsistency_PhysicalValidHeadway(df)
    print(f"Internal Consistency Error: Avg = {e_i_avg}, Std = {e_i_std}, Max = {e_i_max}, Min = {e_i_min}.")
    print(f"Distance Travelled = {distance_travelled}")
    print(f"Platoon Consistency Error: Avg = {e_p_avg}, Std = {e_p_std}, Max = {e_p_max}, Min = {e_p_min}.")
    print(vals_violation, total_vehicle_frames)


    print("\n\n")
    print("*********************** RECONSTRUCTION: CVXOPT ***********************")
    e_i_max, e_i_min, e_i_avg, e_i_std, distance_travelled = calculateInternalConsistency(df_CvxOpt, cs.FILTERING_SAMPLING_FREQUENCY)
    e_p_max, e_p_min, e_p_avg, e_p_std = calculatePlatoonConsistency_Headway(df_CvxOpt, cs.FILTERING_SAMPLING_FREQUENCY)
    vals_violation, total_vehicle_frames = calculatePlatoonConsistency_PhysicalValidHeadway(df_CvxOpt)
    print(f"Internal Consistency Error: Avg = {e_i_avg}, Std = {e_i_std}, Max = {e_i_max}, Min = {e_i_min}.")
    print(f"Distance Travelled = {distance_travelled}")
    print(f"Platoon Consistency Error: Avg = {e_p_avg}, Std = {e_p_std}, Max = {e_p_max}, Min = {e_p_min}.")
    print(vals_violation, total_vehicle_frames)

    print("\n\n")
    print("*********************** RECONSTRUCTION: PI-Butterworth ***********************")
    e_i_max, e_i_min, e_i_avg, e_i_std, distance_travelled = calculateInternalConsistency(df_PIButterworth, cs.FILTERING_SAMPLING_FREQUENCY)
    e_p_max, e_p_min, e_p_avg, e_p_std = calculatePlatoonConsistency_Headway(df_PIButterworth, cs.FILTERING_SAMPLING_FREQUENCY)
    vals_violation, total_vehicle_frames = calculatePlatoonConsistency_PhysicalValidHeadway(df_PIButterworth)
    print(f"Internal Consistency Error: Avg = {e_i_avg}, Std = {e_i_std}, Max = {e_i_max}, Min = {e_i_min}.")
    print(f"Distance Travelled = {distance_travelled}")
    print(f"Platoon Consistency Error: Avg = {e_p_avg}, Std = {e_p_std}, Max = {e_p_max}, Min = {e_p_min}.")
    print(vals_violation, total_vehicle_frames)
    sys.exit(1)

    unique_vehicles = df["Vehicle_ID"].unique()
    for vehicle_id in unique_vehicles:
        vehicle_df = df[df["Vehicle_ID"] == vehicle_id]
        vehicle_df_CvxOpt = df_CvxOpt[df_CvxOpt["Vehicle_ID"] == vehicle_id]
        vehicle_df_PIButterworth = df_PIButterworth[df_PIButterworth["Vehicle_ID"] == vehicle_id]

        mfc_car_id = df_info.loc[df_info["Vehicle_ID"] == vehicle_id, "MFC_CarID"].item()
        with open(cs.VEHICLE_INFO_PATH + f"ID{mfc_car_id}_AccelCapInterp.pkl", 'rb') as f:
            accel_max_spl = pickle.load(f)
        with open(cs.VEHICLE_INFO_PATH + f"ID{mfc_car_id}_DecelCapInterp.pkl", 'rb') as f:
            decel_min_spl = pickle.load(f)

        plt.rc('font', family='sans-serif') 
        plt.rc('font', serif='Arial') 
        fig, axs = plt.subplots(2, 2, figsize=(12, 5), dpi=100)
        axs[0, 0].plot(vehicle_df["Global_Time"], vehicle_df["v_Vel"], label="Original")
        axs[0, 0].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["v_Vel"], label="CvxOpt", linestyle="--")
        axs[0, 0].plot(vehicle_df_PIButterworth["Global_Time"], vehicle_df_PIButterworth["v_Vel"], label="PI-Butterworth", linestyle="-.", alpha=0.75)
        axs[0, 0].set_ylabel("Speed (m/s)")
        axs[0, 0].legend()

        axs[0, 1].plot(vehicle_df["Global_Time"], vehicle_df["Space_Hdwy"], label="Original")
        axs[0, 1].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["Space_Hdwy"], label="CvxOpt", linestyle="--")
        axs[0, 1].plot(vehicle_df_PIButterworth["Global_Time"], vehicle_df_PIButterworth["Space_Hdwy"], label="PI-Butterworth", linestyle="-.", alpha=0.75)
        axs[0, 1].set_ylabel("Space_Headway (m)")
        axs[0, 1].legend()

        axs[1, 0].plot(vehicle_df["Global_Time"], vehicle_df["v_Accel"], label="Original")
        axs[1, 0].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["v_Accel"], label="CvxOpt", linestyle="--")
        axs[1, 0].plot(vehicle_df_PIButterworth["Global_Time"], vehicle_df_PIButterworth["v_Accel"], label="PI-Butterworth", linestyle="-.", alpha=0.75)
        axs[1, 0].set_ylabel("Acceleration (m/s^2)")
        axs[1, 0].legend()

        axs[1, 1].plot(vehicle_df["Global_Time"], vehicle_df["Lane_X"], label="Original")
        axs[1, 1].plot(vehicle_df_CvxOpt["Global_Time"], vehicle_df_CvxOpt["Lane_X"], label="CvxOpt", linestyle="--")
        axs[1, 1].plot(vehicle_df_PIButterworth["Global_Time"], vehicle_df_PIButterworth["Lane_X"], label="PI-Butterworth", linestyle="-.", alpha=0.75)
        axs[1, 1].set_ylabel("Position (m)")
        axs[1, 1].legend()

        plt.figure()
        v_min = min(vehicle_df_CvxOpt["v_Vel"].min(), vehicle_df["v_Vel"].min())
        v_max = max(vehicle_df_CvxOpt["v_Vel"].max(), vehicle_df["v_Vel"].max())
        speeds = np.linspace(v_min, v_max, 100)
        accel_max = accel_max_spl(speeds)
        accel_min = decel_min_spl(speeds)
        plt.plot(vehicle_df["v_Vel"], vehicle_df["v_Accel"], label="Original", alpha=0.75)
        plt.plot(vehicle_df_CvxOpt["v_Vel"], vehicle_df_CvxOpt["v_Accel"], label="CvxOpt", linestyle="--")
        plt.plot(vehicle_df_PIButterworth["v_Vel"], vehicle_df_PIButterworth["v_Accel"], label="PI-Butterworth", linestyle="-.", alpha=0.75)
        plt.plot(speeds, accel_max, label="Max. Accel. Capacity", color="black", linestyle="--")
        plt.plot(speeds, 0.5*accel_max, label="Driver Accel. Capacity", color="black")
        plt.plot(speeds, accel_min, label="Decel. Capacity", color="red")
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Acceleration (m/s^2)")
        plt.legend()
        plt.show()

    sys.exit(1)


# #############################################################################
# 2. L2-gain Estimation
# #############################################################################
# """
Analysis_df = None
for video in ALL_VIDEOS:
    #df = pd.read_csv(DATA_ROOT + video + ".txt", sep=",")
    #df = plot_accelerations(df)
    #plt.close()
    #df_info = pd.read_csv(cs.VEHICLE_INFO_PATH + video + ".txt", sep="\t")
    #if video == "DJI_0940.MOV":
    #    df_reconst_traj = reconstruct_trajectories_cvxopt(df, vehicle_info_df=df_info, end_frame=6540, relax_accel_cnst=False)
    #else:
    #    df_reconst_traj = reconstruct_trajectories_cvxopt(df, vehicle_info_df=df_info, end_frame=None, relax_accel_cnst=False)
    #df_reconst_traj.to_csv(RCSN_ROOT + video + "_norelax.txt", index=False)

    df = pd.read_csv(RCSN_ROOT + video + "_norelax.txt", sep=",")
    df_info = pd.read_csv(cs.VEHICLE_INFO_PATH + video + ".txt", sep="\t")
    
    L2gains_df = estimate_L2gain_CTHpolicy(df, start_frame=250, end_frame=df["Frame_ID"].max()-250)
    LinfGains_df = estimate_LinfinityGain_CTHpolicy(df, start_frame=250, end_frame=df["Frame_ID"].max()-250)
    speed_std_df = speed_standard_deviation(df)
    dft_df = speed_dft(df)

    exp_df = L2gains_df.merge(speed_std_df, on=["Vehicle_ID"], how="left")
    exp_df = exp_df.merge(LinfGains_df, on=["Vehicle_ID"], how="left")
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


#fig, axs = plt.subplots(
#    1, len(ALL_VIDEOS), figsize=[default_plotting_settings["figure_size"][0]*len(ALL_VIDEOS), default_plotting_settings["figure_size"][1]]
#)
#for (ax, video) in zip(axs, ALL_VIDEOS):
#    ax.plot(Analysis_df["Vehicle_Rank"], Analysis_df["L2gain"], linewidth=1.0, linestyle="--")
#    ax.scatter(Analysis_df["Vehicle_Rank"], Analysis_df["L2gain"], marker="^")
#    ax.set_xlabel("Vehicle Rank")
#    ax.set_ylabel(r'\gamma')
#    ax.set_title(video)
#
#plt.show()
#sys.exit(1)


sns.catplot(data=Analysis_df, kind="bar", x="Video", y="L2gain", hue="Vehicle_ID",
            palette=default_plotting_settings["color_palette"])
xmin, xmax = plt.gca().get_xlim()
plt.hlines(1, xmin-1, xmax, color="red", linestyles="--")
plt.xlim([xmin, xmax])

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="LinfGain", hue="Vehicle_ID",
            palette=default_plotting_settings["color_palette"])
xmin, xmax = plt.gca().get_xlim()
plt.hlines(1, xmin-1, xmax, color="red", linestyles="--")
plt.xlim([xmin, xmax])

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="Period", hue="Vehicle_ID",
            palette=default_plotting_settings["color_palette"])

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="Max_Amplitude", hue="Vehicle_ID",
            palette=default_plotting_settings["color_palette"])

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="Speed_Std_Dev", hue="Powertrain",
            palette=default_plotting_settings["color_palette"])

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="L2gain", hue="Powertrain",
            palette=default_plotting_settings["color_palette"])

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="LinfGain", hue="Powertrain",
            palette=default_plotting_settings["color_palette"])
plt.show()
# sys.exit(1)
# """

# #############################################################################
# 3. Windowed Analysis
# #############################################################################
for video in ALL_VIDEOS:
    print(f"**************** {video} ****************")
    df = pd.read_csv(RCSN_ROOT + video + "_norelax.txt", sep=",")
    df_info = pd.read_csv(cs.VEHICLE_INFO_PATH + video + ".txt", sep="\t")

    plot_velocities_and_headways(df)
    plot_time_space_diagram(df)
    #plot_oblique_trajectories(df)
    #plot_oblique_trajectories(df, start_frame=0, end_frame=2500, vehicle_labels_in_plot=True, active_cursor=True)
    plt.show()

    t_start = float(input("Please input start time for windowed analysis: "))
    t_end = float(input("Please input end time for windowed analysis: "))
    start_frame, end_frame = int(t_start*cs.FILTERING_SAMPLING_FREQUENCY), int(t_end*cs.FILTERING_SAMPLING_FREQUENCY)
    L2gains_df = estimate_L2gain_CTHpolicy(df, start_frame=start_frame, end_frame=end_frame)
    LinfGains_df = estimate_LinfinityGain_CTHpolicy(df, start_frame=int(t_start*cs.FILTERING_SAMPLING_FREQUENCY), end_frame=int(t_end*cs.FILTERING_SAMPLING_FREQUENCY))
    speed_std_df = speed_standard_deviation(df[(df["Frame_ID"] >= start_frame) & (df["Frame_ID"] <= end_frame)], start_frame=start_frame)

    exp_df = L2gains_df.merge(speed_std_df, on=["Vehicle_ID"], how="left")
    exp_df = exp_df.merge(LinfGains_df, on=["Vehicle_ID"], how="left")
    exp_df = exp_df.merge(df_info, on=["Vehicle_ID"], how="left")
    exp_df["Video"] = video
    exp_df = exp_df.sort_values(by=["Vehicle_Rank"]).reset_index()

    sns.catplot(data=exp_df, kind="bar", x="Vehicle_ID", y="Speed_Std_Dev", hue="Powertrain",
                palette=default_plotting_settings["color_palette"])
    plt.xticks(rotation=30)

    sns.catplot(data=exp_df, kind="bar", x="Vehicle_ID", y="L2gain", hue="Powertrain",
                palette=default_plotting_settings["color_palette"])
    plt.xticks(rotation=30)

    sns.catplot(data=exp_df, kind="bar", x="Vehicle_ID", y="LinfGain", hue="Powertrain",
                palette=default_plotting_settings["color_palette"])
    plt.xticks(rotation=30)
    plt.show()

    #sys.exit(1)