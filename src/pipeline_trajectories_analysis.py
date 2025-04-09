"""
Consistent Vehicle Trajectory Extraction From Aerial Recordings Using Oriented Object Detection
-------------------------------------------
Authors:        Kevin Riehl, Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024 - 2025
Submitted to:   Scientific Reports
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
from tools_trajectory_filtering import apply_naive_butterworth_filter
from tools_trajectory_filtering import reconstruct_trajectories_wavelet

from tools_trajectory_plotting import plot_time_space_diagram
from tools_trajectory_plotting import plot_oblique_trajectories
from tools_trajectory_plotting import plot_velocities_and_headways
from tools_trajectory_plotting import plot_accelerations

from tools_stability_analysis import estimate_L2gain_CTHpolicy, estimate_LinfinityGain_CTHpolicy
from tools_stability_analysis import speed_standard_deviation, speed_dft, compute_response_time

from _constants import default_plotting_settings
import _constants as cs

# #############################################################################
# CONSTANTS
# #############################################################################
COMPARE_RECONSTRUCTION = False
RUN_RECONSTRUCTION = True

DATA_ROOT = "../data_trajectories/6_final_trajectories/"
RCSN_ROOT = "../data_trajectories/7_final_trajectories_reconstructed/"

ALL_VIDEOS = [
    "DJI_0933.MOV", "DJI_0934.MOV", 
    "DJI_0939.MOV", "DJI_0940.MOV", 
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
if COMPARE_RECONSTRUCTION:
    df = pd.read_csv(DATA_ROOT + RELEVANT_VIDEO + ".txt", sep=",")
    df = plot_accelerations(df)
    plt.close()

    df_info = pd.read_csv(cs.VEHICLE_INFO_PATH + RELEVANT_VIDEO + ".txt", sep="\t")
    
    df_CvxOpt = reconstruct_trajectories_cvxopt(df, vehicle_info_df=df_info, end_frame=None, relax_accel_cnst=False, weight_speed_noise=10.0)
    #df_PIButterworth = apply_physics_informed_butterworth_filter(df, vehicle_info_df=df_info)
    df_Butterworth = apply_naive_butterworth_filter(df)
    df_Wavelet = reconstruct_trajectories_wavelet(df, soft_thresholding=False)
    df_CvxOpt.to_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_CvxOptNoRelax.txt", index=False)
    #df_PIButterworth.to_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_PIButterworth.txt", index=False)
    df_Butterworth.to_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_Butterworth.txt", index=False)
    df_Wavelet.to_csv(RCSN_ROOT + RELEVANT_VIDEO + "_Comparison_Wavelet.txt", index=False)
    

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

    """
    print("\n\n")
    print("*********************** RECONSTRUCTION: PI-Butterworth ***********************")
    e_i_max, e_i_min, e_i_avg, e_i_std, distance_travelled = calculateInternalConsistency(df_PIButterworth, cs.FILTERING_SAMPLING_FREQUENCY)
    e_p_max, e_p_min, e_p_avg, e_p_std = calculatePlatoonConsistency_Headway(df_PIButterworth, cs.FILTERING_SAMPLING_FREQUENCY)
    vals_violation, total_vehicle_frames = calculatePlatoonConsistency_PhysicalValidHeadway(df_PIButterworth)
    print(f"Internal Consistency Error: Avg = {e_i_avg}, Std = {e_i_std}, Max = {e_i_max}, Min = {e_i_min}.")
    print(f"Distance Travelled = {distance_travelled}")
    print(f"Platoon Consistency Error: Avg = {e_p_avg}, Std = {e_p_std}, Max = {e_p_max}, Min = {e_p_min}.")
    print(vals_violation, total_vehicle_frames)
    """

    print("\n\n")
    print("*********************** RECONSTRUCTION: Wavelet Transform ***********************")
    e_i_max, e_i_min, e_i_avg, e_i_std, distance_travelled = calculateInternalConsistency(df_Wavelet, cs.FILTERING_SAMPLING_FREQUENCY)
    e_p_max, e_p_min, e_p_avg, e_p_std = calculatePlatoonConsistency_Headway(df_Wavelet, cs.FILTERING_SAMPLING_FREQUENCY)
    vals_violation, total_vehicle_frames = calculatePlatoonConsistency_PhysicalValidHeadway(df_Wavelet)
    print(f"Internal Consistency Error: Avg = {e_i_avg}, Std = {e_i_std}, Max = {e_i_max}, Min = {e_i_min}.")
    print(f"Distance Travelled = {distance_travelled}")
    print(f"Platoon Consistency Error: Avg = {e_p_avg}, Std = {e_p_std}, Max = {e_p_max}, Min = {e_p_min}.")
    print(vals_violation, total_vehicle_frames)

    print("\n\n")
    print("*********************** RECONSTRUCTION: Naive Butterworth ***********************")
    e_i_max, e_i_min, e_i_avg, e_i_std, distance_travelled = calculateInternalConsistency(df_Butterworth, cs.FILTERING_SAMPLING_FREQUENCY)
    e_p_max, e_p_min, e_p_avg, e_p_std = calculatePlatoonConsistency_Headway(df_Butterworth, cs.FILTERING_SAMPLING_FREQUENCY)
    vals_violation, total_vehicle_frames = calculatePlatoonConsistency_PhysicalValidHeadway(df_Butterworth)
    print(f"Internal Consistency Error: Avg = {e_i_avg}, Std = {e_i_std}, Max = {e_i_max}, Min = {e_i_min}.")
    print(f"Distance Travelled = {distance_travelled}")
    print(f"Platoon Consistency Error: Avg = {e_p_avg}, Std = {e_p_std}, Max = {e_p_max}, Min = {e_p_min}.")
    print(vals_violation, total_vehicle_frames)

    sys.exit(1)


# #############################################################################
# 2. L2-gain Estimation
# #############################################################################
# """
Analysis_df = None
for video in ALL_VIDEOS:
    if RUN_RECONSTRUCTION:
        df = pd.read_csv(DATA_ROOT + video + ".txt", sep=",")
        df = plot_accelerations(df)
        plt.close()
        df_info = pd.read_csv(cs.VEHICLE_INFO_PATH + video + ".txt", sep="\t")
        print(video)
        if video == "DJI_0940.MOV":
            df_reconst_traj = reconstruct_trajectories_cvxopt(df, vehicle_info_df=df_info, end_frame=6540, relax_accel_cnst=False, weight_speed_noise=10.0)
        else:
            df_reconst_traj = reconstruct_trajectories_cvxopt(df, vehicle_info_df=df_info, end_frame=None, relax_accel_cnst=False, weight_speed_noise=10.0)
        df_reconst_traj.to_csv(RCSN_ROOT + video + "_norelax.txt", index=False)

    df = pd.read_csv(RCSN_ROOT + video + "_norelax.txt", sep=",")
    df_info = pd.read_csv(cs.VEHICLE_INFO_PATH + video + ".txt", sep="\t")
    
    L2gains_df = estimate_L2gain_CTHpolicy(df, start_frame=250, end_frame=df["Frame_ID"].max()-250)
    LinfGains_df = estimate_LinfinityGain_CTHpolicy(df, start_frame=250, end_frame=df["Frame_ID"].max()-250)
    speed_std_df = speed_standard_deviation(df)
    dft_df = speed_dft(df)
    tau_df = compute_response_time(df, start_frame=250, end_frame=df["Frame_ID"].max()-250, dtau=0.04)

    exp_df = L2gains_df.merge(speed_std_df, on=["Vehicle_ID"], how="left")
    exp_df = exp_df.merge(LinfGains_df, on=["Vehicle_ID"], how="left")
    exp_df = exp_df.merge(dft_df, on=["Vehicle_ID"], how="left")
    exp_df = exp_df.merge(tau_df, on=["Vehicle_ID"], how="left")
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


sns.catplot(data=Analysis_df, kind="bar", x="Video", y="L2gain_Speed", hue="Vehicle_ID",
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

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="L2gain_Speed", hue="Powertrain",
            palette=default_plotting_settings["color_palette"])

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="LinfGain", hue="Powertrain",
            palette=default_plotting_settings["color_palette"])

sns.catplot(data=Analysis_df, kind="bar", x="Video", y="Response_Time", hue="Powertrain",
            palette=default_plotting_settings["color_palette"])

plt.show()
sys.exit(1)
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