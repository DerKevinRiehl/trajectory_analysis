# Imports
import pandas as pd
import numpy as np
import sys



# Parameters
RELEVANT_VIDEO = "DJI_0933.MOV"
target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
VIDEO_FRAME_RATE = 25
cutway_constant = 60

# Methods
def determineErrorStatistics(trajectory_error: list):
    e_s_max = np.max(abs(trajectory_error))
    e_s_min = np.min(abs(trajectory_error))
    e_s_avg = np.mean(abs(trajectory_error))
    e_s_std = np.std(abs(trajectory_error))
    return e_s_max, e_s_min, e_s_avg, e_s_std

def calculateInternalConsistency(final_trajectory_df: pd.DataFrame, VIDEO_FRAME_RATE: int):
    unique_vehicles = list(set(final_trajectory_df["Vehicle_ID"].tolist()))
    vals_max = []
    vals_min = []
    vals_avg = []
    vals_std = []
    distance_travelled = []
    for vehicle_id in unique_vehicles:
        # select relevant vehicle
        vehicle_df = final_trajectory_df[final_trajectory_df["Vehicle_ID"]==vehicle_id]
        # estimate trajectory based on velocity
        vehicle_df = vehicle_df[["v_Vel", "Lane_X"]]
        vehicle_df["v_Vel"] = abs(vehicle_df["v_Vel"])
        vehicle_df["s_frame"] = vehicle_df["v_Vel"] * (1 / VIDEO_FRAME_RATE)
        vehicle_df["trajectory_estimated_velocity"] = vehicle_df["s_frame"].cumsum()
        # cut tails away for comparison, as kalman filter not good at tails
        vehicle_df = vehicle_df[cutway_constant+10:-cutway_constant-10]
        # determine trajectory error
        vehicle_df["Lane_X"] = vehicle_df["Lane_X"] - vehicle_df["Lane_X"].iloc[0]
        vehicle_df["trajectory_estimated_velocity"] = vehicle_df["trajectory_estimated_velocity"] - vehicle_df["trajectory_estimated_velocity"].iloc[0]
        # scale_factor = vehicle_df["Lane_X"].iloc[-1] / vehicle_df["trajectory_estimated_velocity"].iloc[-1]
        distance_travelled.append(vehicle_df["Lane_X"].iloc[-1])
        # vehicle_df["trajectory_estimated_velocity"] = vehicle_df["trajectory_estimated_velocity"] * scale_factor
        vehicle_df["trajectory_error"] = vehicle_df["trajectory_estimated_velocity"] - vehicle_df["Lane_X"]
        # account error statistics
        e_s_max, e_s_min, e_s_avg, e_s_std = determineErrorStatistics(vehicle_df["trajectory_error"])
        vals_max.append(e_s_max)
        vals_min.append(e_s_min)
        vals_avg.append(e_s_avg)
        vals_std.append(e_s_std)
    max_error = np.nanmean(vals_max)
    min_error = np.nanmean(vals_min)
    avg_error = np.nanmean(vals_avg)
    std_error = np.nanmean(vals_std)
    distance_travelled = np.nanmean(distance_travelled)
    return max_error, min_error, avg_error, std_error, distance_travelled

def calculatePlatoonConsistency_Headway(final_trajectory_df: pd.DataFrame, VIDEO_FRAME_RATE: int):
    unique_vehicles = list(set(final_trajectory_df["Vehicle_ID"].tolist()))
    vals_max = []
    vals_min = []
    vals_avg = []
    vals_std = []
    # determine proceeding order
    proceeding_order = {}
    for vehicle_id in unique_vehicles:
        vehicle_df = final_trajectory_df[final_trajectory_df["Vehicle_ID"]==vehicle_id]
        proceeding_order[vehicle_id] = vehicle_df["Proceeding"].iloc[0]
    for vehicle_id in unique_vehicles:
        # select relevant vehicle
        vehicle_df = final_trajectory_df[final_trajectory_df["Vehicle_ID"]==vehicle_id]
        initial_position_x = vehicle_df["Lane_X"].iloc[0]
        vehicle_df = vehicle_df[["Frame_ID", "v_Vel", "Lane_X", "Space_Hdwy"]]
        sub_merge_df = final_trajectory_df[final_trajectory_df["Vehicle_ID"]==proceeding_order[vehicle_id]]
        # print(vehicle_id, proceeding_order[vehicle_id])
        initial_position_y = sub_merge_df["Lane_X"].iloc[0]
        sub_merge_df = sub_merge_df[["Frame_ID", "v_Vel",]]
        vehicle_df = vehicle_df.merge(sub_merge_df, on="Frame_ID", how="left")
        # estimate trajectory of vehicle and follower based on velocity
        vehicle_df["v_Vel_x"] = abs(vehicle_df["v_Vel_x"])
        vehicle_df["v_Vel_y"] = abs(vehicle_df["v_Vel_y"])
        vehicle_df["s_frame_x"] = vehicle_df["v_Vel_x"] * (1 / VIDEO_FRAME_RATE)
        vehicle_df["s_frame_y"] = vehicle_df["v_Vel_y"] * (1 / VIDEO_FRAME_RATE)
        vehicle_df["trajectory_estimated_velocity_x"] = vehicle_df["s_frame_x"].cumsum()
        vehicle_df["trajectory_estimated_velocity_y"] = vehicle_df["s_frame_y"].cumsum()
        vehicle_df["space_headway_estimated_velocity"] = vehicle_df["trajectory_estimated_velocity_y"] - vehicle_df["trajectory_estimated_velocity_x"]
        initial_headway = abs(initial_position_y - initial_position_x)
        vehicle_df["space_headway_estimated_velocity"] = vehicle_df["space_headway_estimated_velocity"] + initial_headway
        vehicle_df = vehicle_df[["Frame_ID", "Space_Hdwy", "space_headway_estimated_velocity"]]
        # cut tails away for comparison, as kalman filter not good at tails
        vehicle_df = vehicle_df[cutway_constant+10:-cutway_constant-10]
        # determine trajectory error
        vehicle_df["Space_Hdwy"] = vehicle_df["Space_Hdwy"] - vehicle_df["Space_Hdwy"].iloc[0]
        vehicle_df["space_headway_estimated_velocity"] = vehicle_df["space_headway_estimated_velocity"] - vehicle_df["space_headway_estimated_velocity"].iloc[0]
        scale_factor = vehicle_df["Space_Hdwy"].iloc[-1] / vehicle_df["space_headway_estimated_velocity"].iloc[-1]
        vehicle_df["space_headway_estimated_velocity"] = vehicle_df["space_headway_estimated_velocity"] * scale_factor
        vehicle_df["headway_error"] = vehicle_df["space_headway_estimated_velocity"] - vehicle_df["Space_Hdwy"]
        # # account error statistics
        e_s_max, e_s_min, e_s_avg, e_s_std = determineErrorStatistics(vehicle_df["headway_error"])
        vals_max.append(e_s_max)
        vals_min.append(e_s_min)
        vals_avg.append(e_s_avg)
        vals_std.append(e_s_std)
    max_error = np.nanmean(vals_max)
    min_error = np.nanmean(vals_min)
    avg_error = np.nanmean(vals_avg)
    std_error = np.nanmean(vals_std)
    return max_error, min_error, avg_error, std_error



def calculatePlatoonConsistency_PhysicalValidHeadway(final_trajectory_df: pd.DataFrame):
    unique_vehicles = list(set(final_trajectory_df["Vehicle_ID"].tolist()))
    vals_violation = []
    # determine proceeding order
    proceeding_order = {}
    for vehicle_id in unique_vehicles:
        vehicle_df = final_trajectory_df[final_trajectory_df["Vehicle_ID"]==vehicle_id]
        proceeding_order[vehicle_id] = vehicle_df["Proceeding"].iloc[0]
    for vehicle_id in unique_vehicles:
        follower_vehicle_id = proceeding_order[vehicle_id]
        # calculate minimum physical distance
        vehicle_df = final_trajectory_df[final_trajectory_df["Vehicle_ID"]==vehicle_id]
        vehicle_df_follower = final_trajectory_df[final_trajectory_df["Vehicle_ID"]==follower_vehicle_id]
        v_length = vehicle_df["v_Length"].iloc[-1]
        v_length_follower = vehicle_df_follower["v_Length"].iloc[-1]
        physical_min_dist = (v_length+v_length_follower)/2
        # determine times with not physical distance
        times_violating = np.sum(vehicle_df["Space_Hdwy"]<physical_min_dist)
        vals_violation.append(times_violating)
    vals_violation = np.sum(vals_violation)
    total_vehicle_frames = final_trajectory_df.shape[0]
    return vals_violation, total_vehicle_frames




print("=====================")
print("Internal Consistency")
print("=====================")
RELEVANT_VIDEO = "DJI_0933.MOV"
target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
final_trajectory_df = pd.read_csv(target_output_file)
print(RELEVANT_VIDEO)
print (calculateInternalConsistency(final_trajectory_df, VIDEO_FRAME_RATE))

# RELEVANT_VIDEO = "DJI_0934.MOV"
# #target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
# final_trajectory_df = pd.read_csv(target_output_file)
# print(RELEVANT_VIDEO)
# print (calculateInternalConsistency(final_trajectory_df, VIDEO_FRAME_RATE))



print("\n=====================")
print("Platoon Consistency")
print("=====================")
RELEVANT_VIDEO = "DJI_0933.MOV"
target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
final_trajectory_df = pd.read_csv(target_output_file)
print(RELEVANT_VIDEO)
print (calculatePlatoonConsistency_Headway(final_trajectory_df, VIDEO_FRAME_RATE))

# RELEVANT_VIDEO = "DJI_0934.MOV"
# #target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
# final_trajectory_df = pd.read_csv(target_output_file)
# print(RELEVANT_VIDEO)
# print (calculatePlatoonConsistency_Headway(final_trajectory_df, VIDEO_FRAME_RATE))




print("\n=====================")
print("Platoon Consistency (Frames With Physically Impossible Distances)")
print("=====================")
RELEVANT_VIDEO = "DJI_0933.MOV"
target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
final_trajectory_df = pd.read_csv(target_output_file)
print(RELEVANT_VIDEO)
print (calculatePlatoonConsistency_PhysicalValidHeadway(final_trajectory_df))

# RELEVANT_VIDEO = "DJI_0934.MOV"
# #target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
# final_trajectory_df = pd.read_csv(target_output_file)
# print(RELEVANT_VIDEO)
# print (calculatePlatoonConsistency_PhysicalValidHeadway(final_trajectory_df))

