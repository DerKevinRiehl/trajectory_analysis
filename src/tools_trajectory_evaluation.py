# Imports
import pandas as pd
import numpy as np
import sys, os, pickle

from _constants import VEHICLE_INFO_PATH, FILTERING_SAMPLING_FREQUENCY

# Parameters
cutway_constant = 60

# Methods
def determineErrorStatistics(trajectory_error: list):
    e_s_max = np.max(abs(trajectory_error))
    e_s_min = np.min(abs(trajectory_error))
    e_s_avg = np.mean(abs(trajectory_error))
    e_s_std = np.std(abs(trajectory_error))
    return e_s_max, e_s_min, e_s_avg, e_s_std

def calculateInternalConsistency(final_trajectory_df: pd.DataFrame, FILTERING_SAMPLING_FREQUENCY: int):
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
        vehicle_df["s_frame"] = vehicle_df["v_Vel"] * (1 / FILTERING_SAMPLING_FREQUENCY)
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

def calculatePlatoonConsistency_Headway(final_trajectory_df: pd.DataFrame, FILTERING_SAMPLING_FREQUENCY: int):
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
        vehicle_df["s_frame_x"] = vehicle_df["v_Vel_x"] * (1 / FILTERING_SAMPLING_FREQUENCY)
        vehicle_df["s_frame_y"] = vehicle_df["v_Vel_y"] * (1 / FILTERING_SAMPLING_FREQUENCY)
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


def calculateEnergyConsumption(trajectory_df: pd.DataFrame, vehicle_info_df: pd.DataFrame) -> pd.DataFrame:
    f0, f1, f2, g, theta = 213, 0.0861, 0.0027, 9.81, 0
    Ts = 1.0 / FILTERING_SAMPLING_FREQUENCY
    res = {'Vehicle_ID': [], 'Ec': []}
    unique_vehicles = trajectory_df['Vehicle_ID'].unique()
    mod_trajectory_df = None
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"]==vehicle_id].copy()
        if not vehicle_df['Frame_ID'].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by='Frame_ID', ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns='index')
        mfc_car_id = vehicle_info_df.loc[vehicle_info_df["Vehicle_ID"] == vehicle_id, "MFC_CarID"].item()
        with open(os.path.join(VEHICLE_INFO_PATH, f"ID{mfc_car_id}_AccelCapInterp.pkl"), 'rb') as f:
            accel_max_spl = pickle.load(f)
        m = vehicle_info_df.loc[vehicle_info_df["Vehicle_ID"] == vehicle_id, "Mass_kg"].item()

        """
        import matplotlib.pyplot as plt
        
        plt.close('all')
        plt.figure()
        v = np.linspace(0, 50, 1000)
        a_max = accel_max_spl(v)
        Pmax = 1e-03 * np.maximum(0, (f0 + f1*v + f2*np.power(v, 2) + 1.03*m*a_max + m*g*np.sin(theta))*v)
        plt.plot(v, a_max)
        plt.plot(v, Pmax)
        plt.hlines(vehicle_info_df.loc[vehicle_info_df["Vehicle_ID"] == vehicle_id, "Max_Power_KW"].item(), xmin=0, xmax=50)
        plt.xlim([0, 50])
        plt.show()
        sys.exit(1)
        """

        v, a = vehicle_df['v_Vel'].to_numpy(), vehicle_df['v_Accel'].to_numpy()
        a_max = accel_max_spl(v)
        vehicle_df['Pt'] = 1e-03 * np.maximum(0, (f0 + f1*v + f2*np.power(v, 2) + 1.03*m*a + m*g*np.sin(theta))*v)
        vehicle_df['Pmax'] = 1e-03 * np.maximum(0, (f0 + f1*v + f2*np.power(v, 2) + 1.03*m*a_max + m*g*np.sin(theta))*v)
        res['Vehicle_ID'].append(vehicle_id)
        res['Ec'].append(np.sum(vehicle_df['Pt'].to_numpy()*Ts) / (0.036 * np.sum(v*Ts)))
        if mod_trajectory_df is None:
            mod_trajectory_df = vehicle_df.copy()
        else:
            mod_trajectory_df = pd.concat((mod_trajectory_df, vehicle_df))
    return pd.DataFrame(res), mod_trajectory_df


"""
RELEVANT_VIDEO = "DJI_0933.MOV"
target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"

print("=====================")
print("Internal Consistency")
print("=====================")
RELEVANT_VIDEO = "DJI_0933.MOV"
target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
final_trajectory_df = pd.read_csv(target_output_file)
print(RELEVANT_VIDEO)
print (calculateInternalConsistency(final_trajectory_df, FILTERING_SAMPLING_FREQUENCY))

# RELEVANT_VIDEO = "DJI_0934.MOV"
# #target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
# final_trajectory_df = pd.read_csv(target_output_file)
# print(RELEVANT_VIDEO)
# print (calculateInternalConsistency(final_trajectory_df, FILTERING_SAMPLING_FREQUENCY))



print("\n=====================")
print("Platoon Consistency")
print("=====================")
RELEVANT_VIDEO = "DJI_0933.MOV"
target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
final_trajectory_df = pd.read_csv(target_output_file)
print(RELEVANT_VIDEO)
print (calculatePlatoonConsistency_Headway(final_trajectory_df, FILTERING_SAMPLING_FREQUENCY))

# RELEVANT_VIDEO = "DJI_0934.MOV"
# #target_output_file = "../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt"
# target_output_file = "../data/6_final_trajectories_filtered/"+RELEVANT_VIDEO+"_butter.txt"
# final_trajectory_df = pd.read_csv(target_output_file)
# print(RELEVANT_VIDEO)
# print (calculatePlatoonConsistency_Headway(final_trajectory_df, FILTERING_SAMPLING_FREQUENCY))




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
"""
