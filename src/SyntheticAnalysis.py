# #############################################################################
# ########## IMPORTS
# #############################################################################
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools_filtering_angle import boundAnglePositive
from tools_filtering import calculateKalmanFilteredTrajectory, alignTrajectories, featureCalculation
from tools_trajectory_evaluation import calculateInternalConsistency
from tools_trajectory_processing import processTrajectory_synthetic
from tools_trajectory_filtering import reconstruct_trajectories_cvxopt




# #############################################################################
# ########## METHODS
# #############################################################################

def generateSyntheticTrajectory(noise_pos, noise_angle, coverage):
    start_pos = [0, 30]
    data = []
    last_x = start_pos[0]
    last_y = start_pos[1]
    v = 1 # 5 # m/s const
    last_angle = 0
    angle_vel = 2/360*2*np.pi
    delta_time = 1/25
    for frame_nr in range(0, 7500):
        x = last_x + np.cos(last_angle)*v*delta_time
        y = last_y - np.sin(last_angle)*v*delta_time
        angle = boundAnglePositive(last_angle + angle_vel*delta_time)
        noised_x = x + np.random.normal(0, noise_pos)
        noised_y = y + np.random.normal(0, noise_pos)
        data.append([frame_nr, noised_x, noised_y, 5, 5, angle + np.random.normal(0, noise_angle)])
        last_x = x
        last_y = y
        last_angle = angle
        v = 2+1*np.sin(frame_nr/25)
        angle_vel = (2+2*np.sin(frame_nr/25))/360*2*np.pi
    veh_trajectory_raw = pd.DataFrame(data, columns=["frame_nr", "x", "y", "w", "h", "angle_rad"])

    # center the trajectory around the cartesian origin
    max_x, min_x = veh_trajectory_raw["x"].max(), veh_trajectory_raw["x"].min()
    max_y, min_y = veh_trajectory_raw["y"].max(), veh_trajectory_raw["y"].min()
    veh_trajectory_raw["x"] = veh_trajectory_raw["x"] - (max_x-min_x)*0.5 - min_x
    veh_trajectory_raw["y"] = veh_trajectory_raw["y"] - (max_y-min_y)*0.5 - min_y

    remove_n = int(7500*(1-coverage))
    drop_indices = np.random.choice(veh_trajectory_raw.index, remove_n, replace=False)
    drop_indices = list(drop_indices)
    if veh_trajectory_raw.index[0] in drop_indices:
        drop_indices.remove(veh_trajectory_raw.index[0])
    elif veh_trajectory_raw.index[-1] in drop_indices:
        drop_indices.remove(veh_trajectory_raw.index[-1])
    drop_indices = np.array(drop_indices)
    veh_trajectory_raw = veh_trajectory_raw.drop(drop_indices)
    veh_trajectory_raw = featureCalculation(veh_trajectory_raw, video_frames_per_second, obb=True)
    veh_trajectory_raw = veh_trajectory_raw.reset_index()
    del veh_trajectory_raw["index"]
    return veh_trajectory_raw


# #############################################################################
# ########## MAIN - PROCESSING PER FRAME
# #############################################################################

# Determine Kalman Filter Parameters
Q_k_hbb = np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) # P_0.copy() # covariance matrix of error of state
R_k_hbb = np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) # P_0.copy() # covariance matrix of error of output
Q_k_obb = np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) # P_0.copy() # covariance matrix of error of state
R_k_obb = np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) # P_0.copy() # covariance matrix of error of output
video_frames_per_second = 25





"""
############## RUN VISUALIZATION
"""
veh_trajectory_perf = generateSyntheticTrajectory(noise_pos=0, noise_angle=0, coverage=1)
veh_trajectory_raw = generateSyntheticTrajectory(noise_pos=0.5, noise_angle=1/360*(2*np.pi), coverage=1)
first_frame = veh_trajectory_raw["frame_nr"].iloc[0]
last_frame = veh_trajectory_raw["frame_nr"].iloc[-1] 
kalman_filtered_trajectory_rts_hbb = calculateKalmanFilteredTrajectory(veh_trajectory_raw, Q_k_hbb, R_k_hbb, first_frame, last_frame, video_frames_per_second, obb=False)
kalman_filtered_trajectory_rts_obb = calculateKalmanFilteredTrajectory(veh_trajectory_raw, Q_k_obb, R_k_obb, first_frame, last_frame, video_frames_per_second, obb=True)
kalman_filtered_trajectory_perf_rts_obb = calculateKalmanFilteredTrajectory(veh_trajectory_perf, Q_k_obb, R_k_obb, first_frame, last_frame, video_frames_per_second, obb=True)
kalman_filtered_trajectory_rts_obb = alignTrajectories(kalman_filtered_trajectory_rts_obb, kalman_filtered_trajectory_rts_hbb)
kalman_filtered_trajectory_perf_rts_obb = alignTrajectories(kalman_filtered_trajectory_perf_rts_obb, kalman_filtered_trajectory_rts_obb)
kalman_filtered_trajectory_rts_hbb = kalman_filtered_trajectory_rts_hbb[["frame_nr", "time", "state1", "state2", "state3", "state4", "state5", "x", "y"]]
kalman_filtered_trajectory_rts_hbb["vehicle"] = "VEHICLE_1"
kalman_filtered_trajectory_rts_hbb = kalman_filtered_trajectory_rts_hbb.rename(columns={"x": "x_cartesian", "y": "y_cartesian"})
processed_hbb = processTrajectory_synthetic(kalman_filtered_trajectory_rts_hbb)
kalman_filtered_trajectory_rts_obb = kalman_filtered_trajectory_rts_obb[["frame_nr", "time", "state1", "state2", "state3", "state4", "state5", "x", "y"]]
kalman_filtered_trajectory_rts_obb["vehicle"] = "VEHICLE_1"
kalman_filtered_trajectory_rts_obb = kalman_filtered_trajectory_rts_obb.rename(columns={"x": "x_cartesian", "y": "y_cartesian"})
processed_obb = processTrajectory_synthetic(kalman_filtered_trajectory_rts_obb)   
kalman_filtered_trajectory_perf_rts_obb = kalman_filtered_trajectory_perf_rts_obb[["frame_nr", "time", "state1", "state2", "state3", "state4", "state5", "x", "y"]]
kalman_filtered_trajectory_perf_rts_obb["vehicle"] = "VEHICLE_1"
kalman_filtered_trajectory_perf_rts_obb = kalman_filtered_trajectory_perf_rts_obb.rename(columns={"x": "x_cartesian", "y": "y_cartesian"})
processed_obb_perf = processTrajectory_synthetic(kalman_filtered_trajectory_perf_rts_obb) 

"""
import pickle
vehicle_dynamics_path = "../data_benchmark/5_vehicle_information/vehicle_dynamics/"
with open(vehicle_dynamics_path+"accel_capacity_interpolator.pkl", "rb") as f:
    accel_max_spl = pickle.load(f)
with open(vehicle_dynamics_path+"decel_capacity_interpolator.pkl", "rb") as f:
    decel_min_spl = pickle.load(f)
reconstr_hbb = reconstruct_trajectories_cvxopt(processed_hbb, accel_max_spl, decel_min_spl)
reconstr_obb = reconstruct_trajectories_cvxopt(processed_obb, accel_max_spl, decel_min_spl)
"""

plt.figure(figsize=(12,6))

plt.subplot(3,3,1)
plt.title("Trajectory (perfect)")
plt.plot(veh_trajectory_perf["x"], veh_trajectory_perf["y"])

plt.subplot(3,3,4)
plt.title("Trajectory (noised)")
plt.plot(veh_trajectory_raw["x"], veh_trajectory_raw["y"])

plt.subplot(3,3,2)
plt.title("X-Coordinate")
plt.plot(veh_trajectory_perf["frame_nr"],                veh_trajectory_perf["x"], label="perfect")
plt.plot(veh_trajectory_raw["frame_nr"],                 veh_trajectory_raw["x"], label="noised")
plt.plot(kalman_filtered_trajectory_rts_hbb["frame_nr"], kalman_filtered_trajectory_rts_hbb["state1"], label="kalman HBB")
plt.plot(kalman_filtered_trajectory_rts_obb["frame_nr"], kalman_filtered_trajectory_rts_obb["state1"], label="kalman OBB")
# plt.legend()
plt.xlim(0,200)
plt.ylim(0,15)

plt.subplot(3,3,5)
plt.title("Y-Coordinate")
plt.plot(veh_trajectory_perf["frame_nr"],                veh_trajectory_perf["y"], label="perfect")
plt.plot(veh_trajectory_raw["frame_nr"],                 veh_trajectory_raw["y"], label="noised")
plt.plot(kalman_filtered_trajectory_rts_hbb["frame_nr"], kalman_filtered_trajectory_rts_hbb["state2"], label="kalman HBB")
plt.plot(kalman_filtered_trajectory_rts_obb["frame_nr"], kalman_filtered_trajectory_rts_obb["state2"], label="kalman OBB")
plt.legend()
plt.xlim(0,200)
plt.ylim(26,32)

plt.subplot(3,3,3)
plt.title("State 4 (V)")
plt.plot(kalman_filtered_trajectory_rts_hbb["frame_nr"], kalman_filtered_trajectory_rts_hbb["state4"], label="kalman HBB")
plt.plot(kalman_filtered_trajectory_rts_obb["frame_nr"], kalman_filtered_trajectory_rts_obb["state4"], label="kalman OBB")
plt.plot(processed_hbb["Frame_ID"], processed_hbb["v_Vel"], label="kalman HBB proc")
plt.plot(processed_obb["Frame_ID"], processed_obb["v_Vel"], label="kalman OBB proc")
plt.legend()
plt.xlim(0,200)

plt.subplot(3,3,6)
plt.title("State 5 (angle derived)")
plt.plot(kalman_filtered_trajectory_rts_hbb["frame_nr"], kalman_filtered_trajectory_rts_hbb["state5"], label="kalman HBB")
plt.plot(kalman_filtered_trajectory_rts_obb["frame_nr"], kalman_filtered_trajectory_rts_obb["state5"], label="kalman OBB")
plt.plot(processed_hbb["Frame_ID"], processed_hbb["v_AngleVel"], label="kalman HBB proc")
plt.plot(processed_obb["Frame_ID"], processed_obb["v_AngleVel"], label="kalman OBB proc")
plt.legend()

plt.subplot(3,3,7)
plt.title("Lane_X")
plt.plot(processed_hbb["Frame_ID"], processed_hbb["Lane_X"], label="kalman HBB proc")
plt.plot(processed_obb["Frame_ID"], processed_obb["Lane_X"], label="kalman OBB proc")
plt.plot(processed_obb_perf["Frame_ID"], processed_obb_perf["Lane_X"], label="perfect kalman OBB proc")
plt.legend()

plt.subplot(3,3,8)
plt.title("v_Vel")
plt.plot(processed_hbb["Frame_ID"], processed_hbb["v_Vel"], label="kalman HBB proc")
plt.plot(processed_obb["Frame_ID"], processed_obb["v_Vel"], label="kalman OBB proc")
plt.plot(processed_obb_perf["Frame_ID"], processed_obb_perf["v_Vel"], label="perfect kalman OBB proc")
plt.legend()

plt.subplot(3,3,9)
plt.title("Cartesian X-Y")
plt.plot(processed_hbb["Cartesian_X"], processed_hbb["Cartesian_Y"], label="kalman HBB proc")
plt.plot(processed_obb["Cartesian_X"], processed_obb["Cartesian_Y"], label="kalman OBB proc")
plt.plot(processed_obb_perf["Cartesian_X"], processed_obb_perf["Cartesian_Y"], label="perfect kalman OBB proc")
plt.legend()

plt.tight_layout()
plt.show()

print(processed_obb["Frame_ID"].is_monotonic_increasing)
print(processed_obb["Lane_X"].is_monotonic_increasing)




"""
############## RUN WITHOUT RECONSTRUCTION
"""
"""
# Run with Coverage 100%
coverage=0.5 
for n in range(0,10):
    noise_angle = 1/360*(2*np.pi)
    for noise_pos in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:        
        # generate (noised) synthetic trajectory
        veh_trajectory_raw = generateSyntheticTrajectory(noise_pos, noise_angle, coverage)
        first_frame = veh_trajectory_raw["frame_nr"].iloc[0]
        last_frame = veh_trajectory_raw["frame_nr"].iloc[-1] 
        # Calculate Kalman Filtered Trajectory
        kalman_filtered_trajectory_rts_hbb = calculateKalmanFilteredTrajectory(veh_trajectory_raw, Q_k_hbb, R_k_hbb, first_frame, last_frame, video_frames_per_second, obb=False)
        kalman_filtered_trajectory_rts_obb = calculateKalmanFilteredTrajectory(veh_trajectory_raw, Q_k_obb, R_k_obb, first_frame, last_frame, video_frames_per_second, obb=True)
        kalman_filtered_trajectory_rts_obb = alignTrajectories(kalman_filtered_trajectory_rts_obb, kalman_filtered_trajectory_rts_hbb)
        # Process Trajectory
        kalman_filtered_trajectory_rts_hbb = kalman_filtered_trajectory_rts_hbb[["frame_nr", "time", "state1", "state2", "state3", "state4", "state5", "x", "y"]]
        kalman_filtered_trajectory_rts_hbb["vehicle"] = "VEHICLE_1"
        kalman_filtered_trajectory_rts_hbb = kalman_filtered_trajectory_rts_hbb.rename(columns={"x": "x_cartesian", "y": "y_cartesian"})
        processed_hbb = processTrajectory_synthetic(kalman_filtered_trajectory_rts_hbb)
        kalman_filtered_trajectory_rts_obb = kalman_filtered_trajectory_rts_obb[["frame_nr", "time", "state1", "state2", "state3", "state4", "state5", "x", "y"]]
        kalman_filtered_trajectory_rts_obb["vehicle"] = "VEHICLE_1"
        kalman_filtered_trajectory_rts_obb = kalman_filtered_trajectory_rts_obb.rename(columns={"x": "x_cartesian", "y": "y_cartesian"})
        processed_obb = processTrajectory_synthetic(kalman_filtered_trajectory_rts_obb)
        # Evaluate Trajectory
        e_s_max, e_s_min, e_s_avg1, std1, b = calculateInternalConsistency(processed_hbb, video_frames_per_second)
        e_s_max, e_s_min, e_s_avg2, std2, c = calculateInternalConsistency(processed_obb, video_frames_per_second)       
        improvement = e_s_avg1-e_s_avg2
        # Print result
        print(coverage, noise_angle, noise_pos, e_s_avg1, e_s_avg2, improvement)
    
"""




"""
############## RUN WITH RECONSTRUCTION
"""

# Run with Coverage 100% but Reconstructed Trajectories
import pickle
vehicle_dynamics_path = "../data_benchmark/5_vehicle_information/vehicle_dynamics/"
with open(vehicle_dynamics_path+"accel_capacity_interpolator.pkl", "rb") as f:
    accel_max_spl = pickle.load(f)
with open(vehicle_dynamics_path+"decel_capacity_interpolator.pkl", "rb") as f:
    decel_min_spl = pickle.load(f)

coverage=1.0
for n in range(0,10):
    noise_angle = 1/360*(2*np.pi)
    for noise_pos in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:        
        # generate (noised) synthetic trajectory
        veh_trajectory_raw = generateSyntheticTrajectory(noise_pos, noise_angle, coverage)
        first_frame = veh_trajectory_raw["frame_nr"].iloc[0]
        last_frame = veh_trajectory_raw["frame_nr"].iloc[-1] 
        # Calculate Kalman Filtered Trajectory
        kalman_filtered_trajectory_rts_hbb = calculateKalmanFilteredTrajectory(veh_trajectory_raw, Q_k_hbb, R_k_hbb, first_frame, last_frame, video_frames_per_second, obb=False)
        kalman_filtered_trajectory_rts_obb = calculateKalmanFilteredTrajectory(veh_trajectory_raw, Q_k_obb, R_k_obb, first_frame, last_frame, video_frames_per_second, obb=True)
        kalman_filtered_trajectory_rts_obb = alignTrajectories(kalman_filtered_trajectory_rts_obb, kalman_filtered_trajectory_rts_hbb)
        # Process Trajectory
        kalman_filtered_trajectory_rts_hbb = kalman_filtered_trajectory_rts_hbb[["frame_nr", "time", "state1", "state2", "state3", "state4", "state5", "x", "y"]]
        kalman_filtered_trajectory_rts_hbb["vehicle"] = "VEHICLE_1"
        kalman_filtered_trajectory_rts_hbb = kalman_filtered_trajectory_rts_hbb.rename(columns={"x": "x_cartesian", "y": "y_cartesian"})
        processed_hbb = processTrajectory_synthetic(kalman_filtered_trajectory_rts_hbb)
        kalman_filtered_trajectory_rts_obb = kalman_filtered_trajectory_rts_obb[["frame_nr", "time", "state1", "state2", "state3", "state4", "state5", "x", "y"]]
        kalman_filtered_trajectory_rts_obb["vehicle"] = "VEHICLE_1"
        kalman_filtered_trajectory_rts_obb = kalman_filtered_trajectory_rts_obb.rename(columns={"x": "x_cartesian", "y": "y_cartesian"})
        processed_obb = processTrajectory_synthetic(kalman_filtered_trajectory_rts_obb)
        # Reconstruct Trajectory
        processed_hbb = reconstruct_trajectories_cvxopt(processed_hbb, accel_max_spl, decel_min_spl)
        processed_obb = reconstruct_trajectories_cvxopt(processed_obb, accel_max_spl, decel_min_spl)
        # Evaluate Trajectory
        e_s_max, e_s_min, e_s_avg1, std1, b = calculateInternalConsistency(processed_hbb, video_frames_per_second)
        e_s_max, e_s_min, e_s_avg2, std2, c = calculateInternalConsistency(processed_obb, video_frames_per_second)       
        improvement = e_s_avg1-e_s_avg2
        # Print result
        print(coverage, noise_angle, noise_pos, e_s_avg1, e_s_avg2, improvement)
