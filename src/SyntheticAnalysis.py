# #############################################################################
# ########## IMPORTS
# #############################################################################
import numpy as np
import pandas as pd
from tools_filtering_angle import boundAnglePositive
from tools_filtering import calculateKalmanFilteredTrajectory, alignTrajectories, featureCalculation
from tools_trajectory_evaluation import calculateInternalConsistency
from tools_trajectory_processing import processTrajectory_synthetic
from tools_trajectory_filtering import reconstruct_trajectories_cvxopt
import warnings
warnings.filterwarnings('ignore')




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
    for frame_nr in range(0, 7500):
        delta_time = 1/25
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
    