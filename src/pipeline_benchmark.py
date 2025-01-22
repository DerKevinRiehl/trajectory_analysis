"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Kevin Riehl, Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024
Submitted to:   JOURNAL
-------------------------------------------
"""


# #############################################################################
# IMPORTS
# #############################################################################
from tools_video import getNumberOfFramesFromVideo
from tools_annotations import loadAnnotations, saveAnnotations, loadAnnotationsForFiltering, loadUniqueVehicles
from _constants import video_path
from _constants import inference_annotations_path
from _constants import REGION_OF_INTEREST
from _constants import default_drawing_settings
from tools_homography import loadHomography, getFrameHomography, getTransformedRegionOfInterest
from tools_homography import transformAnnotations_CARTESIAN_2_PIX
from tools_frame_processing import processFrameAnnotations
from tools_video import renderAnnotatedVideo
from tools_trajectorization import generateTrajectories, determineUniqueTrajectoryLabels
from tools_trajectorization import generateEmptyTrajectoryLabelVehicleMap, loadTrajectoryLabelVehicleMap
from tools_filtering import calculateKalmanFilteredTrajectory, alignTrajectories, featureCalculation
from tools_trajectory_processing import processTrajectory
from tools_trajectory_filtering import reconstruct_trajectories_cvxopt
import numpy as np
import pickle
import pandas as pd
import os




# #############################################################################
# CONSTANTS
# #############################################################################
RELEVANT_VIDEO = "DJI_0933.MOV"

models = [
    "Inference_cfa_r50_fpn_40e_dota_oc",
    "Inference_faster-rcnn_r50_fpn_hbb_DOTA",
    "Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90",
    "Inference_redet_re50_refpn_1x_dota_ms_rr_le90",
    "Inference_retinanet_r50_fpn_hbb_DOTA",
    "Inference_roi_trans_r50_fpn_1x_dota_ms_le90",
    "Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90",
    "Inference_s2anet_r50_fpn_fp16_1x_dota_le135",
    "Inference_yolo_l_hbb",
    "Inference_yolo_l_obb",
    "Inference_yolo_m_hbb",
    "Inference_yolo_m_obb",
    "Inference_yolo_n_hbb",
    "Inference_yolo_n_obb",
    "Inference_yolo_s_hbb",
    "Inference_yolo_s_obb",
    "Inference_yolo_x_hbb",
    "Inference_yolo_x_obb",
]

RELEVANT_MODEL = models[0]
RELEVANT_MODEL = models[17]

RELEVANT_MODEL = models[0]

# #############################################################################
# LOADING - FILES
# #############################################################################
# VIDEO
video_file_path = video_path+RELEVANT_VIDEO
num_frames =  getNumberOfFramesFromVideo(video_file_path)
# ANNOTATIONS
annotations = loadAnnotations("../data_benchmark/1_annotations/"+RELEVANT_MODEL+".zip")
# HOMOGRAPHY
df_homography = loadHomography("../data_benchmark/1_homography/"+RELEVANT_VIDEO+"_circle.txt")
# REGION OF INTEREST
region_of_interest = REGION_OF_INTEREST[RELEVANT_VIDEO]



"""
# #############################################################################
# STEP 1: SINGLE FRAME PROCESSING
# #############################################################################
processed_annotations = {}
processed_annotations_pix = {}
for frame_counter in range(0, num_frames):
    frame_annotations = annotations[frame_counter]
    frame_homography = getFrameHomography(df_homography, frame_counter)
    frame_region_of_interest = getTransformedRegionOfInterest(region_of_interest, df_homography, frame_counter)
    processed_frame_annotations = processFrameAnnotations(frame_annotations, frame_region_of_interest, frame_homography)
    processed_annotations[frame_counter] = processed_frame_annotations
    processed_annotations_pix[frame_counter] = transformAnnotations_CARTESIAN_2_PIX(processed_frame_annotations, frame_homography)
saveAnnotations("../data_benchmark/2_frame_processed/"+RELEVANT_MODEL+".txt", processed_annotations)
"""




"""
# #############################################################################
# STEP 2: TRAJECTORY GENERATION
# #############################################################################

processed_annotations = loadAnnotations("../data_benchmark/2_frame_processed/"+RELEVANT_MODEL+".txt")
trajectorized_annotations = generateTrajectories(processed_annotations, print_status=True)
saveAnnotations("../data_benchmark/3_A_trajectorized_unlabelled/"+RELEVANT_MODEL+".txt", trajectorized_annotations)

unique_trajectory_labels = determineUniqueTrajectoryLabels(trajectorized_annotations)

map_file = "../data_benchmark/3_B_trajectorized_mapping/"+RELEVANT_MODEL+".txt"
generateEmptyTrajectoryLabelVehicleMap(map_file, unique_trajectory_labels)
# trajectory_vehicle_map = loadTrajectoryLabelVehicleMap(map_file)
# see 3_B_trajectorized_mapping/mapper/automatic_mapping.py
# see 3_C_vehiclized/labeller/automatic_labeller.py
"""



"""
# #############################################################################
# STEP 3: EXTENDED KALMAN FILTERING
# #############################################################################

# Determine Kalman Filter Parameters
Q_k_hbb = np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) # P_0.copy() # covariance matrix of error of state
R_k_hbb = np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) # P_0.copy() # covariance matrix of error of output
Q_k_obb = np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) # P_0.copy() # covariance matrix of error of state
R_k_obb = np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) # P_0.copy() # covariance matrix of error of output

# Determine Video parameters
obb_mode = True
selected_time_frames = [0, 7517+1]
video_frames_per_second = 25

# Determine File parameters
annotation_file = "../data_benchmark/3_C_vehiclized/"+RELEVANT_MODEL+".txt"
target_output_folder = "../data_benchmark/4_kalman_filtered/"+RELEVANT_MODEL.replace(".txt", "")+"/"

unique_vehicles = loadUniqueVehicles(annotation_file)

for selected_vehicle in unique_vehicles:
    if selected_vehicle == "UNDEFINED":
        continue
    print(annotation_file, selected_vehicle, "...")    
        # Load Raw Data
    veh_trajectory_raw = loadAnnotationsForFiltering(annotation_file, selected_vehicle, selected_time_frames)
    veh_trajectory_raw = featureCalculation(veh_trajectory_raw, video_frames_per_second, obb=obb_mode)
    veh_trajectory_raw = veh_trajectory_raw.reset_index()
    del veh_trajectory_raw["index"]
    first_frame = veh_trajectory_raw["frame_nr"].iloc[0]
    last_frame = veh_trajectory_raw["frame_nr"].iloc[-1]  

        # Calculate Kalman Filtered Trajectory
    kalman_filtered_trajectory_rts_hbb = calculateKalmanFilteredTrajectory(veh_trajectory_raw, Q_k_hbb, R_k_hbb, first_frame, last_frame, video_frames_per_second, obb=False)
    if obb_mode:
        kalman_filtered_trajectory_rts_obb = calculateKalmanFilteredTrajectory(veh_trajectory_raw, Q_k_obb, R_k_obb, first_frame, last_frame, video_frames_per_second, obb=True)
        kalman_filtered_trajectory_rts_obb = alignTrajectories(kalman_filtered_trajectory_rts_obb, kalman_filtered_trajectory_rts_hbb)
        
        # Save Calculated Trajectories
    os.makedirs(target_output_folder, exist_ok=True)
    kalman_filtered_trajectory_rts_hbb.to_csv(target_output_folder+selected_vehicle+"_hbb.csv",index=False)
    if obb_mode:
        kalman_filtered_trajectory_rts_obb.to_csv(target_output_folder+selected_vehicle+"_obb.csv",index=False)
"""



"""
# #############################################################################
# STEP 4: TRAJECTORY PROCESSING
# #############################################################################

video_trajectory_path = "../data_benchmark/4_kalman_filtered/"+RELEVANT_MODEL+"/"
vehiclized_file_path = "../data_benchmark/3_C_vehiclized/"+RELEVANT_MODEL+".txt"
vehicle_proceeding_order_file_path = "../data_benchmark/5_vehicle_information/proceeding_order/"+RELEVANT_VIDEO+".txt"
first_vehicle = "VEHICLE_1"

target_output_file = "../data_benchmark/6_final_trajectories/"+RELEVANT_MODEL+"_OBB.txt"
trajectory_type = "obb"
final_trajectory_df = processTrajectory(video_trajectory_path, vehiclized_file_path, 
                  vehicle_proceeding_order_file_path, trajectory_type, first_vehicle)
final_trajectory_df.to_csv(target_output_file, index=False)

target_output_file = "../data_benchmark/6_final_trajectories/"+RELEVANT_MODEL+"_HBB.txt"
trajectory_type = "hbb"
final_trajectory_df = processTrajectory(video_trajectory_path, vehiclized_file_path, 
                  vehicle_proceeding_order_file_path, trajectory_type, first_vehicle)
final_trajectory_df.to_csv(target_output_file, index=False)
# """



# """
# #############################################################################
# STEP 5: TRAJECTORY RECONSTRUCTION
# #############################################################################

vehicle_trajectory_final_path = "../data_benchmark/6_final_trajectories/"
vehicle_dynamics_path = "../data_benchmark/5_vehicle_information/vehicle_dynamics/"
vehicle_trajectory_reconstructed_path = "../data_benchmark/7_final_trajectories_reconstructed/"

with open(vehicle_dynamics_path+"accel_capacity_interpolator.pkl", "rb") as f:
    accel_max_spl = pickle.load(f)
with open(vehicle_dynamics_path+"decel_capacity_interpolator.pkl", "rb") as f:
    decel_min_spl = pickle.load(f)

trajectory_type = "obb"
df_final_traj = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+trajectory_type+".txt", sep=",")
df_reconst_traj = reconstruct_trajectories_cvxopt(df_final_traj, accel_max_spl, decel_min_spl)
df_reconst_traj.to_csv(vehicle_trajectory_reconstructed_path+RELEVANT_MODEL+"_"+trajectory_type+".txt", index=False)
trajectory_type = "hbb"
df_final_traj = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+trajectory_type+".txt", sep=",")
df_reconst_traj = reconstruct_trajectories_cvxopt(df_final_traj, accel_max_spl, decel_min_spl)
df_reconst_traj.to_csv(vehicle_trajectory_reconstructed_path+RELEVANT_MODEL+"_"+trajectory_type+".txt", index=False)
# """