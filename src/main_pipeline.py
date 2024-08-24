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
import numpy as np




# #############################################################################
# CONSTANTS
# #############################################################################
#RELEVANT_FRAME = 2000
RELEVANT_VIDEO = "DJI_0933.MOV"
# RELEVANT_VIDEO = "DJI_0934.MOV"
# RELEVANT_VIDEO = "DJI_0939.MOV"
# RELEVANT_VIDEO = "DJI_0940.MOV"
# RELEVANT_VIDEO = "DJI_0943.MOV"
# RELEVANT_VIDEO = "DJI_0944.MOV"




# #############################################################################
# LOADING - FILES
# #############################################################################
# VIDEO
video_file_path = video_path+RELEVANT_VIDEO
num_frames =  getNumberOfFramesFromVideo(video_file_path)
# ANNOTATIONS
inf_file = inference_annotations_path+RELEVANT_VIDEO+".zip"
annotations = loadAnnotations(inf_file)
# HOMOGRAPHY
df_homography = loadHomography("../data/1_homography/"+RELEVANT_VIDEO+"_circle.txt")
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
saveAnnotations("../data/2_frame_processed/"+RELEVANT_VIDEO+".txt", processed_annotations)
saveAnnotations("../data/2_frame_processed/"+RELEVANT_VIDEO+"_PIX.txt", processed_annotations_pix)

elements = {
    "homography": df_homography,
    "region_of_interest": region_of_interest,
    "vehicle_annotations": processed_annotations_pix,
}
drawing_settings = default_drawing_settings.copy()
drawing_settings["vehicle_annotations"]["draw_alpha"] = 1.0
drawing_settings["vehicle_annotations"]["line_width"] = 1.5
renderAnnotatedVideo(video_file_path_source=video_file_path, 
                     video_file_path_destination="../videos/2_frame_processed/"+"Test_"+RELEVANT_VIDEO, 
                     elements=elements, 
                     design=drawing_settings, 
                     max_num_frames=None, 
                     print_status=True)
"""




"""
# #############################################################################
# STEP 2: TRAJECTORY GENERATION
# #############################################################################

processed_annotations = loadAnnotations("../data/2_frame_processed/"+RELEVANT_VIDEO+".txt")
trajectorized_annotations = generateTrajectories(processed_annotations, print_status=True)
saveAnnotations("../data/3_A_trajectorized_unlabelled/"+RELEVANT_VIDEO+".txt", trajectorized_annotations)

unique_trajectory_labels = determineUniqueTrajectoryLabels(trajectorized_annotations)

map_file = "../data/3_B_trajectorized_mapping/"+RELEVANT_VIDEO+".txt"
generateEmptyTrajectoryLabelVehicleMap(map_file, unique_trajectory_labels)
# HERE YOU NEED TO MANUALLY EDIT THE MAP_FILE BEFORE LOADING
trajectory_vehicle_map = loadTrajectoryLabelVehicleMap(map_file)
# TODO: WRITE A FUNCTION THAT CREATES ANOTHER COLUMN IN ANNOTATIONS WHICH USES MAP TO HAVE VEHICLE LABELS
# see manual_labelling_app.py
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
annotation_file = "../data/3_C_vehiclized/"+RELEVANT_VIDEO+".txt"
target_output_file = "../data/4_kalman_filtered/"+RELEVANT_VIDEO+"_"

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
    kalman_filtered_trajectory_rts_hbb.to_csv(target_output_file+selected_vehicle+"_hbb.csv",index=False)
    if obb_mode:
        kalman_filtered_trajectory_rts_obb.to_csv(target_output_file+selected_vehicle+"_obb.csv",index=False)


# Code to display raw and Kalman filtered trajectories to compare
# E.G: VEHICLE_4
import pandas as pd
selected_time_frames = [0, 7517+1]
selected_vehicle = "VEHICLE_4"
annotation_file = "../data/3_C_vehiclized/"+RELEVANT_VIDEO+".txt"
df_raw = loadAnnotationsForFiltering(annotation_file, selected_vehicle, selected_time_frames)
df_filtered_hbb = pd.read_csv("../data/4_kalman_filtered/"+RELEVANT_VIDEO+"_"+selected_vehicle+"_hbb"+".csv")
df_filtered_obb = pd.read_csv("../data/4_kalman_filtered/"+RELEVANT_VIDEO+"_"+selected_vehicle+"_obb"+".csv")

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(df_raw["frame_nr"], df_raw["x"], label="raw")
plt.plot(df_filtered_hbb["frame_nr"], df_filtered_hbb["x"], label="HBB filtered")
plt.plot(df_filtered_obb["frame_nr"], df_filtered_obb["x"], label="OBB filtered")
plt.legend()
plt.subplot(1,2,2)
plt.plot(df_raw["frame_nr"], df_raw["y"], label="raw")
plt.plot(df_filtered_hbb["frame_nr"], df_filtered_hbb["y"], label="HBB filtered")
plt.plot(df_filtered_obb["frame_nr"], df_filtered_obb["y"], label="OBB filtered")
plt.legend()