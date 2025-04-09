"""
Consistent Vehicle Trajectory Extraction From Aerial Recordings Using Oriented Object Detection
-------------------------------------------
Authors:        Kevin Riehl, Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024 - 2025
Submitted to:   Scientific Reports
-------------------------------------------
This example shows how to render a frame with different annotations.
"""


# #############################################################################
# IMPORTS
# #############################################################################
from tools_video import getNumberOfFramesFromVideo
from tools_annotations import loadAnnotations
from _constants import inference_annotations_path, trajectorized_labelled_path, trajectorized_unlabelled_path
from _constants import REGION_OF_INTEREST
from _constants import default_drawing_settings
from tools_homography import loadHomography, transformAnnotations_CARTESIAN_2_PIX, getFrameHomography
import pandas as pd
import numpy as np



# #############################################################################
# CONSTANTS
# #############################################################################
RELEVANT_VIDEO = "DJI_0933.MOV"
video_file_path = "C:/VIDEO_ETH/"+RELEVANT_VIDEO
selected_vehicle = "VEHICLE_1"
selected_kalman = "_obb"
zoom_factor = 3.5
history_horizon = 100



# #############################################################################
# LOADING - FILES
# #############################################################################
# VIDEO
num_frames =  getNumberOfFramesFromVideo(video_file_path)
# ANNOTATIONS
inf_file = inference_annotations_path+RELEVANT_VIDEO+".zip"
annotations = loadAnnotations(inf_file)
filt_file = "../data/2_frame_processed/"+RELEVANT_VIDEO+".txt"
annotations_filtered = loadAnnotations(inf_file)
# LABELLED ANNOTATIONS
labelled_file = trajectorized_unlabelled_path+RELEVANT_VIDEO+".txt"
annotations_labelled = loadAnnotations(labelled_file)
# HOMOGRAPHY
df_homography = loadHomography("../data/1_homography/"+RELEVANT_VIDEO+"_circle.txt")
annotations_filtered_pix = {}
annotations_labelled_pix = {}
for frame in annotations_labelled:
    frame_homography = getFrameHomography(df_homography, frame)
    # filtered
    frame_annotations = annotations_filtered[frame]
    frame_annotations_pix = transformAnnotations_CARTESIAN_2_PIX(frame_annotations, frame_homography)
    annotations_filtered_pix[frame] = frame_annotations_pix
    # labelled
    frame_labelled_annotations = annotations_labelled[frame]
    frame_labelled_annotations_pix = transformAnnotations_CARTESIAN_2_PIX(frame_labelled_annotations, frame_homography)
    annotations_labelled_pix[frame] = frame_labelled_annotations_pix
# REGION OF INTEREST
region_of_interest = REGION_OF_INTEREST[RELEVANT_VIDEO]
# KALMAN FILTERED
df_kalman_vehicle_data = pd.read_csv("../data/4_kalman_filtered/"+RELEVANT_VIDEO.replace(".MOV", "")+"/"+selected_vehicle+selected_kalman+".csv")
zoom_factor_array = zoom_factor*np.ones(num_frames)
zoom_factor_array = zoom_factor_array.tolist()
df_trajectory_data = pd.read_csv("../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt")
df_trajectory_data = df_trajectory_data[df_trajectory_data["Vehicle_ID"]==selected_vehicle]
df_space_headway = df_trajectory_data[["Frame_ID", "Space_Hdwy"]]
# FINAL TRAJECTORY
df_final_trajectory = pd.read_csv("../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt")


# #############################################################################
# DEFINITION OF INPUT
# #############################################################################
elements = {
    "homography": df_homography,
    "region_of_interest": region_of_interest,
    # "vehicle_annotations": annotations,
    "vehicle_annotations": annotations_filtered_pix,
    "labelled_vehicle_annnotations": annotations_labelled_pix,
    "labelled_final_trajectory": df_final_trajectory,
    "transformation": {"kalman": df_kalman_vehicle_data, "zoom": zoom_factor_array, "norotation": True},
    "hud": {"headway": df_space_headway, "history": df_trajectory_data, "horizon":history_horizon}
}
drawing_settings = default_drawing_settings.copy()



# #############################################################################
# RENDER VIDEO
# #############################################################################
from tools_video import renderAnnotatedVideo
renderAnnotatedVideo(video_file_path_source=video_file_path, 
                     video_file_path_destination="ai_center_video/EVideo_7_kalmanannotations.mov", 
                     elements=elements, 
                     design=drawing_settings, 
                     start_frame=None,
                     end_frame=2000,#100,#None, 
                     print_status=True)
