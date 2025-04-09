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
from tools_video import getNumberOfFramesFromVideo, extractFrameFromVideo
import matplotlib.pyplot as plt
import pandas as pd
from tools_annotations import loadAnnotations
from _constants import inference_annotations_path, trajectorized_labelled_path
from _constants import REGION_OF_INTEREST
from _constants import default_drawing_settings
from tools_homography import loadHomography, getFrameHomography, getTransformedRegionOfInterest
from tools_homography import transformPointFrom_CARTESIAN_2_PIX, transformAnnotations_CARTESIAN_2_PIX



# #############################################################################
# CONSTANTS
# #############################################################################
RELEVANT_FRAME = 3000
RELEVANT_VIDEO = "DJI_0933.MOV"
video_file_path = "C:/VIDEO_ETH/"+RELEVANT_VIDEO
selected_vehicle = "VEHICLE_1"
selected_kalman = "_obb"
zoom_factor = 5
history_horizon = 100



# #############################################################################
# LOADING - FILES
# #############################################################################
# VIDEO
num_frames =  getNumberOfFramesFromVideo(video_file_path)
# ANNOTATIONS
inference_file = inference_annotations_path+RELEVANT_VIDEO+".zip"
annotations = loadAnnotations(inference_file)
# LABELLED ANNOTATIONS
labelled_file = trajectorized_labelled_path+RELEVANT_VIDEO+".txt"
annotations_labelled = loadAnnotations(labelled_file)
# HOMOGRAPHY
df_homography = loadHomography("../data/1_homography/"+RELEVANT_VIDEO+"_circle.txt")
# REGION OF INTEREST
region_of_interest = REGION_OF_INTEREST[RELEVANT_VIDEO]
# KALMAN FILTERED
df_kalman_vehicle_data = pd.read_csv("../data/4_kalman_filtered/"+RELEVANT_VIDEO.replace(".MOV", "")+"/"+selected_vehicle+selected_kalman+".csv")
df_trajectory_data = pd.read_csv("../data/6_final_trajectories/"+RELEVANT_VIDEO+".txt")
df_trajectory_data = df_trajectory_data[df_trajectory_data["Vehicle_ID"]==selected_vehicle]
df_space_headway = df_trajectory_data[["Frame_ID", "Space_Hdwy"]]



# #############################################################################
# LOADING - FRAME
# #############################################################################
# VIDEO
succsss, frame = extractFrameFromVideo(video_file_path, RELEVANT_FRAME)
# ANNOTATIONS
frame_annotations = annotations[RELEVANT_FRAME]
frame_labelled_annotations = annotations_labelled[RELEVANT_FRAME]
# HOMOGRAPHY
frame_homography = getFrameHomography(df_homography, RELEVANT_FRAME)
frame_labelled_annotations2 = transformAnnotations_CARTESIAN_2_PIX(frame_labelled_annotations, frame_homography)
# REGION OF INTEREST
frame_region_of_interest = getTransformedRegionOfInterest(region_of_interest, df_homography, RELEVANT_FRAME)
# KALMAN FILTERED
df_vehicle_frame = df_kalman_vehicle_data[df_kalman_vehicle_data["frame_nr"]==RELEVANT_FRAME]
kalman_annotations = df_vehicle_frame.iloc[0].tolist()
kalman_coordinate = kalman_annotations[4:5+1]
kalman_frame_coordinates = transformPointFrom_CARTESIAN_2_PIX(kalman_coordinate, frame_homography)
kalman_frame_angle = kalman_annotations[8]
space_headway = df_space_headway[df_space_headway["Frame_ID"]==RELEVANT_FRAME]["Space_Hdwy"].iloc[0]
# HISTORY
df_history = df_trajectory_data[df_trajectory_data["Frame_ID"]<=RELEVANT_FRAME]
df_history = df_history[df_history["Frame_ID"]>=RELEVANT_FRAME-history_horizon]
positions_transformed_x = []
positions_transformed_y = []
for idx, row in df_history.iterrows():
    pos = [row["Cartesian_X"], row["Cartesian_Y"]]
    posT = transformPointFrom_CARTESIAN_2_PIX(pos, frame_homography)
    positions_transformed_x.append(posT[0])
    positions_transformed_y.append(posT[1])
df_history["x_pixel"] = positions_transformed_x
df_history["y_pixel"] = positions_transformed_y



# #############################################################################
# DEFINITION OF INPUT
# #############################################################################
elements = {
    "homography": frame_homography,
    "region_of_interest": frame_region_of_interest,
    "vehicle_annotations": frame_annotations,
    "labelled_vehicle_annnotations": frame_labelled_annotations2
}
drawing_settings = default_drawing_settings.copy()



# #############################################################################
# RENDER & DISPLAY IMAGE
# #############################################################################
from tools_video import renderAnnotatedFrame, renderTransformedFrame, getVehicleEgoPerspectiveTransformation
from tools_video import renderHUDAnnotatedFrame

paint_frame = renderAnnotatedFrame(frame, elements, drawing_settings)
transformation = getVehicleEgoPerspectiveTransformation(frame, kalman_frame_coordinates, kalman_frame_angle, zoom_factor)
paint_frame2 = renderTransformedFrame(paint_frame, transformation)
paint_frame3 = renderHUDAnnotatedFrame(paint_frame2, kalman_annotations, df_history, space_headway, history_horizon)

plt.figure()
plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(frame)
plt.subplot(2,2,2)
plt.title("Annotated Image")
plt.imshow(paint_frame)
plt.subplot(2,2,3)
plt.title("Annotated Image + EGO Vehicle 1")
plt.imshow(paint_frame2)
plt.subplot(2,2,4)
plt.title("EGO + HUD Visual")
plt.imshow(paint_frame3)
