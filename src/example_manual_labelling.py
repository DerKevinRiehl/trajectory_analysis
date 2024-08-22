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
from tools_annotations import loadAnnotations, saveAnnotations
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


# #############################################################################
# CONSTANTS
# #############################################################################
RELEVANT_FRAME = 0
# RELEVANT_VIDEO = "DJI_0933.MOV"
RELEVANT_VIDEO = "DJI_0934.MOV"
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




# #############################################################################
# STEP 2: TRAJECTORY GENERATION
# #############################################################################

processed_annotations = loadAnnotations("../data/2_frame_processed/"+RELEVANT_VIDEO+".txt")
trajectorized_annotations = loadAnnotations("../data/3_A_trajectorized_unlabelled/"+RELEVANT_VIDEO+".txt")
unique_trajectory_labels = determineUniqueTrajectoryLabels(trajectorized_annotations)
trajectory_vehicle_map = loadTrajectoryLabelVehicleMap("../data/3_B_trajectorized_mapping/"+RELEVANT_VIDEO+".txt")


def map_trajectories_to_vehicles(trajectorized_annotations, trajectory_vehicle_map):
    vehiclized_annotations = {}
    vehiclized_annotations[RELEVANT_FRAME] = []
    for annotation in trajectorized_annotations[RELEVANT_FRAME]:
        traj = annotation[-1]
        veh = trajectory_vehicle_map[traj]
        if veh == "REMOVE":
            continue
        #annotation.append(veh)
        vehiclized_annotations[RELEVANT_FRAME].append(annotation + [veh])
    return vehiclized_annotations


vehiclized_annotations = map_trajectories_to_vehicles(trajectorized_annotations, trajectory_vehicle_map)

# #############################################################################
# EXEMPLARY VISUALIZATION OF TRAJECTORY LABELS
# #############################################################################
from tools_video import extractFrameFromVideo, renderAnnotatedFrame
# VIDEO
succsss, frame = extractFrameFromVideo(video_file_path, RELEVANT_FRAME)
# ANNOTATIONS
frame_annotations = processed_annotations[RELEVANT_FRAME]
# HOMOGRAPHY
frame_homography = getFrameHomography(df_homography, RELEVANT_FRAME)
# REGION OF INTEREST
frame_region_of_interest = getTransformedRegionOfInterest(region_of_interest, df_homography, RELEVANT_FRAME)

drawing_settings = default_drawing_settings.copy()
elements = {
    "homography": frame_homography,
    "region_of_interest": frame_region_of_interest,
    #"vehicle_annotations": frame_annotations, 
    "labelled_vehicle_annnotations": transformAnnotations_CARTESIAN_2_PIX(vehiclized_annotations[RELEVANT_FRAME], frame_homography),
}
frame_finished = renderAnnotatedFrame(frame, elements, design=drawing_settings)
    
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(frame_finished)
plt.show(block=True)