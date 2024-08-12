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
from _constants import inference_annotations_path
from _constants import REGION_OF_INTEREST
from tools_homography import loadHomography, getFrameHomography, getTransformedRegionOfInterest
from tools_frame_processing import processFrameAnnotations



# #############################################################################
# CONSTANTS
# #############################################################################
RELEVANT_FRAME = 2000
RELEVANT_VIDEO = "DJI_0933.MOV"




# #############################################################################
# LOADING - FILES
# #############################################################################
# VIDEO
video_file_path = "C:/VIDEO_ETH/"+RELEVANT_VIDEO
num_frames =  getNumberOfFramesFromVideo(video_file_path)
# ANNOTATIONS
inf_file = inference_annotations_path+RELEVANT_VIDEO+".zip"
annotations = loadAnnotations(inf_file)
# HOMOGRAPHY
df_homography = loadHomography("../data/1_homography/"+RELEVANT_VIDEO+"_circle.txt")
# REGION OF INTEREST
region_of_interest = REGION_OF_INTEREST[RELEVANT_VIDEO]




# #############################################################################
# STEP 1: SINGLE FRAME PROCESSING
# #############################################################################
processed_annotations = {}
frame_counter = 0
for frame_counter in range(0, num_frames):
    frame_annotations = annotations[RELEVANT_FRAME]
    frame_homography = getFrameHomography(df_homography, RELEVANT_FRAME)
    frame_region_of_interest = getTransformedRegionOfInterest(region_of_interest, df_homography, RELEVANT_FRAME)
    processed_frame_annotations = processFrameAnnotations(frame_annotations, frame_region_of_interest, frame_homography)
    processed_annotations[frame_counter] = processed_frame_annotations

saveAnnotations("../data/2_frame_processed/"+RELEVANT_VIDEO+".txt", processed_annotations)
