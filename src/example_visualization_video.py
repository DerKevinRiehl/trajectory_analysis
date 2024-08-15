"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Kevin Riehl, Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024
Submitted to:   JOURNAL
-------------------------------------------
This example shows how to render a frame with different annotations.
"""


# #############################################################################
# IMPORTS
# #############################################################################
from tools_video import getNumberOfFramesFromVideo
from tools_annotations import loadAnnotations
from _constants import inference_annotations_path
from _constants import REGION_OF_INTEREST
from _constants import default_drawing_settings
from tools_homography import loadHomography




# #############################################################################
# CONSTANTS
# #############################################################################
RELEVANT_FRAME = 2000
RELEVANT_VIDEO = "DJI_0933.MOV"
video_file_path = "C:/VIDEO_ETH/"+RELEVANT_VIDEO




# #############################################################################
# LOADING - FILES
# #############################################################################
# VIDEO
num_frames =  getNumberOfFramesFromVideo(video_file_path)
# ANNOTATIONS
inf_file = inference_annotations_path+RELEVANT_VIDEO+".zip"
annotations = loadAnnotations(inf_file)
# HOMOGRAPHY
df_homography = loadHomography("../data/1_homography/"+RELEVANT_VIDEO+"_circle.txt")
# REGION OF INTEREST
region_of_interest = REGION_OF_INTEREST[RELEVANT_VIDEO]




# #############################################################################
# DEFINITION OF INPUT
# #############################################################################
elements = {
    "homography": df_homography,
    "region_of_interest": region_of_interest,
    "vehicle_annotations": annotations,
}
drawing_settings = default_drawing_settings.copy()




# #############################################################################
# RENDER VIDEO
# #############################################################################
from tools_video import renderAnnotatedVideo
renderAnnotatedVideo(video_file_path_source=video_file_path, 
                     video_file_path_destination="Test2.mov", 
                     elements=elements, 
                     design=drawing_settings, 
                     max_num_frames=1000,#100,#None, 
                     print_status=True)
