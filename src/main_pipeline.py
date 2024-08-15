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



# #############################################################################
# CONSTANTS
# #############################################################################
#RELEVANT_FRAME = 2000
RELEVANT_VIDEO = "DJI_0933.MOV"




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