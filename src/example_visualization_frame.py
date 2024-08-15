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
from tools_video import getNumberOfFramesFromVideo, extractFrameFromVideo
import matplotlib.pyplot as plt
from tools_annotations import loadAnnotations
from _constants import inference_annotations_path
from _constants import REGION_OF_INTEREST
from _constants import default_drawing_settings
from tools_homography import loadHomography, getFrameHomography, getTransformedRegionOfInterest




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
inference_file = inference_annotations_path+RELEVANT_VIDEO+".zip"
annotations = loadAnnotations(inference_file)
# HOMOGRAPHY
df_homography = loadHomography("../data/1_homography/"+RELEVANT_VIDEO+"_circle.txt")
# REGION OF INTEREST
region_of_interest = REGION_OF_INTEREST[RELEVANT_VIDEO]




# #############################################################################
# LOADING - FRAME
# #############################################################################
# VIDEO
succsss, frame = extractFrameFromVideo(video_file_path, RELEVANT_FRAME)
# ANNOTATIONS
frame_annotations = annotations[RELEVANT_FRAME]
# HOMOGRAPHY
frame_homography = getFrameHomography(df_homography, RELEVANT_FRAME)
# REGION OF INTEREST
frame_region_of_interest = getTransformedRegionOfInterest(region_of_interest, df_homography, RELEVANT_FRAME)




# #############################################################################
# DEFINITION OF INPUT
# #############################################################################
elements = {
    "homography": frame_homography,
    "region_of_interest": frame_region_of_interest,
    "vehicle_annotations": frame_annotations,
}
drawing_settings = default_drawing_settings.copy()




# #############################################################################
# RENDER & DISPLAY IMAGE
# #############################################################################
from tools_video import renderAnnotatedFrame
paint_frame = renderAnnotatedFrame(frame, elements, drawing_settings)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(frame)
plt.subplot(1,2,2)
plt.imshow(paint_frame)

