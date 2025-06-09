"""
Consistent Vehicle Trajectory Extraction From Aerial Recordings Using Oriented Object Detection
-------------------------------------------
Authors:        Kevin Riehl, Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024 - 2025
Submitted to:   Scientific Reports
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from _constants import video_path, homography_path, kalman_filtered_path
from _constants import inference_annotations_path, trajectorized_labelled_path
from _constants import REGION_OF_INTEREST
from _constants import default_drawing_settings
from tools_video import getNumberOfFramesFromVideo, extractFrameFromVideo
from tools_annotations import loadAnnotations
from tools_homography import loadHomography, getFrameHomography, getTransformedRegionOfInterest
from tools_homography import transformPointFrom_CARTESIAN_2_PIX, transformAnnotations_CARTESIAN_2_PIX

# #############################################################################
# CONSTANTS
# #############################################################################
RELEVANT_VIDEO = "DJI_0933.MOV"
RELEVANT_FRAME = 3000

video_file_path = video_path+RELEVANT_VIDEO

detection_model_1 = "s2anet_r50_fpn_fp16_1x_dota_le135"
detection_model_2 = "oriented_rcnn_r50_fpn_fp16_1x_dota_le90"
detection_model_3 = "yolo_m_obb"

drawing_settings = default_drawing_settings.copy()
drawing_settings["vehicle_annotations"]["line_width"] = 2
drawing_settings["vehicle_annotations"]["draw_alpha"] = 1.0
#drawing_settings["labelled_vehicle_annnotations"]["draw_alpha"] = 1.0
drawing_settings["labelled_vehicle_annnotations"]["font_size"] = 30

# #############################################################################
# LOADING - FILES
# #############################################################################
# VIDEO
num_frames =  getNumberOfFramesFromVideo(video_file_path)
# HOMOGRAPHY
df_homography = loadHomography(homography_path+RELEVANT_VIDEO+"_circle.txt")
# REGION OF INTEREST
region_of_interest = REGION_OF_INTEREST[RELEVANT_VIDEO]

# ANNOTATIONS
inference_file = "../data_benchmark/1_annotations/Inference_"+detection_model_1+".zip"
annotations_1 = loadAnnotations(inference_file)

inference_file = "../data_benchmark/1_annotations/Inference_"+detection_model_2+".zip"
annotations_2 = loadAnnotations(inference_file)

inference_file = "../data_benchmark/1_annotations/Inference_"+detection_model_3+".zip"
annotations_3 = loadAnnotations(inference_file)

# LABELLED ANNOTATIONS
labelled_file = "../data_benchmark/3_C_vehiclized/Inference_"+detection_model_1+".txt"
annotations_labelled_1 = loadAnnotations(labelled_file)

labelled_file = "../data_benchmark/3_C_vehiclized/Inference_"+detection_model_2+".txt"
annotations_labelled_2 = loadAnnotations(labelled_file)

labelled_file = "../data_benchmark/3_C_vehiclized/Inference_"+detection_model_3+".txt"
annotations_labelled_3 = loadAnnotations(labelled_file)


# #############################################################################
# LOADING - FRAME
# #############################################################################
# VIDEO
success, frame = extractFrameFromVideo(video_file_path, RELEVANT_FRAME)
# HOMOGRAPHY
frame_homography = getFrameHomography(df_homography, RELEVANT_FRAME)
# REGION OF INTEREST
frame_region_of_interest = getTransformedRegionOfInterest(region_of_interest, df_homography, RELEVANT_FRAME)

# ANNOTATIONS
frame_annotations_1 = annotations_1[RELEVANT_FRAME]
frame_labelled_annotations_1 = annotations_labelled_1[RELEVANT_FRAME]
frame_labelled_annotations_pix_1 = transformAnnotations_CARTESIAN_2_PIX(frame_labelled_annotations_1, frame_homography)

frame_annotations_2 = annotations_2[RELEVANT_FRAME]
frame_labelled_annotations_2 = annotations_labelled_2[RELEVANT_FRAME]
frame_labelled_annotations_pix_2 = transformAnnotations_CARTESIAN_2_PIX(frame_labelled_annotations_2, frame_homography)

frame_annotations_3 = annotations_3[RELEVANT_FRAME]
frame_labelled_annotations_3 = annotations_labelled_3[RELEVANT_FRAME]
frame_labelled_annotations_pix_3 = transformAnnotations_CARTESIAN_2_PIX(frame_labelled_annotations_3, frame_homography)

# #############################################################################
# RENDER & DISPLAY IMAGE
# #############################################################################
from tools_video import renderAnnotatedFrame

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

elements = {
    "homography": frame_homography,
    "region_of_interest": frame_region_of_interest,
    "vehicle_annotations": frame_annotations_1,
}
paint_frame = renderAnnotatedFrame(frame, elements, drawing_settings)
axs[0, 0].imshow(paint_frame)
axs[0, 0].set_title(detection_model_1, fontsize=12)
axs[0, 0].set_ylabel("Annotations", fontsize=12)

elements = {
    "homography": frame_homography,
    "region_of_interest": frame_region_of_interest,
    "vehicle_annotations": frame_annotations_2,
}
paint_frame = renderAnnotatedFrame(frame, elements, drawing_settings)
axs[0, 1].imshow(paint_frame)
axs[0, 1].set_title(detection_model_2, fontsize=12)

elements = {
    "homography": frame_homography,
    "region_of_interest": frame_region_of_interest,
    "vehicle_annotations": frame_annotations_3,
}
paint_frame = renderAnnotatedFrame(frame, elements, drawing_settings)
axs[0, 2].imshow(paint_frame)
axs[0, 2].set_title(detection_model_3, fontsize=12)

elements = {
    "homography": frame_homography,
    "region_of_interest": frame_region_of_interest,
    "labelled_vehicle_annnotations": frame_labelled_annotations_pix_1
}
paint_frame = renderAnnotatedFrame(frame, elements, drawing_settings)
axs[1, 0].imshow(paint_frame)
axs[1, 0].set_ylabel("Labels", fontsize=12)

elements = {
    "homography": frame_homography,
    "region_of_interest": frame_region_of_interest,
    "labelled_vehicle_annnotations": frame_labelled_annotations_pix_2
}
paint_frame = renderAnnotatedFrame(frame, elements, drawing_settings)
axs[1, 1].imshow(paint_frame)

elements = {
    "homography": frame_homography,
    "region_of_interest": frame_region_of_interest,
    "labelled_vehicle_annnotations": frame_labelled_annotations_pix_3
}
paint_frame = renderAnnotatedFrame(frame, elements, drawing_settings)
axs[1, 2].imshow(paint_frame)

for ax in axs.flatten():
    ax.set_xlim([1050, 3250])
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()
