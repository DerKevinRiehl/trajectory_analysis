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
from tools_trajectory_evaluation import calculateInternalConsistency, calculatePlatoonConsistency_Headway
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





# #############################################################################
# LOADING - FILES
# #############################################################################

# FRAME RATE
VIDEO_FRAME_RATE = 25




# #############################################################################
# INTERNAL CONSISTENCY CHECK
# #############################################################################

def evaluateModel_internalConsistency(RELEVANT_MODEL, simple_name, dual):
    vehicle_trajectory_final_path = "../data_benchmark/6_final_trajectories/"
    final_trajectory_df_hbb = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"HBB"+".txt", sep=",")
    max_error_h, min_error_h, avg_error_h, std_error_h, distance_travelled_h = calculateInternalConsistency(final_trajectory_df_hbb, VIDEO_FRAME_RATE)
    result = [RELEVANT_MODEL, simple_name, dual, max_error_h, min_error_h, avg_error_h, std_error_h]
    if dual:
        final_trajectory_df_obb = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"OBB"+".txt", sep=",")
        max_error_o, min_error_o, avg_error_o, std_error_o, distance_travelled_o = calculateInternalConsistency(final_trajectory_df_obb, VIDEO_FRAME_RATE)
        result.append(max_error_o)
        result.append(min_error_o)
        result.append(avg_error_o)
        result.append(std_error_o)
        improvement = avg_error_h - avg_error_o
        result.append(improvement)
    else:
        result.append(-1)
        result.append(-1)
        result.append(-1)
        result.append(-1)
        result.append(-1)
    print(RELEVANT_MODEL, simple_name, dual)
    return result
     
internal_consistency = []
internal_consistency.append(evaluateModel_internalConsistency("Inference_cfa_r50_fpn_40e_dota_oc",                 "CFA", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90", "RCNN", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_redet_re50_refpn_1x_dota_ms_rr_le90",     "REDET", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_roi_trans_r50_fpn_1x_dota_ms_le90",       "ROI_TRANS", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90", "RetinaNet", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_s2anet_r50_fpn_fp16_1x_dota_le135", "S2A", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_n_obb",                 "Yolo N", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_s_obb",                 "Yolo S", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_m_obb",                 "Yolo M", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_l_obb",                 "Yolo L", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_x_obb",                 "Yolo X", dual=True))
internal_consistency.append(evaluateModel_internalConsistency("Inference_faster-rcnn_r50_fpn_hbb_DOTA", "RCNN", dual=False))
internal_consistency.append(evaluateModel_internalConsistency("Inference_retinanet_r50_fpn_hbb_DOTA",   "RetinaNet", dual=False))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_n_hbb",                 "Yolo N", dual=False))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_s_hbb",                 "Yolo S", dual=False))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_m_hbb",                 "Yolo M", dual=False))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_l_hbb",                 "Yolo L", dual=False))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_x_hbb",                 "Yolo X", dual=False))
internal_consistencyDF = pd.DataFrame(internal_consistency, columns=["model", "name", "dual", "max_err_h", "min_err_h", "avg_err_h", "std_err_h", "max_err_o", "min_err_o", "avg_err_o", "std_err_o", "improvement"])




# #############################################################################
# PLATOON CONSISTENCY CHECK
# #############################################################################

def evaluateModel_platoonConsistency(RELEVANT_MODEL, simple_name, dual):
    vehicle_trajectory_final_path = "../data_benchmark/6_final_trajectories/"
    final_trajectory_df_hbb = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"HBB"+".txt", sep=",")
    max_error_h, min_error_h, avg_error_h, std_error_h = calculatePlatoonConsistency_Headway(final_trajectory_df_hbb, VIDEO_FRAME_RATE)
    result = [RELEVANT_MODEL, simple_name, dual, max_error_h, min_error_h, avg_error_h, std_error_h]
    if dual:
        final_trajectory_df_obb = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"OBB"+".txt", sep=",")
        max_error_o, min_error_o, avg_error_o, std_error_o = calculatePlatoonConsistency_Headway(final_trajectory_df_obb, VIDEO_FRAME_RATE)
        result.append(max_error_o)
        result.append(min_error_o)
        result.append(avg_error_o)
        result.append(std_error_o)
        improvement = avg_error_h - avg_error_o
        result.append(improvement)
    else:
        result.append(-1)
        result.append(-1)
        result.append(-1)
        result.append(-1)
        result.append(-1)
    print(RELEVANT_MODEL, simple_name, dual)
    return result

platoon_consistency = []
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_cfa_r50_fpn_40e_dota_oc",                 "CFA", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90", "RCNN", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_redet_re50_refpn_1x_dota_ms_rr_le90",     "REDET", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_roi_trans_r50_fpn_1x_dota_ms_le90",       "ROI_TRANS", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90", "RetinaNet", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_s2anet_r50_fpn_fp16_1x_dota_le135", "S2A", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_n_obb",                 "Yolo N", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_s_obb",                 "Yolo S", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_m_obb",                 "Yolo M", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_l_obb",                 "Yolo L", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_x_obb",                 "Yolo X", dual=True))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_faster-rcnn_r50_fpn_hbb_DOTA", "RCNN", dual=False))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_retinanet_r50_fpn_hbb_DOTA",   "RetinaNet", dual=False))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_n_hbb",                 "Yolo N", dual=False))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_s_hbb",                 "Yolo S", dual=False))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_m_hbb",                 "Yolo M", dual=False))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_l_hbb",                 "Yolo L", dual=False))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_x_hbb",                 "Yolo X", dual=False))
platoon_consistencyDF = pd.DataFrame(platoon_consistency, columns=["model", "name", "dual", "max_err_h", "min_err_h", "avg_err_h", "std_err_h", "max_err_o", "min_err_o", "avg_err_o", "std_err_o", "improvement"])


