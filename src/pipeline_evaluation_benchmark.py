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
from tools_trajectory_evaluation import calculateInternalConsistency, calculatePlatoonConsistency_Headway
import pandas as pd




# #############################################################################
# CONSTANTS
# #############################################################################





# #############################################################################
# LOADING - FILES
# #############################################################################

# FRAME RATE
VIDEO_FRAME_RATE = 25




# #############################################################################
# INTERNAL CONSISTENCY CHECK - PROCESSED
# #############################################################################
vehicle_trajectory_final_path = "../data_benchmark/6_final_trajectories/"

def evaluateModel_internalConsistency(RELEVANT_MODEL, simple_name, dual, vehicle_trajectory_final_path):
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
internal_consistency.append(evaluateModel_internalConsistency("Inference_cfa_r50_fpn_40e_dota_oc",                 "CFA", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90", "RCNN", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_redet_re50_refpn_1x_dota_ms_rr_le90",     "REDET", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_roi_trans_r50_fpn_1x_dota_ms_le90",       "ROI_TRANS", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90", "RetinaNet", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_s2anet_r50_fpn_fp16_1x_dota_le135", "S2A", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_n_obb",                 "Yolo N", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_s_obb",                 "Yolo S", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_m_obb",                 "Yolo M", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_l_obb",                 "Yolo L", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_x_obb",                 "Yolo X", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_faster-rcnn_r50_fpn_hbb_DOTA", "RCNN", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_retinanet_r50_fpn_hbb_DOTA",   "RetinaNet", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_n_hbb",                 "Yolo N", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_s_hbb",                 "Yolo S", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_m_hbb",                 "Yolo M", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_l_hbb",                 "Yolo L", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_x_hbb",                 "Yolo X", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistencyDF = pd.DataFrame(internal_consistency, columns=["model", "name", "dual", "max_err_h", "min_err_h", "avg_err_h", "std_err_h", "max_err_o", "min_err_o", "avg_err_o", "std_err_o", "improvement"])




# #############################################################################
# PLATOON CONSISTENCY CHECK - PROCESSED
# #############################################################################
vehicle_trajectory_final_path = "../data_benchmark/6_final_trajectories/"

def evaluateModel_platoonConsistency(RELEVANT_MODEL, simple_name, dual, vehicle_trajectory_final_path):
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
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_cfa_r50_fpn_40e_dota_oc",                 "CFA", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90", "RCNN", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_redet_re50_refpn_1x_dota_ms_rr_le90",     "REDET", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_roi_trans_r50_fpn_1x_dota_ms_le90",       "ROI_TRANS", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90", "RetinaNet", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_s2anet_r50_fpn_fp16_1x_dota_le135", "S2A", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_n_obb",                 "Yolo N", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_s_obb",                 "Yolo S", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_m_obb",                 "Yolo M", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_l_obb",                 "Yolo L", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_x_obb",                 "Yolo X", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_faster-rcnn_r50_fpn_hbb_DOTA", "RCNN", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_retinanet_r50_fpn_hbb_DOTA",   "RetinaNet", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_n_hbb",                 "Yolo N", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_s_hbb",                 "Yolo S", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_m_hbb",                 "Yolo M", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_l_hbb",                 "Yolo L", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_x_hbb",                 "Yolo X", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistencyDF = pd.DataFrame(platoon_consistency, columns=["model", "name", "dual", "max_err_h", "min_err_h", "avg_err_h", "std_err_h", "max_err_o", "min_err_o", "avg_err_o", "std_err_o", "improvement"])




# #############################################################################
# INTERNAL CONSISTENCY CHECK - RECONSTRUCTED
# #############################################################################
vehicle_trajectory_final_path = "../data_benchmark/7_final_trajectories_reconstructed/"

def load_relax(file):
    f = open(file, "r")
    content = f.read()
    f.close()
    content = content.split("\n")
    rel_1 = float(content[0])
    rel_2 = float(content[1])
    return rel_1, rel_2

def evaluateModel_internalConsistency(RELEVANT_MODEL, simple_name, dual, vehicle_trajectory_final_path):
    result = [RELEVANT_MODEL, simple_name, dual, ]
    if dual==-1:
        result.append(-1)
        result.append(-1)
        result.append(-1)
        result.append(-1)
        result.append(-1)
        result.append(-1)
        final_trajectory_df_obb = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"obb"+".txt", sep=",")
        max_error_o, min_error_o, avg_error_o, std_error_o, distance_travelled_o = calculateInternalConsistency(final_trajectory_df_obb, VIDEO_FRAME_RATE)
        o_rel_1, o_rel_2 = load_relax(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"obb"+"_wc.txt")
        result.append(max_error_o)
        result.append(min_error_o)
        result.append(avg_error_o)
        result.append(std_error_o)
        result.append(-1)
        result.append(o_rel_1)
        result.append(o_rel_2)
    else:
        final_trajectory_df_hbb = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"hbb"+".txt", sep=",")
        max_error_h, min_error_h, avg_error_h, std_error_h, distance_travelled_h = calculateInternalConsistency(final_trajectory_df_hbb, VIDEO_FRAME_RATE)
        h_rel_1, h_rel_2 = load_relax(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"hbb"+"_wc.txt")
        result.append(max_error_h)
        result.append(min_error_h)
        result.append(avg_error_h)
        result.append(std_error_h)
        result.append(h_rel_1)
        result.append(h_rel_2)
        if dual:
            final_trajectory_df_obb = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"obb"+".txt", sep=",")
            max_error_o, min_error_o, avg_error_o, std_error_o, distance_travelled_o = calculateInternalConsistency(final_trajectory_df_obb, VIDEO_FRAME_RATE)
            o_rel_1, o_rel_2 = load_relax(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"obb"+"_wc.txt")
            result.append(max_error_o)
            result.append(min_error_o)
            result.append(avg_error_o)
            result.append(std_error_o)
            improvement = avg_error_h - avg_error_o
            result.append(improvement)
            result.append(o_rel_1)
            result.append(o_rel_2)
        else:
            result.append(-1)
            result.append(-1)
            result.append(-1)
            result.append(-1)
            result.append(-1)
            result.append(-1)
            result.append(-1)
    print(RELEVANT_MODEL, simple_name, dual)
    return result
     
internal_consistency = []
internal_consistency.append(evaluateModel_internalConsistency("Inference_cfa_r50_fpn_40e_dota_oc",                 "CFA", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90", "RCNN", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_redet_re50_refpn_1x_dota_ms_rr_le90",     "REDET", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_roi_trans_r50_fpn_1x_dota_ms_le90",       "ROI_TRANS", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90", "RetinaNet", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_s2anet_r50_fpn_fp16_1x_dota_le135", "S2A", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_n_obb",                 "Yolo N", dual=-1, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_s_obb",                 "Yolo S", dual=-1, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_m_obb",                 "Yolo M", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_l_obb",                 "Yolo L", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_x_obb",                 "Yolo X", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_faster-rcnn_r50_fpn_hbb_DOTA", "RCNN", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistency.append(evaluateModel_internalConsistency("Inference_retinanet_r50_fpn_hbb_DOTA",   "RetinaNet", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_n_hbb",                 "Yolo N", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_s_hbb",                 "Yolo S", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_m_hbb",                 "Yolo M", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_l_hbb",                 "Yolo L", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# internal_consistency.append(evaluateModel_internalConsistency("Inference_yolo_x_hbb",                 "Yolo X", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
internal_consistencyDF_R = pd.DataFrame(internal_consistency, columns=["model", "name", "dual", "max_err_h", "min_err_h", "avg_err_h", "std_err_h", "h_relax_1", "h_relax_2", "max_err_o", "min_err_o", "avg_err_o", "std_err_o", "improvement", "o_relax_1", "o_relax_2"])




# #############################################################################
# PLATOON CONSISTENCY CHECK - RECONSTRUCTED
# #############################################################################
vehicle_trajectory_final_path = "../data_benchmark/7_final_trajectories_reconstructed/"

def evaluateModel_platoonConsistency(RELEVANT_MODEL, simple_name, dual, vehicle_trajectory_final_path):
    result = [RELEVANT_MODEL, simple_name, dual, ]
    if dual==-1:
        result.append(-1)
        result.append(-1)
        result.append(-1)
        result.append(-1)
        final_trajectory_df_obb = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"OBB"+".txt", sep=",")
        max_error_o, min_error_o, avg_error_o, std_error_o = calculatePlatoonConsistency_Headway(final_trajectory_df_obb, VIDEO_FRAME_RATE)
        result.append(max_error_o)
        result.append(min_error_o)
        result.append(avg_error_o)
        result.append(std_error_o)
        result.append(-1)
    else:
        final_trajectory_df_hbb = pd.read_csv(vehicle_trajectory_final_path+RELEVANT_MODEL+"_"+"HBB"+".txt", sep=",")
        max_error_h, min_error_h, avg_error_h, std_error_h = calculatePlatoonConsistency_Headway(final_trajectory_df_hbb, VIDEO_FRAME_RATE)
        result.append(max_error_h)
        result.append(min_error_h)
        result.append(avg_error_h)
        result.append(std_error_h)
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
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_cfa_r50_fpn_40e_dota_oc",                 "CFA", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90", "RCNN", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_redet_re50_refpn_1x_dota_ms_rr_le90",     "REDET", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_roi_trans_r50_fpn_1x_dota_ms_le90",       "ROI_TRANS", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90", "RetinaNet", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_s2anet_r50_fpn_fp16_1x_dota_le135", "S2A", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_n_obb",                 "Yolo N", dual=-1, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_s_obb",                 "Yolo S", dual=-1, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_m_obb",                 "Yolo M", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_l_obb",                 "Yolo L", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_x_obb",                 "Yolo X", dual=True, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_faster-rcnn_r50_fpn_hbb_DOTA", "RCNN", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistency.append(evaluateModel_platoonConsistency("Inference_retinanet_r50_fpn_hbb_DOTA",   "RetinaNet", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_n_hbb",                 "Yolo N", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_s_hbb",                 "Yolo S", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_m_hbb",                 "Yolo M", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_l_hbb",                 "Yolo L", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
# platoon_consistency.append(evaluateModel_platoonConsistency("Inference_yolo_x_hbb",                 "Yolo X", dual=False, vehicle_trajectory_final_path=vehicle_trajectory_final_path))
platoon_consistencyDF_R = pd.DataFrame(platoon_consistency, columns=["model", "name", "dual", "max_err_h", "min_err_h", "avg_err_h", "std_err_h", "max_err_o", "min_err_o", "avg_err_o", "std_err_o", "improvement"])
