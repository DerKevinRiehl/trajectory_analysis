# #############################################################################
# ########## IMPORTS
# #############################################################################
from shapely.geometry.polygon import Polygon
import pandas as pd





        
    


# #############################################################################
# ########## PARAMETERS
# #############################################################################

# Model Files
model_file = "Inference_cfa_r50_fpn_40e_dota_oc.txt"
model_file = "Inference_faster-rcnn_r50_fpn_hbb_DOTA.txt"
model_file = "Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90.txt"
model_file = "Inference_redet_re50_refpn_1x_dota_ms_rr_le90.txt"
model_file = "Inference_retinanet_r50_fpn_hbb_DOTA.txt"
model_file = "Inference_roi_trans_r50_fpn_1x_dota_ms_le90.txt"
model_file = "Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.txt"
model_file = "Inference_s2anet_r50_fpn_fp16_1x_dota_le135.txt"
model_file = "Inference_yolo_l_hbb.txt"
model_file = "Inference_yolo_l_obb.txt"
model_file = "Inference_yolo_m_hbb.txt"
model_file = "Inference_yolo_m_obb.txt"
model_file = "Inference_yolo_n_hbb.txt"
model_file = "Inference_yolo_n_obb.txt"
model_file = "Inference_yolo_s_hbb.txt"
model_file = "Inference_yolo_s_obb.txt"
model_file = "Inference_yolo_x_hbb.txt"
model_file = "Inference_yolo_x_obb.txt"

    # Video & Frame
video_file = "../../data/movs/DJI_0933.MOV"
    # Annotaton File
annotation_file = "../../3_A_trajectorized_unlabelled/"+model_file
    # Trajectory Labels File
trajectory_labels = "../../3_B_trajectorized_mapping/"+model_file
    # Targetoutput File
target_output_file = "../"+model_file
    # CIRCLE File
circle_file = "../../1_homography/DJI_0933.MOV_circle.txt"
    # Output
    # HOMOGRAPHY
CIRCLE_DIAMETER = 65.17 # [m]
    # Filtering Polygon DJI_0933.MOV
ROI_INCL_SHAPE = Polygon([
    (330,  1500),    
    (550,  2050),
    (2700, 2050),
    (3370, 1060),
    (2850, 260),
    (2400, 40),
    (1860, 100),
])
ROI_EXCL_SHAPE = Polygon([
    (520,  1550), 
    (620,  1750),
    (1510, 1805),
    (1130, 990),
    (560, 1550),
])




# #############################################################################
# ########## MAIN - PROCESSING PER FRAME
# #############################################################################

# Load Raw Annotation Data
df_unlabelled_annotations = pd.read_csv(annotation_file, sep="\t", index_col=False, names=["frame_nr", "ann_class", "x", "y", "w", "h", "angle", "conf", "trajectory"])

# Determine Unique Trajectories
unique_trajectories = list(set(df_unlabelled_annotations["trajectory"].tolist()))
unique_trajectories.sort()

import json
trajectory_labels = json.load(open(trajectory_labels, "r"))
df_labelled_annotations = df_unlabelled_annotations.copy() 
for key in trajectory_labels:
    df_labelled_annotations.loc[df_labelled_annotations["trajectory"]==key, "vehicle"] = trajectory_labels[key] 

df_labelled_annotations = df_labelled_annotations.dropna()
df_labelled_annotations.to_csv(target_output_file, sep="\t", index=False, header=False)