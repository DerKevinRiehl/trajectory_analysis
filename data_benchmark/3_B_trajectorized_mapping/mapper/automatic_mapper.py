# #############################################################################
# ########## IMPORTS
# #############################################################################
from shapely.geometry.polygon import Polygon
import pandas as pd
import numpy as np

    


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
annotation_file = "../../3_A_trajectorized_unlabelled/"+model_file
    # target output file
target_output_file = "../../3_B_trajectorized_mapping/"+model_file
    # Ground Truth
ground_truth = "Inference_groundtruth.txt"
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
    # DISTANCE MATCHING
MAX_MATCHING_DISTANCE = 0.5
MAX_FRAME_GAP = 5


# #############################################################################
# ########## MAIN - PROCESSING PER FRAME
# #############################################################################

# Load Annotation Data
df_labelled_annotations = pd.read_csv(ground_truth, sep="\t", index_col=False, names=["frame_nr", "ann_class", "x", "y", "w", "h", "angle", "conf", "trajectory", "vehicle"])
df_unlabelled_annotations = pd.read_csv(annotation_file, sep="\t", index_col=False, names=["frame_nr", "ann_class", "x", "y", "w", "h", "angle", "conf", "trajectory"])

# Determine Unique Trajectories & Vehicles
unique_vehicles = list(set(df_labelled_annotations["vehicle"].tolist()))
# unique_vehicles.sort()
unique_trajectories = list(set(df_unlabelled_annotations["trajectory"].tolist()))
unique_trajectories.sort()

# Automatic Matching
ctr = 0
match_dict = {}
for trajectory in unique_trajectories:
    matches = []
    for vehicle in unique_vehicles:
        df_traj = df_unlabelled_annotations[df_unlabelled_annotations["trajectory"]==trajectory]
        df_veh = df_labelled_annotations[df_labelled_annotations["vehicle"]==vehicle]
        df_traj = df_traj.merge(df_veh, on="frame_nr", how="left")
        df_traj["delta_x"] = abs(df_traj["x_x"] - df_traj["x_y"])
        df_traj["delta_y"] = abs(df_traj["y_x"] - df_traj["y_y"])
        df_traj["dist"] = np.sqrt(df_traj["delta_x"]*df_traj["delta_x"] + df_traj["delta_y"]*df_traj["delta_y"])
        dist_mean = np.mean(df_traj["dist"])
        matches.append([vehicle, dist_mean])
    matches = pd.DataFrame(matches, columns=["vehicle", "dist"])
    matches = matches.sort_values("dist")
    matched_vehicle = matches["vehicle"].iloc[0]
    matched_dist = matches["dist"].iloc[0]
    ctr +=1
    if matched_dist<MAX_MATCHING_DISTANCE:
        df_unlabelled_annotations.loc[df_unlabelled_annotations["trajectory"]==trajectory, "vehicle"] = matched_vehicle
        print(ctr,"/",len(unique_trajectories),"\tMatched", trajectory, "to", matched_vehicle, matched_dist)
        match_dict[trajectory] = matched_vehicle
    else:
        print(trajectory,"not mathced", matched_dist)
# Remove Duplicates
df_unlabelled_annotations = df_unlabelled_annotations.dropna()

# Store to file
import json
json_obj = json.dumps(match_dict, indent=4)
f = open(target_output_file, "w+")
f.write(json_obj)
f.close()

# # Plot Result
# import matplotlib.pyplot as plt
# plt.figure(figsize=(15,10))
# for vehicle in unique_vehicles:
#     df_sel = df_labelled_annotations[df_labelled_annotations["vehicle"]==vehicle]
#     plt.plot(df_sel["frame_nr"], df_sel["x"], label=vehicle, color="gray")
    
#     df_sel2 = df_unlabelled_annotations[df_unlabelled_annotations["vehicle"]==vehicle]
#     plt.plot(df_sel2["frame_nr"], df_sel2["x"], "--",  label=vehicle, color="cyan")

