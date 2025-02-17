# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np






# WIDTH & HEIGHT Analysis

folder = "../full_pipeline/processed/2_trajectorized_labelled/"
folder = "../data_benchmark/3_C_vehiclized/"
files_hbb_x = [ None, None, None, None,
                "Inference_retinanet_r50_fpn_hbb_DOTA.txt", "Inference_faster-rcnn_r50_fpn_hbb_DOTA.txt",
                "Inference_yolo_n_hbb.txt", "Inference_yolo_s_hbb.txt", "Inference_yolo_m_hbb.txt", "Inference_yolo_l_hbb.txt", "Inference_yolo_x_hbb.txt", ]
files_obb_x = [ "Inference_cfa_r50_fpn_40e_dota_oc.txt", "Inference_redet_re50_refpn_1x_dota_ms_rr_le90.txt", "Inference_roi_trans_r50_fpn_1x_dota_ms_le90.txt", 
               "Inference_s2anet_r50_fpn_fp16_1x_dota_le135.txt", "Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.txt", "Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90.txt",
                "Inference_yolo_n_obb.txt", "Inference_yolo_s_obb.txt", "Inference_yolo_m_obb.txt", "Inference_yolo_l_obb.txt", "Inference_yolo_x_obb.txt", ]

# import warnings
# warnings.filterwarnings('ignore')
# for file in files_obb_x:
#     if file is None:
#         print(file, "X", "X")
#         continue
#     df = pd.read_csv(folder+file, sep="\t", names=["frame_nr", "class", "x", "y", "w", "h", "angle", "conf", "traj", "veh"])
#     stds_pos = []
#     stds_ang = []
#     for sel_vehicle in ["Vehicle_1","Vehicle_2","Vehicle_3","Vehicle_4","Vehicle_5","Vehicle_6","Vehicle_7","Vehicle_8","Vehicle_9","Vehicle_10","Vehicle_11","Vehicle_12","Vehicle_13","Vehicle_14"]:
#         dfc = df[df["veh"]==sel_vehicle.upper()]
#         dfc["delta_x"] = dfc["x"].diff(1)
#         dfc["delta_y"] = dfc["y"].diff(1)
#         dfc["delta_z"] = np.sqrt(dfc["delta_x"]*dfc["delta_x"] + dfc["delta_y"]*dfc["delta_y"])
#         dfc["delta_angle"] = dfc["angle"].diff(1)
#         dfc["delta_frame"] = dfc["frame_nr"].diff(1)
#         dfc["filter"] = dfc["delta_frame"]==1
#         stds_pos.append(np.std(dfc[dfc["filter"]]["delta_z"]))
#         stds_ang.append(np.std(dfc[dfc["filter"]]["delta_angle"]))
#     print(file, np.mean(stds_pos), np.mean(stds_ang)/(2*3.14159)*360)
# import sys
# sys.exit(0)

# import warnings
# warnings.filterwarnings('ignore')
# for file in files_hbb_x:
#     if file is None:
#         print(file, "X", "X")
#         continue
#     df = pd.read_csv(folder+file, sep="\t", names=["frame_nr", "class", "x", "y", "w", "h", "angle", "conf", "traj", "veh"])
#     stds_w = []
#     stds_h = []
#     for sel_vehicle in ["Vehicle_1","Vehicle_2","Vehicle_3","Vehicle_4","Vehicle_5","Vehicle_6","Vehicle_7","Vehicle_8","Vehicle_9","Vehicle_10","Vehicle_11","Vehicle_12","Vehicle_13","Vehicle_14"]:
#         dfc = df[df["veh"]==sel_vehicle.upper()]
#         dfc["width"] = dfc[["w", "h"]].min(axis=1)
#         dfc["height"] =  dfc[["w", "h"]].max(axis=1)
#         stds_w.append(np.std(dfc["width"]))
#         stds_h.append(np.std(dfc["height"]))
#     print(file, np.mean(stds_w), np.mean(stds_h))
# import sys
# sys.exit(0)

widths_std_hbb = [0.000000000, 0.000000000, 0.000000000, 0.000000000,
                    0.7905818043954771, 0.7674172552552638,
                    0.7401845574418856, 0.6659412596961077, 0.670669519126078, 0.6630572779806689, 0.7442974291021572]
heights_std_hbb = [0.000000000, 0.000000000, 0.000000000, 0.000000000,
                     0.1987527698895726, 0.23051094121261048,
                     0.1972541812333737, 0.21966439870572857, 0.17434412704796612, 0.2221717919667086, 0.1828723249165304]

pos_std_hbb = [0.000000000, 0.000000000, 0.000000000, 0.000000000,
               0.06086757662574714, 0.0802137351779828,
               0.06412934090815066, 0.05857270965655998, 0.0580602381724197, 0.05343365712732079, 0.05488317193422854]





widths_std_obb = [0.1938340182193131, 0.1084013664863446, 0.13042308072444422, 0.1646702082145943,
                  0.14331848971457253, 0.16592409371114353,
                  0.08031609766708434, 0.1072291824125342, 0.07522767049535532, 0.08527299855543306, 0.08113934382038777]
heights_std_obb = [0.25093530182702606, 0.11478409979135629, 0.1597685628739031, 0.2253521877908387, 
                   0.24621222249446093, 0.19332232146414574,
                   0.12033224242493866, 0.1297041551258273, 0.09341428857250207, 0.1031993578263399, 0.09230776815242617]

pos_std_obb = [0.09327364423968625, 0.05829270782561979, 0.06421315878197094, 0.10166220685302005,
               0.11167077844225268, 0.07569109148303944,
               0.06364828510033368, 0.0599293488721109, 0.056171966013379926, 0.05404074148246096, 0.055603930779439384]

angles_std_obb = [ 9.230663735132529, 13.81071468926735, 14.388513140386406, 20.46476012211457,
                   11.18572028785761, 15.216895794377676,
                   25.308379651538893, 21.528749817058674, 21.580520003847788, 20.994709621895787, 19.995226650130256]
angles_std_obb = [angle/5 for angle in angles_std_obb]



# PLOT
data_obb_x = [ "CFA", "REDET", "ROI_TRANS", "S2A", 
                "RetinaNet",  "RCNN", 
                "Yolo N", "Yolo S", "Yolo M", "Yolo L", "Yolo X"]

df_hbb = pd.read_csv(folder+"Inference_faster-rcnn_r50_fpn_hbb_DOTA.txt",            sep="\t", names=["frame_nr", "class", "x", "y", "w", "h", "angle", "conf", "traj", "veh"])
df_hbb = df_hbb[df_hbb["veh"]=="VEHICLE_3"]
df_obb = pd.read_csv(folder+"Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90.txt", sep="\t", names=["frame_nr", "class", "x", "y", "w", "h", "angle", "conf", "traj", "veh"])
df_obb = df_obb[df_obb["veh"]=="VEHICLE_3"]


plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
fig = plt.figure(figsize=(12, 5), dpi=100)


plt.subplot(3,2,1)
plt.title("Exemplary Trajectory (RCNN)")
plt.xlim(0,4000)
plt.grid(zorder=0)
plt.plot(df_obb["frame_nr"], df_obb["x"], color="gray")
plt.plot(df_hbb["frame_nr"]+500, df_hbb["x"], color="gray")
mask = np.diff(df_obb["frame_nr"]) == 1
mask = np.insert(mask, 0, False) 
segments = np.split(df_obb.index, np.where(~mask)[0])
for segment in segments:
    if len(segment) > 1:
        plt.plot(df_obb.loc[segment, "frame_nr"], df_obb.loc[segment, "x"], color="aqua")
plt.gca().set_xticklabels([])
mask = np.diff(df_hbb["frame_nr"]) == 1
mask = np.insert(mask, 0, False) 
segments = np.split(df_hbb.index, np.where(~mask)[0])
for segment in segments:
    if len(segment) > 1:
        plt.plot(df_hbb.loc[segment, "frame_nr"]+500, df_hbb.loc[segment, "x"], color="blue")        

plt.subplot(3,2,3)
plt.plot(df_hbb["frame_nr"], df_hbb["x"], color="blue", label="HBB")
plt.plot(df_obb["frame_nr"], df_obb["x"], color="aqua", label="OBB")
plt.gca().set_xticklabels([])
plt.legend()
plt.grid(zorder=0)
plt.xlim(0,100)
plt.ylim(-31.75,-30)

plt.subplot(3,2,5)
plt.ylabel("Position [m]")
df = pd.DataFrame(np.asarray([pos_std_hbb, pos_std_obb]).transpose(), columns=["HBB", "OBB"])
df.plot.bar(ax=plt.gca(), edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
plt.xticks(rotation=45)
plt.gca().set_xticklabels(data_obb_x)
plt.grid(zorder=0)

plt.subplot(3,2,2)
plt.title("Trajectory Noise Analysis OBB (Variance)")
plt.ylabel("Angle [°]")
df = pd.DataFrame(np.asarray([angles_std_obb]).transpose(), columns=["OBB"])
df.plot.bar(ax=plt.gca(), edgecolor="black", color=["aqua"], width=0.75/2, legend=False, zorder=3)
plt.gca().set_xticklabels([])
plt.grid(zorder=0)

plt.subplot(3,2,2+2)
plt.ylabel("Width [m]")
df = pd.DataFrame(np.asarray([widths_std_obb ]).transpose(), columns=["OBB Trajectory", ])
df.plot.bar(ax=plt.gca(), edgecolor="black", color=["aqua"], width=0.75/2, legend=False, zorder=3)
plt.gca().set_xticklabels([])
plt.grid(zorder=0)

plt.subplot(3,2,4+2)
plt.ylabel("Height [m]")
df = pd.DataFrame(np.asarray([heights_std_obb]).transpose(), columns=["OBB"])
df.plot.bar(ax=plt.gca(), edgecolor="black", color=["aqua"], width=0.75/2, legend=False, zorder=3)
plt.xticks(rotation=45)
plt.gca().set_xticklabels(data_obb_x)
plt.grid(zorder=0)

plt.tight_layout()


