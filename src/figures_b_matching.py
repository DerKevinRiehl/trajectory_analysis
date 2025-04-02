# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np






# TimeGap Analysis
# Files
path = "../data_benchmark/3_C_vehiclized/"
files = os.listdir(path)
files = [file for file in files if file.endswith(".txt")]

# # Analysis 1: Share of Gaps
# for file in files:   
#     f = open(path+file, "r")
#     content = f.read()
#     f.close()
#     content = content.split("\n")
#     max_lines = 14*7518
#     print(file, len(content)/max_lines)

# import sys
# sys.exit(0)

# Inference_cfa_r50_fpn_40e_dota_oc.py.txt 0.6057937141336982
# Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90.py.txt 0.4771120738798313
# Inference_redet_re50_refpn_1x_dota_ms_rr_le90.py.txt 0.9127902557671114
# Inference_roi_trans_r50_fpn_1x_dota_ms_le90.py.txt 0.8556036939915631
# Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py.txt 0.9250370539277163
# Inference_s2anet_r50_fpn_fp16_1x_dota_le135.py.txt 0.9416258123361076
# Inference_yolo_l_obb.txt 0.5301086915213012
# Inference_yolo_m_obb.txt 0.6546288906624103
# Inference_yolo_n_obb.txt 0.6001311138980732
# Inference_yolo_s_obb.txt 0.5344506517690876
# Inference_yolo_x_obb.txt 0.7689450081708661


data_obb_x = [ "CFA", "REDET", "ROI_TRANS", "S2A", 
                "RetinaNet",  "RCNN", 
                "Yolo N", "Yolo S", "Yolo M", "Yolo L", "Yolo X"]


data_hbb_y_sha_tra = [  0.000000000, 0.000000000, 0.000000000, 0.000000000,
                        0.939573594801049, 0.9619294645232395,
                        0.8124121156842625, 0.6183730475430396, 0.3067590164557443, 0.5112776954357162, 0.6322825219473264]
data_obb_y_sha_tra = [  0.8821020028122981, 0.9920571580587543, 0.9617679474024247, 0.9996864667654771,
                        0.9894633831186106, 0.8314806369475164,
                        0.7924504997529738, 0.8142173070345456, 0.8796602439858625, 0.8393379698248015, 0.9499486945616236]




# # Analysis 2: Length Distribution of Gaps
def loadGaps(files):
    gaps_lst = []
    for file in files:
        if file is not None:
            df = pd.read_csv(path+file, delimiter="\t", names=["frameNr", "type", "x", "y", "w", "h", "angle", "confid", "label", "XX"])
            vehicles = list(set(df["label"].tolist()))
            vehicle = vehicles[0]
            df_sel = df[df["label"]==vehicle]
            gaps = []
            last_framenr = -1
            for idx, row in df_sel.iterrows():
                if row["frameNr"] != last_framenr+1:
                    # print("gap! ", row["frameNr"], row["frameNr"]-last_framenr)
                    gaps.append(row["frameNr"]-last_framenr)
                    last_framenr = row["frameNr"]
                else:
                    last_framenr = row["frameNr"]
            gaps_lst.append(gaps)
        else:
            gaps_lst.append([-1, -1 , -1])
    return gaps_lst

files_obb = [   "Inference_cfa_r50_fpn_40e_dota_oc.txt", "Inference_redet_re50_refpn_1x_dota_ms_rr_le90.txt", "Inference_roi_trans_r50_fpn_1x_dota_ms_le90.txt", "Inference_s2anet_r50_fpn_fp16_1x_dota_le135.txt",
                "Inference_rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.txt", "Inference_oriented_rcnn_r50_fpn_fp16_1x_dota_le90.txt",
                "Inference_yolo_n_obb.txt", "Inference_yolo_s_obb.txt", "Inference_yolo_m_obb.txt", "Inference_yolo_l_obb.txt", "Inference_yolo_x_obb.txt", ]
files_hbb = [   None, None, None, None,
                "Inference_retinanet_r50_fpn_hbb_DOTA.txt", "Inference_faster-rcnn_r50_fpn_hbb_DOTA.txt",
                "Inference_yolo_n_hbb.txt", "Inference_yolo_s_hbb.txt", "Inference_yolo_m_hbb.txt", "Inference_yolo_l_hbb.txt", "Inference_yolo_x_hbb.txt", ]

gaps_obb = loadGaps(files_obb)
gaps_obb_max = [max(gaps) for gaps in gaps_obb ]
gaps_obb_min = [min(gaps) for gaps in gaps_obb ]
gaps_obb_avg = [np.mean(gaps) for gaps in gaps_obb]

gaps_hbb = loadGaps(files_hbb)
gaps_hbb_max = [max(gaps) for gaps in gaps_hbb]
gaps_hbb_min = [min(gaps) for gaps in gaps_hbb]
gaps_hbb_avg = [np.mean(gaps) for gaps in gaps_hbb]






# PLOT
data_obb_x = [ "CFA", "REDET", "ROI_TRANS", "S2A", 
                "RetinaNet",  "RCNN", 
                "Yolo N", "Yolo S", "Yolo M", "Yolo L", "Yolo X"]

plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
fig = plt.figure(figsize=(12, 4), dpi=100)

plt.subplot(2,2,1)
plt.ylabel("Coverage [%]")
df = pd.DataFrame(np.asarray([data_hbb_y_sha_tra, data_obb_y_sha_tra ]).transpose()*100, columns=["HBB Trajectory", "OBB Trajectory", ])
df.plot.bar(ax=plt.gca(), edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
plt.gca().set_xticklabels([])
plt.legend(loc='upper left', ncol=2, fontsize=8)
plt.grid(zorder=0)
plt.ylim(0,120)

plt.subplot(2,2,2)
plt.ylabel("Maximum\nGap Length [Frames]")
df = pd.DataFrame(np.asarray([gaps_hbb_max, gaps_obb_max]).transpose(), columns=["HBB", "OBB"])
df.plot.bar(ax=plt.gca(), edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
plt.gca().set_xticklabels([])
plt.grid(zorder=0)
plt.yscale("log")
plt.gca().set_xticklabels([])
plt.ylim(1.3358746602850649, 9584.731547544832)

plt.subplot(2,2,3)
plt.ylabel("Average\nGap Length [Frames]")
df = pd.DataFrame(np.asarray([gaps_hbb_avg, gaps_obb_avg]).transpose(), columns=["HBB", "OBB"])
df.plot.bar(ax=plt.gca(), edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
plt.gca().set_xticklabels(data_obb_x)
plt.xticks(rotation=45)
plt.grid(zorder=0)
plt.yscale("log")

plt.subplot(2,2,4)
plt.ylabel("Minimum\nGap Length [Frames]")
df = pd.DataFrame(np.asarray([gaps_hbb_min, gaps_obb_min]).transpose(), columns=["HBB", "OBB"])
df.plot.bar(ax=plt.gca(), edgecolor="black", color=["blue","aqua"], width=0.75, legend=False, zorder=3)
plt.gca().set_xticklabels([])
plt.xticks(rotation=45)
plt.gca().set_xticklabels(data_obb_x)
plt.grid(zorder=0)
plt.yscale("log")
plt.ylim(1.3358746602850649, 9584.731547544832)


plt.tight_layout()


# plt.boxplot(gaps)