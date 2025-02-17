# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###############################################################################
###############################################################################
########### ALL EVALUATION DONE ON DJI_0933.MOV ###############################
###############################################################################
###############################################################################


# Models
data_hbb_x = [ "CFA", "REDET", "ROI_TRANS", "S2A", 
                "RetinaNet",  "RCNN", 
                "Yolo N", "Yolo S", "Yolo M", "Yolo L", "Yolo X"]
data_obb_x = [  "CFA", "REDET", "ROI_TRANS", "S2A", 
                "RetinaNet",  "RCNN", 
                "Yolo N", "Yolo S", "Yolo M", "Yolo L", "Yolo X"]

# =========== UNPROCESSED ===========
# %     # number of annotations
data_hbb_y_num_ann = [  0.000000000, 0.000000000, 0.000000000, 0.000000000,
                        98.37935621175845, 62.64977387603086,
                        32.12024475, 20.62742751, 17.12995478, 17.91899441, 21.42790636]
data_obb_y_num_ann = [  108.6235701, 43.78930567, 53.92537909, 121.2748071,
                        162.7801277, 37.44892259,
                        33.71189146, 32.78784251, 33.97180101, 32.18741687, 35.8056664]

# %     # avg confidence of annotation
data_hbb_y_con_ann = [  0.000000000, 0.000000000, 0.000000000, 0.000000000,
                        0.2443964336166935, 0.36212262911568566,
                        0.622100212, 0.628608955, 0.604254695, 0.633631998, 0.706031031]
data_obb_y_con_ann = [  0.113516425, 0.571120568, 0.320256854, 0.125308922,
                        0.117447941, 0.239942899,
                        0.596756791, 0.644846571, 0.641650365, 0.662720141, 0.701951753]


# =========== PROCESSED (SFAF - Single Frame Annotation Filter  ===========

sfaf_obb_y_num_ann = [  15.207368981111998, 13.893987762702846, 13.540569300345837, 15.81098696461825,
                        21.37776004256451, 11.64658153764299,
                        11.127028465017291, 11.410082468741686, 12.32242617717478, 11.752593774940143, 13.306597499334929]
sfaf_obb_y_con_ann = [  0.19273914408136203, 0.8053176797354342, 0.5714697509456961, 0.22457615483844268,
                        0.19593743465097138, 0.34130987192462137,
                        0.515756746536285, 0.6287278206479288, 0.6513589647560447, 0.6308729619493867, 0.6970100870360554]

sfaf_hbb_y_num_ann = [  0.000000000, 0.000000000, 0.000000000, 0.000000000,
                        13.668129821761106, 14.073955839318968,
                        8.861399308326682, 7.162543229582336, 4.318939091030255, 8.66267624368183, 11.381750465549349]
sfaf_hbb_y_con_ann = [  0.000000000, 0.000000000, 0.000000000, 0.000000000,
                        0.3877167017260684, 0.5982890511808995,
                        0.5854212955310403, 0.5790241941313766, 0.6146112033636795, 0.6333482339362664, 0.7427137269738825]





# PLOT
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
fig = plt.figure(figsize=(12, 3), dpi=100)

############### Number of Annotations
plt.subplot(1,2,1)
plt.ylabel("# Annotations per Frame")
df = pd.DataFrame(np.asarray([data_hbb_y_num_ann, sfaf_hbb_y_num_ann, data_obb_y_num_ann, sfaf_obb_y_num_ann,  ]).transpose(), columns=["HBB Raw", "HBB Processed", "OBB Raw", "OBB Processed", ])
bars = df.plot.bar(ax=plt.gca(), edgecolor="black", color=["blue","blue","aqua","aqua"], width=0.75, legend=False, zorder=3)
counter = 1
for bar in bars.patches:
    if counter >11 and counter <= 22:
        bar.set_hatch("//")
    if counter >33 and counter <= 44:
        bar.set_hatch("//")
    if counter <= 22:
        bar.set_edgecolor("white")
    counter += 1
plt.gca().set_xticklabels([])
plt.legend(loc='upper right',  # bbox_to_anchor=(0.5, 1.05),
          ncol=2, fontsize=6)
plt.plot([-1, 11], [14, 14], "--", color="black", alpha=0.5)
plt.xticks(rotation=45)
plt.gca().set_xticklabels(data_obb_x)
plt.grid(zorder=0)
plt.yscale("log")

plt.subplot(1,2,2)
# plt.xlabel("Models")
plt.ylabel("Annotation Confidence [%]")
df = pd.DataFrame(np.asarray([data_hbb_y_con_ann, sfaf_hbb_y_con_ann, data_obb_y_con_ann, sfaf_obb_y_con_ann, ]).transpose()*100, columns=["HBB Raw", "HBB Processed", "OBB Raw", "OBB Processed", ])
bars = df.plot.bar(ax=plt.gca(), edgecolor="black", color=["blue","blue","aqua","aqua"], width=0.75, legend=False, zorder=3)
counter = 1
for bar in bars.patches:
    if counter >11 and counter <= 22:
        bar.set_hatch("//")
    if counter >33 and counter <= 44:
        bar.set_hatch("//")
    if counter <= 22:
        bar.set_edgecolor("white")
    counter += 1
plt.xticks(rotation=45)
plt.gca().set_xticklabels(data_obb_x)
plt.grid(zorder=0)


plt.tight_layout()
