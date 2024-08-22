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
import sys
import json
import mplcursors
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.Qt import QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QWidget, QApplication

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
from tools_trajectorization import generateTrajectories, determineUniqueTrajectoryLabels
from tools_trajectorization import generateEmptyTrajectoryLabelVehicleMap, loadTrajectoryLabelVehicleMap


# #############################################################################
# CONSTANTS
RELEVANT_VIDEO = "DJI_0934.MOV"


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
# SOME FUNCTIONS
# #############################################################################
def show_annation(sel):
    try:
        row = zoom_df.iloc[sel.index]
        sel.annotation.set_text(f"{row['trajectory']}\n{row['vehicle']}")
    except KeyError:
        pass


def convert_annotations_dataframe_to_dict(df, df_homography):
    df = df.drop(columns=['time', 'legend_label'])
    grouped = df.groupby(by=['frame_num'])
    ann_dict = {}
    ann_pix_dict = {}
    for (frame_num,), group_df in grouped:
        frame_homography = getFrameHomography(df_homography, frame_num)
        ann_dict[frame_num] = group_df.drop(columns=['frame_num']).values.tolist()
        ann_pix_dict[frame_num] = transformAnnotations_CARTESIAN_2_PIX(ann_dict[frame_num], frame_homography)
    return ann_dict, ann_pix_dict


def map_trajectories_to_vehicles(df, trajectory_vehicle_map):
    df["vehicle"] = df["trajectory"]
    df["vehicle"] = df["vehicle"].map(trajectory_vehicle_map)
    df = df[df["vehicle"] != "REMOVE"]
    df["legend_label"] = df["vehicle"]
    df.loc[df["vehicle"] == "UNDEFINED", "legend_label"] = df.loc[df["vehicle"] == "UNDEFINED", "trajectory"]
    return df


def find_and_remove_redundant_trajectories(df, trajectory_vehicle_map, map_file):
    subdf = df[df['vehicle'] != 'UNDEFINED']
    grouped = subdf.groupby(by=['vehicle'])
    max_coherent_time = np.Inf
    vehicle_ending = None
    for (veh,), group_df in grouped:
        if group_df['time'].max() < max_coherent_time:
            vehicle_ending = veh
            max_coherent_time = group_df['time'].max()
    print(vehicle_ending, max_coherent_time)

    remove_trajectories = []
    subdf = df[df['vehicle'] == 'UNDEFINED']
    grouped = subdf.groupby(by=['trajectory'])
    for (traj,), group_df in grouped:
        if group_df['time'].max() < max_coherent_time:
            remove_trajectories.append(traj)
    print(remove_trajectories)

    if len(remove_trajectories) > 0:
        for traj in remove_trajectories:
            if trajectory_vehicle_map[traj] == 'UNDEFINED':
                trajectory_vehicle_map[traj] = 'REMOVE'
        with open(map_file, 'w') as file:
            file.write(json.dumps(trajectory_vehicle_map, indent=4))
        print('Removed redundant trajectories and saved!')
    
    df = map_trajectories_to_vehicles(df, trajectory_vehicle_map)
    return df, trajectory_vehicle_map, vehicle_ending, max_coherent_time


# #############################################################################
# STEP 2: LOADING AND MAPPING TRAJECTORY/VEHICLE ANNOTATIONS
# #############################################################################
trajectorized_annotation_file = "../data/3_A_trajectorized_unlabelled/"+RELEVANT_VIDEO+".txt"
df = pd.read_csv(trajectorized_annotation_file, sep="\t", header=None)
df.columns = ["frame_num", "ann_class", "x", "y", "w", "h", "angle", "conf", "trajectory"]
df["time"] = df["frame_num"] / 25.0

map_file = "../data/3_B_trajectorized_mapping/"+RELEVANT_VIDEO+".txt"
trajectory_vehicle_map = loadTrajectoryLabelVehicleMap(map_file)
df = map_trajectories_to_vehicles(df, trajectory_vehicle_map)


# #############################################################################
# MAPPING TRAJECTORY WIDGET
# #############################################################################
class MapTrajectoryWidget(QWidget):
    def __init__(self, traj_list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.button = QPushButton("Select")
        self.button.clicked.connect(self.on_clicled)

        self.current_text = None
        self.combo = QComboBox(self)
        self.options_list = ["Select the most_plausible trajectory ..."] + traj_list
        self.combo.addItems(self.options_list)
        self.combo.currentTextChanged.connect(self.on_combobox_func)

        v_box = QVBoxLayout()
        v_box.addWidget(self.combo)
        v_box.addStretch()
        v_box.addWidget(self.button)    
        h_box = QHBoxLayout(self)
        h_box.addLayout(v_box)
        self.show()
        plt.show(block=True)

    def on_combobox_func(self, text):
        self.current_text  = text 
        
    def on_clicled(self):
        if self.current_text == self.options_list[0]:
            pass
        else:
            self.close()
            plt.close()


# #############################################################################
# LABELLING
# #############################################################################
max_num_frames, max_time = df['frame_num'].max(), df['time'].max()
while True:
    df, trajectory_vehicle_map, vehicle_ending, max_coherent_time = find_and_remove_redundant_trajectories(df, trajectory_vehicle_map, map_file)
    if max_coherent_time >= max_time:
        print("Finished labelling for the video ", RELEVANT_VIDEO)
        break
    
    start_zoom_time = max(max_coherent_time - 2.5, 0)
    end_zoom_time = min(max_coherent_time + 8.0, max_time)
    sns.color_palette("dark")
    zoom_df = df[(df.time >= start_zoom_time) & (df.time <= end_zoom_time) & (df['vehicle'].isin([vehicle_ending, 'UNDEFINED']))]
    fig, axs = plt.subplots(2, 1)
    dots = sns.scatterplot(data=zoom_df, x="time", y="x", hue="legend_label", style="legend_label", ax=axs[0])
    cursor1 = mplcursors.cursor(dots, hover=True)
    cursor1.connect('add', show_annation)
    sns.lineplot(data=zoom_df, x="time", y="x", hue="legend_label", style="legend_label", ax=axs[0])

    dots = sns.scatterplot(data=zoom_df, x="time", y="y", hue="legend_label", style="legend_label", ax=axs[1])
    cursor2 = mplcursors.cursor(dots, hover=True)
    cursor2.connect('add', show_annation)
    sns.lineplot(data=zoom_df, x="time", y="y", hue="legend_label", style="legend_label", ax=axs[1])

    demo = MapTrajectoryWidget(zoom_df.loc[zoom_df['vehicle'] != vehicle_ending, 'trajectory'].unique().tolist())
    selected_trajectory = demo.current_text
    del demo, fig, axs, dots, cursor1, cursor2, zoom_df

    trajectory_vehicle_map[selected_trajectory] = vehicle_ending
    df = map_trajectories_to_vehicles(df, trajectory_vehicle_map)
    with open(map_file, 'w') as file:
        file.write(json.dumps(trajectory_vehicle_map, indent=4))
    print(f'Mapped {selected_trajectory} to {vehicle_ending} and saved!')


vehiclized_annotations, vehiclized_annotations_pix = convert_annotations_dataframe_to_dict(df, df_homography)
saveAnnotations("../data/3_C_vehiclized/"+RELEVANT_VIDEO+".txt", vehiclized_annotations)
saveAnnotations("../data/3_C_vehiclized/"+RELEVANT_VIDEO+"_PIX.txt", vehiclized_annotations_pix)

elements = {
    "homography": df_homography,
    "region_of_interest": region_of_interest,
    "labelled_vehicle_annnotations": vehiclized_annotations_pix,
}
drawing_settings = default_drawing_settings.copy()
renderAnnotatedVideo(video_file_path_source=video_file_path, 
                        video_file_path_destination="../videos/3_B_trajectorized_mapping/"+RELEVANT_VIDEO, 
                        elements=elements, 
                        design=drawing_settings, 
                        max_num_frames=None, 
                        print_status=True)

# TODO: handle coincident annotations of the same vehicle according to their confidence (weighted average of both OR dropping less confident one)