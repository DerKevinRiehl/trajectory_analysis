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
from PyQt5.Qt import QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QGridLayout, QWidget, QApplication
from alive_progress import alive_bar

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
from tools_video import extractFrameFromVideo, renderAnnotatedFrame
from tools_trajectorization import generateTrajectories, determineUniqueTrajectoryLabels
from tools_trajectorization import generateEmptyTrajectoryLabelVehicleMap, loadTrajectoryLabelVehicleMap


# #############################################################################
# CONSTANTS
RELEVANT_VIDEO = "DJI_0943.MOV"
INITIAL_MAPPING_DONE = False


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


class InitialMapTrajectoryWidget(QWidget):
    def __init__(self, traj_list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.options_list = [f"VEHICLE_{i+1}" for i in range(15)]
        self.options_list = ["Select Label ..."] + ["REMOVE"] + self.options_list

        num_rows = int(len(traj_list)//2) + int(len(traj_list)%2)
        self.num_trajectories = len(traj_list)
        self.map_dict = {}
        self.traj_list = []
        for i in range(num_rows):
            if 2*i+1 < len(traj_list):
                self.traj_list.append([traj_list[2*i], traj_list[2*i+1]])
            else:
                self.traj_list.append([traj_list[2*i]])

        self.grid_layout = QGridLayout()
        for r in range(num_rows):
            label = QLabel()
            label.setText(f"Select Label for {self.traj_list[r][0]}: ")
            self.grid_layout.addWidget(label, r, 0)
            comboBox = QComboBox()
            comboBox.addItems(self.options_list)
            comboBox.currentTextChanged.connect(lambda text, row=r, col=0: self.get_mapping(row, col, text))
            self.grid_layout.addWidget(comboBox, r, 1)
            if len(self.traj_list[r]) == 2:
                label = QLabel()
                label.setText(f"Select Label for {self.traj_list[r][1]}: ")
                self.grid_layout.addWidget(label, r, 2)
                comboBox = QComboBox()
                comboBox.addItems(self.options_list)
                comboBox.currentTextChanged.connect(lambda text, row=r, col=1: self.get_mapping(row, col, text))
                self.grid_layout.addWidget(comboBox, r, 3)

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_clicled)
        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.clicked.connect(self.visualize_clicked)
        self.reset_visual_button = QPushButton("Reset Visualization")
        self.reset_visual_button.clicked.connect(self.reset_visual_clicked)
        self.vbox = QVBoxLayout(self)
        self.vbox.addLayout(self.grid_layout)
        self.vbox.addStretch()
        self.vbox.addWidget(self.visualize_button)
        self.vbox.addStretch()
        self.vbox.addWidget(self.reset_visual_button)
        self.vbox.addStretch()
        self.vbox.addWidget(self.confirm_button)    
        self.show()
        plt.show(block=True)
    
    def get_mapping(self, row, col, text):
        if text != self.options_list[0]:
            self.map_dict[self.traj_list[row][col]] = text
    
    def confirm_clicled(self):
        if self.num_trajectories == len(self.map_dict):
            self.close()
            plt.close()
    
    def visualize_clicked(self):
        if self.num_trajectories == len(self.map_dict):
            plt.close()
            vehiclized_df_0 = df_0.copy()
            vehiclized_df_0["vehicle"] = vehiclized_df_0["trajectory"]
            vehiclized_df_0["vehicle"] = vehiclized_df_0["vehicle"].map(self.map_dict)
            vehiclized_df_0 = vehiclized_df_0[vehiclized_df_0["vehicle"] != "REMOVE"]
            vehiclized_annotations_0 = vehiclized_df_0.values.tolist()
            elements["labelled_vehicle_annnotations"] = transformAnnotations_CARTESIAN_2_PIX(vehiclized_annotations_0, frame_homography)
            frame_finished = renderAnnotatedFrame(frame, elements, design=drawing_settings)
            plt.figure()
            plt.imshow(frame_finished)
            plt.show(block=True)
    
    def reset_visual_clicked(self):
        plt.close()
        elements["labelled_vehicle_annnotations"] = transformAnnotations_CARTESIAN_2_PIX(trajectorized_annotations_0, frame_homography)
        frame_finished = renderAnnotatedFrame(frame, elements, design=drawing_settings)
        plt.figure()
        plt.imshow(frame_finished)
        plt.show(block=True)



# #############################################################################
# LABELLING
# #############################################################################
if not INITIAL_MAPPING_DONE:
    success, frame = extractFrameFromVideo(video_file_path, 0)
    frame_homography = getFrameHomography(df_homography, 0)
    frame_region_of_interest = getTransformedRegionOfInterest(region_of_interest, df_homography, 0)
    df_0 = df[df["frame_num"] == 0].copy()
    df_0 = df_0.drop(columns=['frame_num', 'time', 'legend_label', 'vehicle'])
    trajectorized_annotations_0 = df_0.values.tolist()

    drawing_settings = default_drawing_settings.copy()
    elements = {
        "homography": frame_homography,
        "region_of_interest": frame_region_of_interest,
        "labelled_vehicle_annnotations": transformAnnotations_CARTESIAN_2_PIX(trajectorized_annotations_0, frame_homography),
    }
    frame_finished = renderAnnotatedFrame(frame, elements, design=drawing_settings)
    plt.figure()
    plt.imshow(frame_finished)
    demo = InitialMapTrajectoryWidget(df_0["trajectory"].unique().tolist())
    for traj in demo.map_dict.keys():
        trajectory_vehicle_map[traj] = demo.map_dict[traj]
    del demo, frame, frame_homography, frame_region_of_interest, df_0, trajectorized_annotations_0, drawing_settings, elements, frame_finished
    df = map_trajectories_to_vehicles(df, trajectory_vehicle_map)
    with open(map_file, 'w') as file:
        file.write(json.dumps(trajectory_vehicle_map, indent=4))
    print(f'Saved inital mapping!')

with alive_bar(manual=True) as bar:
    max_num_frames, max_time = df['frame_num'].max(), df['time'].max()
    while True:
        df, trajectory_vehicle_map, vehicle_ending, max_coherent_time = find_and_remove_redundant_trajectories(df, trajectory_vehicle_map, map_file)
        bar(max_coherent_time/max_time)
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