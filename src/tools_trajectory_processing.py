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
import os
import pandas as pd
import numpy as np
import json
from tools_filtering_angle import boundAngleListPositive




# #############################################################################
# METHODS
# #############################################################################

def _load_all_vehicles_trajectories(video_trajectory_path:str, trajectory_type: str):
    """
    This method loads the Kalman filtered trajectories from all vehicles in a given video.

    Parameters
    ----------
    video_trajectory_path : str
        The path to the folder with all Kalman filtered trajectories of a video file.
        
    Returns
    -------
    annotations: pd.DataFrame
        The annotations. Each key represents a frame number. Each value is a list
        of single annotations. Each single annotation is a list of elements,
        including: [annotation_type, position_x, position_y, width, height, angle, confidence].
        The annotation_type depends on the DOTA dataset classification scheme.
        The position, width and height are coordinate-system specific, e.g.
        pixel coordinates, Cartesian coordinates, or lane coordinates.
        The angle is in radians. The confidence is in %, and output of the neural
        network used for annotation.
    unique_vehicles: list[str]
        The list of unique vehicle labels.
    """
    files = os.listdir(video_trajectory_path)
    unique_vehicles = []
    for file in files:
        vehicle = file.replace(".csv","").replace("_hbb","").replace("_obb", "")
        if vehicle not in unique_vehicles:
            unique_vehicles.append(vehicle)
    trajectory_df = None
    for vehicle in unique_vehicles:
        vehicle_df = pd.read_csv(video_trajectory_path+vehicle+"_"+trajectory_type+".csv")
        vehicle_df["vehicle"] = vehicle
        if trajectory_df is None:
            trajectory_df = vehicle_df.copy()
        else:
            trajectory_df = pd.concat((trajectory_df, vehicle_df))
    trajectory_df = trajectory_df.reset_index()
    trajectory_df = trajectory_df[["frame_nr", "time", "state1", "state2", "state3", "state4", "state5", "x", "y", "vehicle"]]
    trajectory_df = trajectory_df.rename(columns={"x": "x_cartesian", "y": "y_cartesian"})
    return trajectory_df, unique_vehicles

def determineVehicleDimensions(vehiclized_file_path: str, unique_vehicles: list):
    """
    This method loads the Kalman filtered trajectories from all vehicles in a given video.

    Parameters
    ----------
    vehiclized_file_path : str
        The path to the vehiclized trajectory file.
    unique_vehicles: list[str]
        The list of unique vehicle labels.
        
    Returns
    -------
    vehicle_widths: dict
        The vehicle widths. The keys represent the vehicle_id, and the value represents the vehicle widths.
    vehicle_heights: dict
        The vehicle heights. The keys represent the vehicle_id, and the value represents the vehicle heights.
    """
    vehiclized_df = pd.read_csv(vehiclized_file_path, header=None, sep="\t", names=["frame_nr", "obj_type", "x", "y", "w", "h", "angle_rad", "confid", "trajectory", "vehicle"])
    vehicle_widths = {}
    vehicle_heights = {}
    for vehicle_id in unique_vehicles:
        vehicle_df = vehiclized_df[vehiclized_df["vehicle"]==vehicle_id]
        vehicle_df["w_vehicle"] = vehicle_df[["w","h"]].min(axis=1)
        vehicle_df["h_vehicle"] = vehicle_df[["w","h"]].max(axis=1)
        median_width = np.median(vehicle_df["w_vehicle"])
        median_height = np.median(vehicle_df["h_vehicle"])
        vehicle_widths[vehicle_id] = median_width
        vehicle_heights[vehicle_id] = median_height
    return vehicle_widths, vehicle_heights

def _integrate_lane_progress(lst_lane_progress):
    new_vals = []
    last_val = 0
    diff = 0
    for val in lst_lane_progress:
        if val-last_val>last_val*0.1:
            diff += val-last_val
        if val-last_val<-last_val*0.1:
            diff += val-last_val
        new_vals.append(-(val-diff))
        last_val = val
    return new_vals




video_trajectory_path = "../data/4_kalman_filtered/DJI_0933/"
vehiclized_file_path = "../data/3_C_vehiclized/DJI_0933.MOV.txt"
vehicle_proceedings_order_file_path = "../data/5_vehicle_relationships/proceeding_order/DJI_0933.MOV.txt"
vehicle_reference_position_file_path = "../data/5_vehicle_relationships/reference_start_distances/DJI_0933.MOV.txt"
trajectory_type = "obb"



# Load Trajectories For All Vehicles
trajectory_df, unique_vehicles = _load_all_vehicles_trajectories(video_trajectory_path=video_trajectory_path, trajectory_type=trajectory_type)

# Determine Vehicle Dimensions
vehicle_widths, vehicle_heights = determineVehicleDimensions(vehiclized_file_path=vehiclized_file_path, unique_vehicles=unique_vehicles)
trajectory_df["width"] = trajectory_df["vehicle"].map(vehicle_widths)
trajectory_df["height"] = trajectory_df["vehicle"].map(vehicle_heights)

# Determine Proceeding Vehicle
vehicles_proceeding_order = json.load(open(vehicle_proceedings_order_file_path, "r"))
trajectory_df["proceeding_vehicle"] = trajectory_df["vehicle"].map(vehicles_proceeding_order)

# Calculate Polar Coordinates
trajectory_df["x_polar"] = np.arctan(trajectory_df["y_cartesian"] / trajectory_df["x_cartesian"])
trajectory_df["x_polar"] = boundAngleListPositive(trajectory_df["x_polar"], angle_format="rad")
trajectory_df["y_polar"] = np.linalg.norm(np.asarray([trajectory_df["x_cartesian"], trajectory_df["y_cartesian"]]), axis=0)

# Calculate Lane Coordinates
lane_coordinate_df = None
for vehicle_id in unique_vehicles:
    vehicle_df = trajectory_df[trajectory_df["vehicle"]==vehicle_id].copy()
    vehicle_df["x_lane"] = -vehicle_df["x_polar"]*vehicle_df["y_polar"]
    vehicle_df["x_lane"] = _integrate_lane_progress(vehicle_df["x_lane"] )
    vehicle_df["y_lane"] = vehicle_df["y_polar"]
    vehicle_df = vehicle_df[["frame_nr", "vehicle", "x_lane", "y_lane"]]
    if lane_coordinate_df is None:
        lane_coordinate_df = vehicle_df.copy()
    else:
        lane_coordinate_df = pd.concat((lane_coordinate_df, vehicle_df))
lane_coordinate_df = lane_coordinate_df.reset_index()
del lane_coordinate_df["index"]
trajectory_df = trajectory_df.merge(lane_coordinate_df, on=["frame_nr", "vehicle"], how="left")
    

# import matplotlib.pyplot as plt
# for vehicle_id in unique_vehicles:
#     sel = trajectory_df[trajectory_df["vehicle"]==vehicle_id]
#     plt.plot(sel["frame_nr"], sel["x_lane"], label=vehicle_id)
# plt.legend()


# Calculate References Lane Coordinates
vehicles_lane_coord_start_position = json.load(open(vehicle_reference_position_file_path, "r"))

trajectory_df["x_lane_ref"] = trajectory_df["x_lane"].copy()
trajectory_df["y_lane_ref"] = trajectory_df["y_lane"].copy()
for vehicle_id in unique_vehicles:
    trajectory_df.loc[trajectory_df["vehicle"]==vehicle_id, "x_lane_ref"] += vehicles_lane_coord_start_position[vehicle_id]
    
import matplotlib.pyplot as plt
for vehicle_id in unique_vehicles:
    sel = trajectory_df[trajectory_df["vehicle"]==vehicle_id]
    plt.plot(sel["frame_nr"], sel["x_lane_ref"], label=vehicle_id)
plt.legend()


# Calculate Space Headway

# Calculate Time Headway

# Vehicle_ID,
# Frame_ID,
# Total_Frames,
# Global_Time,
# Local_X,
# Local_Y,
# Global_X,
# Global_Y,
# v_Length,
# v_Width,
# v_Class,
# v_Vel,
# v_Acc,
# Lane_ID,
# Preceeding,
# Following,
# Space_Hdwy,
# Time_Hdwy
