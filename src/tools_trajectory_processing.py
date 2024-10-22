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
import os, sys
import pandas as pd
import numpy as np
import json
from tools_filtering_angle import boundAngleListPositive
import _constants as cs

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode



# #############################################################################
# METHODS
# #############################################################################

def _determineVehiclesLaneCoordinateStartPositions(trajectory_df, start_frame=0):
    sel = trajectory_df[trajectory_df["frame_nr"]==start_frame]
    sel = _sort_by_vehicle_number(sel, vehicle_label_column="vehicle")
    # assume average radius for calculation
    av_radius = np.median(sel["y_polar"])
    full_cirumference = 2*np.pi*av_radius
    """
    # calculate difference between car and leader
    sel["lane_temp"] = sel["x_polar"]*av_radius
    sel["lane_temp_diff"] = sel["lane_temp"].diff(1)
    sel = sel.reset_index()
    sel.loc[0, "lane_temp_diff"] = 0
    # convert negative distances according to circumference
    for index, row in sel.iterrows():
        if row["lane_temp_diff"]<0:
            new_val = row["lane_temp_diff"]
            while new_val<0:
                new_val += full_cirumference
            sel.loc[index, "lane_temp_diff"] = new_val
    # TODO: calculate final offset
    sel["offset"] = sel["lane_temp_diff"].cumsum()
    """
    sel["offset"] = sel["x_polar"]*av_radius
    offset_df = sel[["vehicle", "offset"]]
    return offset_df, full_cirumference

def _sort_by_vehicle_number(df: pd.DataFrame, vehicle_label_column="vehicle"):
    df[["vehicleMeaningless","vehicleIndexNumber"]] = df[vehicle_label_column].str.split("_", expand=True)
    df["vehicleIndexNumber"] = df["vehicleIndexNumber"].astype(int)
    df = df.sort_values(by="vehicleIndexNumber")
    del df["vehicleIndexNumber"]
    del df["vehicleMeaningless"]
    return df

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
        new_vals.append((val-diff))
        last_val = val
    return new_vals

def _correctZeroDiffs(vals):
    # vals = sel["x_lane"]
    new_vals = []
    second_last_val = None
    last_val = None
    diff = 0
    for val in vals:
        if last_val is None:
            new_vals.append(val)
        else:
            if abs(val - last_val) <= cs.PROCESSING_INTEGRATION_CORRECTION_THRESHOLD:
                last_diff = last_val - second_last_val
                diff += last_diff
            new_vals.append(val+diff)
        second_last_val = last_val
        last_val = val
    return new_vals

def _correctZeroDiffsRepeatedly(vals):
    for i in range(0, cs.PROCESSING_INTEGRATION_CORRECTION_REPETITIONS):
        vals = _correctZeroDiffs(vals)
    return vals

def processTrajectory(video_trajectory_path: str, vehiclized_file_path: str, 
                      vehicle_proceeding_order_file_path: str, trajectory_type: str, 
                      first_vehicle: str):
    """
    This method loads the Kalman filtered trajectories from all vehicles in a given video.

    Parameters
    ----------
    video_trajectory_path : str
        The path to the folder contianing the Kalman filtered video trajectories for each vehicle.
    vehiclized_file_path: str
        The path to the vehiclized vehicle trajectories.
    vehicle_proceeding_order_file_path: str
        The path to the file that defines the vehicle proceeding_order.
    trajectory_type: str
        The trajectory type. Options: "obb", "hbb".
    first_vehicle: str
        The first vehicle (reference vehicle), that starts at lane position 0 at time 0.
        All other lane coordinates are in reference to this first vehicle's position at time 0.
        
    Returns
    -------
    trajectory_df: pd.DataFrame
        The final trajectory dataframe.
    """
    
    # Load Trajectories For All Vehicles
    trajectory_df, unique_vehicles = _load_all_vehicles_trajectories(video_trajectory_path=video_trajectory_path, trajectory_type=trajectory_type)
    
    # Determine Vehicle Dimensions
    vehicle_widths, vehicle_heights = determineVehicleDimensions(vehiclized_file_path=vehiclized_file_path, unique_vehicles=unique_vehicles)
    trajectory_df["width"] = trajectory_df["vehicle"].map(vehicle_widths)
    trajectory_df["height"] = trajectory_df["vehicle"].map(vehicle_heights)
    
    # Determine Proceeding Vehicle
    vehicles_proceeding_order = json.load(open(vehicle_proceeding_order_file_path, "r"))
    trajectory_df["proceeding_vehicle"] = trajectory_df["vehicle"].map(vehicles_proceeding_order)
    
    # Calculate Polar Coordinates
    trajectory_df["x_polar"] = np.arctan2(-trajectory_df["state2"], trajectory_df["state1"])
    trajectory_df.loc[trajectory_df["x_polar"] <= 0, "x_polar"] += 2*np.pi
    trajectory_df["y_polar"] = np.linalg.norm(np.asarray([trajectory_df["state1"], -trajectory_df["state2"]]), axis=0)
    # trajectory_df["x_polar"] = np.arctan2(trajectory_df["y_cartesian"], trajectory_df["x_cartesian"])
    # trajectory_df["x_polar"] = boundAngleListPositive(trajectory_df["x_polar"], angle_format="rad")
    # trajectory_df["y_polar"] = np.linalg.norm(np.asarray([trajectory_df["x_cartesian"], -trajectory_df["y_cartesian"]]), axis=0)
    
    # Calculate Lane Coordinates
    lane_coordinate_df = None
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["vehicle"]==vehicle_id].copy()
        vehicle_df["x_lane"] = vehicle_df["x_polar"]*vehicle_df["y_polar"]
        vehicle_df["x_lane"] = _integrate_lane_progress(vehicle_df["x_lane"])
        vehicle_df["x_lane"] = _correctZeroDiffsRepeatedly(vehicle_df["x_lane"])
        vehicle_df["y_lane"] = vehicle_df["y_polar"]
        vehicle_df = vehicle_df[["frame_nr", "vehicle", "x_lane", "y_lane"]]
        if lane_coordinate_df is None:
            lane_coordinate_df = vehicle_df.copy()
        else:
            lane_coordinate_df = pd.concat((lane_coordinate_df, vehicle_df))
    lane_coordinate_df = lane_coordinate_df.reset_index()
    del lane_coordinate_df["index"]
    trajectory_df = trajectory_df.merge(lane_coordinate_df, on=["frame_nr", "vehicle"], how="left")
    
    # Determine Relative Start Positions
    vehicles_lane_coord_start_position, full_circumference = _determineVehiclesLaneCoordinateStartPositions(trajectory_df)
    first_vehicle = vehicles_lane_coord_start_position.loc[vehicles_lane_coord_start_position["offset"].idxmax(), "vehicle"]
    
    # Calculate References Lane Coordinates    
    trajectory_df = trajectory_df.merge(vehicles_lane_coord_start_position, left_on="vehicle", right_on="vehicle", how="left")
    trajectory_df["x_lane_ref"] = trajectory_df["x_lane"]+trajectory_df["offset"]
    del trajectory_df["offset"]
    
    # Calculate Space Headway
    vehicle_positions = trajectory_df[["frame_nr", "vehicle", "x_lane_ref"]]
    trajectory_df = trajectory_df.merge(vehicle_positions, left_on=["frame_nr", "proceeding_vehicle"], right_on=["frame_nr", "vehicle"], how="left")
    trajectory_df["space_headway"] = trajectory_df["x_lane_ref_y"] - trajectory_df["x_lane_ref_x"]
    trajectory_df.loc[trajectory_df["vehicle_x"]==first_vehicle, "space_headway"] = trajectory_df["space_headway"] + full_circumference
    del trajectory_df["vehicle_y"]
    del trajectory_df["x_lane_ref_y"]
    trajectory_df = trajectory_df.rename(columns={"vehicle_x": "vehicle", "x_lane_ref_x": "x_lane_ref"})
    
    # Calculate Time Headway
    trajectory_df["time_headway"] = np.abs(trajectory_df["space_headway"] / trajectory_df["state4"])
    
    # Finalize Columns
    trajectory_df = trajectory_df.rename(columns={    
            "frame_nr": "Frame_ID",
            "time": "Global_Time",
            "state1": "Cartesian_X",
            "state2": "Cartesian_Y",
            "state3": "v_Angle",
            "state4": "v_Vel",
            "state5": "v_AngleVel",
            "width": "v_Width",
            "height": "v_Length",
            "proceeding_vehicle": "Proceeding",
            "x_polar": "Polar_X",
            "y_polar": "Polar_Y",
            "x_lane": "Lane_X",
            "y_lane": "Lane_Y",
            "x_lane_ref": "Lane_X_Ref",
            "space_headway": "Space_Hdwy",
            "time_headway": "Time_Hdwy",
            "vehicle": "Vehicle_ID"
        })
    trajectory_df = trajectory_df[[
            "Vehicle_ID",
            "Frame_ID",
            "Global_Time",
            "Cartesian_X",
            "Cartesian_Y",
            "Polar_X",
            "Polar_Y",
            "Lane_X",
            "Lane_Y",
            "Lane_X_Ref",
            "v_Length",
            "v_Width",
            "v_Vel",
            "v_Angle",
            "v_AngleVel",
            "Proceeding",
            "Space_Hdwy",
            "Time_Hdwy"
        ]]
    return trajectory_df