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
import json
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import _constants as cs
from scipy.optimize import isotonic_regression

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


def _sort_by_vehicle_number(df: pd.DataFrame, vehicle_label_column="vehicle"):
    df[["vehicleMeaningless","vehicleIndexNumber"]] = df[vehicle_label_column].str.split("_", n=1, expand=True)
    df["vehicleIndexNumber"] = df["vehicleIndexNumber"].astype(int)
    df = df.sort_values(by="vehicleIndexNumber")
    del df["vehicleIndexNumber"]
    del df["vehicleMeaningless"]
    return df


def _determineVehiclesLaneCoordinateStartPositions(trajectory_df, start_frame=0):
    sel = trajectory_df[trajectory_df["frame_nr"]==start_frame]
    sel = _sort_by_vehicle_number(sel, vehicle_label_column="vehicle")
    # assume average radius for calculation
    av_radius = np.median(sel["y_polar"])
    sel["offset"] = sel["x_polar"]*av_radius
    offset_df = sel[["vehicle", "offset"]]
    return offset_df, av_radius


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
            if abs(val - last_val) <= cs.PROCESSING_INTEGRATION_CORRECTION_THRESHOLD and second_last_val is not None:
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


def _filter_median_deviation(series, kernel_size, threshold):
    rolling_median = series.rolling(window=kernel_size, center=True, min_periods=1).median()
    deviation = np.abs(series - rolling_median)
    mask = deviation > threshold
    # Calculate weights
    weight_series = 1 / (np.maximum(deviation - threshold, 0) + 1)
    weight_median = 1 - weight_series
    # Apply weighted average where deviation > threshold
    filtered = pd.Series(index=series.index)
    filtered[mask] = (series[mask] * weight_series[mask] + 
                      rolling_median[mask] * weight_median[mask])
    filtered[~mask] = series[~mask]
    return filtered


def _filterKalmanStates(trajectory_df):
    vehicles = trajectory_df["vehicle"].unique()
    filtered_trajectory_df = pd.DataFrame()
    for vehicle in vehicles:
        vehicle_df = trajectory_df[trajectory_df["vehicle"]==vehicle]
        # CARTESIAN COORDINATES
        # State 1&2 (x&y)
        vehicle_df["state1"] = vehicle_df["state1"].clip(upper=cs.PROCESSING_MAX_POSITION, lower=cs.PROCESSING_MIN_POSITION)
        vehicle_df["state2"] = vehicle_df["state2"].clip(upper=cs.PROCESSING_MAX_POSITION, lower=cs.PROCESSING_MIN_POSITION)
        # Velocity Outlier Correction
        vehicle_df["state1"] = _filter_median_deviation(vehicle_df["state1"], kernel_size=cs.PROCESSING_CARTESIAN_OUTLIER, threshold=cs.PROCESSING_CARTESIAN_THRESHOLD)
        vehicle_df["state2"] = _filter_median_deviation(vehicle_df["state2"], kernel_size=cs.PROCESSING_CARTESIAN_OUTLIER, threshold=cs.PROCESSING_CARTESIAN_THRESHOLD)
        # Cartesian MovingAverage Filter
        vehicle_df["state1"] = vehicle_df["state1"].rolling(window=int(cs.FILTERING_SAMPLING_FREQUENCY/5), center=True, min_periods=1).mean()
        vehicle_df["state2"] = vehicle_df["state2"].rolling(window=int(cs.FILTERING_SAMPLING_FREQUENCY/5), center=True, min_periods=1).mean()
        # ANGLE (State 3)
        # vehicle_df["state3"] = vehicle_df["state3"].rolling(window=int(cs.FILTERING_SAMPLING_FREQUENCY*4), center=True, min_periods=1).mean()
        # MERGE TO COMPLETE DATAFRAME
        filtered_trajectory_df = pd.concat([filtered_trajectory_df, vehicle_df], ignore_index=True)
    return filtered_trajectory_df


def _filterVelocity(vehicle_df):
    # VELOCITY (State 4)
    # Velocity MAX Capping
    vehicle_df["velocity_cartesian"] = vehicle_df["velocity_cartesian"].clip(upper=cs.PROCESSING_MAX_VELOCITY)
    # Velocity Outlier Correction
    vehicle_df["velocity_cartesian"] = _filter_median_deviation(vehicle_df["velocity_cartesian"], kernel_size=int(len(vehicle_df)*cs.POST_FILTERING_KERNEL_A), threshold=cs.PROCESSING_THR_VELOCITY)
    # Velocity Tail Correction
    to_idx = int(len(vehicle_df)*cs.POST_FILTERING_KERNEL_A)
    const_val = vehicle_df["velocity_cartesian"].iloc[to_idx]
    vehicle_df.iloc[0:to_idx+1, vehicle_df.columns.get_loc("velocity_cartesian")] = const_val
    to_idx = len(vehicle_df)-int(len(vehicle_df)*cs.POST_FILTERING_KERNEL_A)
    const_val = vehicle_df["velocity_cartesian"].iloc[to_idx]
    vehicle_df.iloc[to_idx+1:, vehicle_df.columns.get_loc("velocity_cartesian")] = const_val
    return vehicle_df


def processTrajectory_clean(video_trajectory_path: str, vehiclized_file_path: str, 
                      vehicle_proceeding_order_file_path: str, trajectory_type: str):
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

    # Filter Kalman States
    trajectory_df = _filterKalmanStates(trajectory_df)

    # Calculate Polar Coordinates
    trajectory_df["x_polar"] = np.arctan2(-trajectory_df["state2"], trajectory_df["state1"])
    trajectory_df.loc[trajectory_df["x_polar"] <= 0, "x_polar"] += 2*np.pi
    trajectory_df["y_polar"] = np.linalg.norm(np.asarray([trajectory_df["state1"], -trajectory_df["state2"]]), axis=0)

    # Determine Relative Start Positions
    vehicles_lane_coord_start_position, av_radius = _determineVehiclesLaneCoordinateStartPositions(trajectory_df)
    first_vehicle = vehicles_lane_coord_start_position.loc[vehicles_lane_coord_start_position["offset"].idxmin(), "vehicle"]
    trajectory_df = trajectory_df.merge(vehicles_lane_coord_start_position, left_on="vehicle", right_on="vehicle", how="left")
    del vehicles_lane_coord_start_position

    # Calculate space headway in lane coordinates and velocity in cartesian coordinates
    # And Lane Coordinates
    lane_coordinate_df = None
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["vehicle"]==vehicle_id].copy()
        if not vehicle_df["frame_nr"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by="frame_nr", ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])
        prec_vehicle_df = trajectory_df[trajectory_df["vehicle"]==vehicles_proceeding_order[vehicle_id]].copy()
        if not prec_vehicle_df["frame_nr"].is_monotonic_increasing:
            prec_vehicle_df = prec_vehicle_df.sort_values(by="frame_nr", ascending=True)
        prec_vehicle_df = prec_vehicle_df.reset_index().drop(columns=["index"])
        vehicle_df["angle_headway"] = prec_vehicle_df["x_polar"] - vehicle_df["x_polar"]
        vehicle_df.loc[vehicle_df["angle_headway"] < 0, "angle_headway"] += 2*np.pi
        vehicle_df["space_headway_linear"] = np.sqrt((vehicle_df["state1"] - prec_vehicle_df["state1"])**2 + (vehicle_df["state2"] - prec_vehicle_df["state2"])**2)
        vehicle_df["space_headway"] = vehicle_df["space_headway_linear"] * vehicle_df["angle_headway"] / np.sqrt(2*(1-np.cos(vehicle_df["angle_headway"])))
        vehicle_df["y_lane"] = vehicle_df["y_polar"]
        if vehicle_id == first_vehicle:
            vehicle_df["x_lane"] = vehicle_df["x_polar"]*vehicle_df["y_polar"]
            vehicle_df["x_lane"] = _integrate_lane_progress(vehicle_df["x_lane"])
            vehicle_df["x_lane"] = _correctZeroDiffsRepeatedly(vehicle_df["x_lane"])
            vehicle_df["x_lane"] = vehicle_df["x_lane"] + vehicle_df["offset"]
        else:
            vehicle_df["x_lane"] = pd.NA
        sampling_interval = vehicle_df["time"].diff(1).mean()
        vehicle_df["velocity_x"] = vehicle_df["state1"].diff(1).shift(1).fillna(0) / sampling_interval
        vehicle_df["velocity_y"] = vehicle_df["state2"].diff(1).shift(1).fillna(0) / sampling_interval
        vehicle_df["velocity_cartesian"] = np.sqrt(np.square(vehicle_df["velocity_x"]) + np.square(vehicle_df["velocity_y"]))
        vehicle_df = _filterVelocity(vehicle_df)
        vehicle_df = vehicle_df[["frame_nr", "vehicle", "x_lane", "y_lane", "space_headway", "velocity_cartesian"]]
        if lane_coordinate_df is None:
            lane_coordinate_df = vehicle_df.copy()
        else:
            lane_coordinate_df = pd.concat((lane_coordinate_df, vehicle_df))
    lane_coordinate_df = lane_coordinate_df.reset_index()
    lane_coordinate_df = lane_coordinate_df.drop(columns=["index"])
    trajectory_df = trajectory_df.merge(lane_coordinate_df, on=["frame_nr", "vehicle"], how="left")
    del lane_coordinate_df, vehicle_df, prec_vehicle_df
    remaining_vehicles = set(unique_vehicles) - set([first_vehicle])
    vehicle_id = first_vehicle
    vehicle_df = trajectory_df[trajectory_df["vehicle"]==vehicle_id].copy()
    if not vehicle_df["frame_nr"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by="frame_nr", ascending=True)
    vehicle_df = vehicle_df.reset_index()
    trajectory_df_list = [vehicle_df]
    while len(remaining_vehicles) > 0:
        prec_vehicle_df = trajectory_df[trajectory_df["vehicle"]==vehicles_proceeding_order[vehicle_id]].copy()
        if not prec_vehicle_df["frame_nr"].is_monotonic_increasing:
            prec_vehicle_df = prec_vehicle_df.sort_values(by="frame_nr", ascending=True)
        prec_vehicle_df = prec_vehicle_df.reset_index()
        # prec_vehicle_df["x_lane"] = vehicle_df["x_lane"] + vehicle_df["space_headway"]
        # vehicle_df["space_headway"] = prec_vehicle_df["x_lane"] - vehicle_df["x_lane"]
        # Clipping Space Headway ?
        filtered_headway = vehicle_df["space_headway"].copy()
        filtered_headway = filtered_headway.clip(lower=cs.PROCESSING_MIN_HEADWAY_DIST)
        vehicle_df["space_headway"] = filtered_headway
        prec_vehicle_df["x_lane"] = vehicle_df["x_lane"] + vehicle_df["space_headway"]
        # Store results
        trajectory_df_list.append(prec_vehicle_df)
        remaining_vehicles = remaining_vehicles - set([vehicles_proceeding_order[vehicle_id]])
        vehicle_id = vehicles_proceeding_order[vehicle_id]
        vehicle_df = prec_vehicle_df
        
    # Assembly Trajectory DF
    trajectory_df = pd.concat(trajectory_df_list).reset_index().drop(columns=["index", "level_0"])
    del vehicle_df, prec_vehicle_df, trajectory_df_list

    # Calculate Time Headway
    trajectory_df["time_headway"] = trajectory_df["space_headway"] / trajectory_df["velocity_cartesian"]
    first_non_inf = trajectory_df["time_headway"].loc[~np.isinf(trajectory_df["time_headway"])].iloc[0]
    trajectory_df["time_headway"] = trajectory_df["time_headway"].replace({np.inf: first_non_inf}, method='ffill')

    # Finalize Columns
    trajectory_df[["vehicleMeaningless","vehicleIndexNumber"]] = trajectory_df["vehicle"].str.split("_", n=1, expand=True)
    trajectory_df["vehicleIndexNumber"] = trajectory_df["vehicleIndexNumber"].astype(int)
    trajectory_df = trajectory_df.sort_values(by=["frame_nr", "vehicleIndexNumber"])
    trajectory_df = trajectory_df.drop(columns=["vehicleIndexNumber", "vehicleMeaningless"])
    trajectory_df = trajectory_df.rename(columns={    
            "frame_nr": "Frame_ID",
            "time": "Global_Time",
            "state1": "Cartesian_X",
            "state2": "Cartesian_Y",
            "state3": "v_Angle",
            "velocity_cartesian": "v_Vel",
            "state5": "v_AngleVel",
            "width": "v_Width",
            "height": "v_Length",
            "proceeding_vehicle": "Proceeding",
            "x_polar": "Polar_X",
            "y_polar": "Polar_Y",
            "x_lane": "Lane_X",
            "y_lane": "Lane_Y",
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


def processTrajectory_synthetic(trajectory_df):
    """
    This method loads the Kalman filtered trajectories from all vehicles in a given video.

    Parameters
    ----------
    trajectory_df: pd.DataFrame
        The synthetically generated trajectory that was Kalman filtered.
        
    Returns
    -------
    trajectory_df: pd.DataFrame
        The final trajectory dataframe.
    """
    # Load Trajectories For All Vehicles
    unique_vehicles = ["VEHICLE_1"]   
    # Determine Vehicle Dimensions
    trajectory_df["width"] = 5
    trajectory_df["height"] = 5
    # Determine Proceeding Vehicle
    trajectory_df["proceeding_vehicle"] = "VEHICLE_1"
    # Filter Kalman States
    #trajectory_df = _filterKalmanStates(trajectory_df)
    # Calculate Polar Coordinates
    trajectory_df["x_polar"] = np.arctan2(-trajectory_df["state2"], trajectory_df["state1"])
    trajectory_df.loc[trajectory_df["x_polar"] <= 0, "x_polar"] += 2*np.pi
    trajectory_df["y_polar"] = np.linalg.norm(np.asarray([trajectory_df["state1"], -trajectory_df["state2"]]), axis=0)
    # Determine Relative Start Positions
    vehicles_lane_coord_start_position, av_radius = _determineVehiclesLaneCoordinateStartPositions(trajectory_df)
    first_vehicle = vehicles_lane_coord_start_position.loc[vehicles_lane_coord_start_position["offset"].idxmin(), "vehicle"]
    trajectory_df = trajectory_df.merge(vehicles_lane_coord_start_position, left_on="vehicle", right_on="vehicle", how="left")
    del vehicles_lane_coord_start_position
    # Calculate space headway in lane coordinates and velocity in cartesian coordinates
    # And Lane Coordinates
    lane_coordinate_df = None
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["vehicle"]==vehicle_id].copy()
        if not vehicle_df["frame_nr"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by="frame_nr", ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])
        vehicle_df["y_lane"] = vehicle_df["y_polar"]
        if vehicle_id == first_vehicle:
            vehicle_df["x_lane"] = vehicle_df["x_polar"]*vehicle_df["y_polar"]
            vehicle_df["x_lane"] = _integrate_lane_progress(vehicle_df["x_lane"])
            vehicle_df["x_lane"] = _correctZeroDiffsRepeatedly(vehicle_df["x_lane"])
            vehicle_df["x_lane"] = vehicle_df["x_lane"] + vehicle_df["offset"]
        else:
            vehicle_df["x_lane"] = pd.NA
        sampling_interval = vehicle_df["time"].diff(1).mean()
        vehicle_df["velocity_x"] = vehicle_df["state1"].diff(1).shift(1).fillna(0) / sampling_interval
        vehicle_df["velocity_y"] = vehicle_df["state2"].diff(1).shift(1).fillna(0) / sampling_interval
        vehicle_df["velocity_cartesian"] = np.sqrt(np.square(vehicle_df["velocity_x"]) + np.square(vehicle_df["velocity_y"]))
        #vehicle_df = _filterVelocity(vehicle_df)
        vehicle_df = vehicle_df[["frame_nr", "vehicle", "x_lane", "y_lane", "velocity_cartesian"]]
        if lane_coordinate_df is None:
            lane_coordinate_df = vehicle_df.copy()
        else:
            lane_coordinate_df = pd.concat((lane_coordinate_df, vehicle_df))
    lane_coordinate_df = lane_coordinate_df.reset_index()
    lane_coordinate_df = lane_coordinate_df.drop(columns=["index"])
    trajectory_df = trajectory_df.merge(lane_coordinate_df, on=["frame_nr", "vehicle"], how="left")
    # Finalize Columns
    trajectory_df[["vehicleMeaningless","vehicleIndexNumber"]] = trajectory_df["vehicle"].str.split("_", n=1, expand=True)
    trajectory_df["vehicleIndexNumber"] = trajectory_df["vehicleIndexNumber"].astype(int)
    trajectory_df = trajectory_df.sort_values(by=["frame_nr", "vehicleIndexNumber"])
    trajectory_df = trajectory_df.drop(columns=["vehicleIndexNumber", "vehicleMeaningless"])
    trajectory_df = trajectory_df.rename(columns={    
            "frame_nr": "Frame_ID",
            "time": "Global_Time",
            "state1": "Cartesian_X",
            "state2": "Cartesian_Y",
            "state3": "v_Angle",
            "velocity_cartesian": "v_Vel",
            "state5": "v_AngleVel",
            "width": "v_Width",
            "height": "v_Length",
            "proceeding_vehicle": "Proceeding",
            "x_polar": "Polar_X",
            "y_polar": "Polar_Y",
            "x_lane": "Lane_X",
            "y_lane": "Lane_Y",
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
            "v_Length",
            "v_Width",
            "v_Vel",
            "v_Angle",
            "v_AngleVel",
            "Proceeding"
        ]]
    return trajectory_df