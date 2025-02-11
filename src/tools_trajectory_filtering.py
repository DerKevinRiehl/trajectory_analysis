"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish, Kevin Riehl
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cvxpy as cp

from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.interpolate._interpolate import PPoly
from scipy.integrate import cumulative_simpson
from scipy.optimize import isotonic_regression
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional

from _constants import FILTERING_SAMPLING_FREQUENCY


# #############################################################################
# METHODS: Constrained Optimization Approach
# #############################################################################
def reconstruct_trajectories_cvxopt(trajectory_df: pd.DataFrame, accel_max_spl: PPoly, decel_min_spl: PPoly,
                                    recon_window: float = 10.0, recon_step: float = 8.0, end_frame: Optional[int] = None,
                                    relax_accel_cnst: bool = True) -> pd.DataFrame:
    """
    This method performs traectory reconstruction/filtering through a constrained optimization problem.
    Each vehicle trajectory is filtered by windowing for a specific time period and then stepping through to the next.
    The trajectory is optimized for physical plausibility and consistency.

    Parameters:
    -----------
        trajectory_df: pd.DataFrame
            contains processed unfiltered trajectories of all vehicles in the experiment
        accel_max_spl: PPoly
            Cubic spline definig the acceleration capacity versus speed curve for a model vehicle
        decel_min_spl: PPoly
            Cubic spline definig the acceleration capacity versus speed curve for a model vehicle
        recon_window: float = 10.0
            Reconstruction window in seconds
        recon_step: float = 8.0
            Step size to move reconstruction window in seconds, needs to less than or equal recon_window
        end_frame: Optional[int] = None
            Ending frame number, Optional integer, This is necessary for video 'DJI_0940.MOV' with end_frame = 6800-1
        relax_accel_cnst: bool = True
            Whether to relax the acceleration constraints or not

    
    Returns:
    --------
        reconstructed_trajectory_df: pd.DataFrame
            contains reconstructed trajectories of all vehicles in the experiment
    """
    reconstructed_trajectory_df = None
    if end_frame is not None:
        trajectory_df = trajectory_df[trajectory_df["Frame_ID"] <= end_frame]

    begin_df = trajectory_df[trajectory_df["Frame_ID"] == 0].copy()
    idx = begin_df["Lane_X"].idxmin()
    idx = begin_df["Lane_X"].idxmax()
    last_vehicle = begin_df.loc[idx, "Vehicle_ID"]
    del idx, begin_df

    remaining_vehicles = set(trajectory_df["Vehicle_ID"].unique())
    vehicle_id = last_vehicle
    prec_vehicle_df = None
    prec_vehicle_length = None

    relax_1_records = []
    relax_2_records = []
    while len(remaining_vehicles) > 0:
        print(f"Now handling {vehicle_id} with {len(remaining_vehicles)} remaining vehicles.")
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])
        
        # Handle noise in x-lane coordinate through an isotonic regression
        if not vehicle_df["Lane_X"].is_monotonic_increasing:
            res = isotonic_regression(vehicle_df["Lane_X"].to_numpy(), increasing=True)
            vehicle_df["Lane_X"] = res.x
        vehicle_df["v_Vel"] = vehicle_df["Lane_X"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
        vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY

        # Handle noise in velocity and acceleration through the optimization procedure
        if vehicle_id != last_vehicle:
            lead_position = prec_vehicle_df["Lane_X"].to_numpy()
        position = vehicle_df["Lane_X"].to_numpy()
        speed = np.diff(position, n=1) * FILTERING_SAMPLING_FREQUENCY
        speed = np.maximum(0, speed)
        accel = np.diff(speed, n=1) * FILTERING_SAMPLING_FREQUENCY               
        w_recon = int(recon_window*FILTERING_SAMPLING_FREQUENCY)
        step = int(recon_step*FILTERING_SAMPLING_FREQUENCY)

        sos = signal.butter(N=1, Wn=0.75, btype='lowpass', fs=FILTERING_SAMPLING_FREQUENCY, output='sos')
        speed_filtered = signal.sosfilt(sos, speed)
        position_recon = np.copy(position)
        speed_recon = np.copy(speed)
        accel_recon = np.copy(accel)

        for k in range(0, len(speed), step):
            x_noised= position_recon[k:k+w_recon+1]
            v_noised = speed_recon[k:k+w_recon]
            v_filtered = speed_filtered[k:k+w_recon]
            
            a_max = max(accel_max_spl(v_noised[0]), accel_max_spl(np.mean(v_filtered)))
            a_driver_max = 0.5 * a_max
            a_min = max(decel_min_spl(v_noised[0]), decel_min_spl(np.mean(v_filtered)))

            v_recon = cp.Variable(shape=(len(v_noised),))
            
            a_recon = cp.diff(v_recon, k=1) * FILTERING_SAMPLING_FREQUENCY
            lam_soft_cnst = 0.05

            if relax_accel_cnst:
                relax_1 = cp.Variable()
                relax_2 = cp.Variable()
                obj = cp.norm(v_recon - v_filtered, 2)**2 + cp.std(a_recon)**2 + cp.sum(lam_soft_cnst * (a_recon/a_driver_max - 1)) + relax_1**2 + relax_2**2
                cnst = [
                    v_recon >= 0,
                    cp.sum(v_recon)/FILTERING_SAMPLING_FREQUENCY == x_noised[-1] - x_noised[0],
                    a_recon - a_min >= relax_1,
                    a_recon - a_max <= relax_2,
                    relax_1 <= 0,
                    relax_2 >= 0
                ]
            else:
                obj = cp.norm(v_recon - v_filtered, 2)**2 + cp.std(a_recon)**2 + cp.sum(lam_soft_cnst * (a_recon/a_driver_max - 1))
                cnst = [
                    v_recon >= 0,
                    cp.sum(v_recon)/FILTERING_SAMPLING_FREQUENCY == x_noised[-1] - x_noised[0],
                    a_recon - a_min >= 0,
                    a_recon - a_max <= 0,
                ]
            if vehicle_id != last_vehicle:
                x_lead = lead_position[k:k+w_recon+1]
                for t in range(len(v_noised)):
                    cnst += [
                        x_lead[t+1] - prec_vehicle_length - (x_noised[0] + cp.sum(v_recon[:t])/FILTERING_SAMPLING_FREQUENCY) >= 1.0
                    ]
            prob = cp.Problem(cp.Minimize(obj), cnst)         
            prob.solve(solver=cp.MOSEK, verbose=False)
            try:
                speed_recon[k:k+w_recon] = np.maximum(0, v_recon.value)
                accel_recon[k:k+w_recon-1] = a_recon.value
                position_recon[k:k+w_recon+1] = np.cumsum(np.hstack([x_noised[0], v_recon.value/FILTERING_SAMPLING_FREQUENCY]))
                if relax_accel_cnst:
                    relax_1_records.append(relax_1.value)
                    relax_2_records.append(relax_2.value)
            except:
                print("ERROR with CvxOpt!")
                sys.exit(1)
        vehicle_df["Lane_X"] = position_recon
        vehicle_df["v_Vel"] = np.hstack((speed_recon, 0))
        vehicle_df["v_Accel"] = np.hstack((accel_recon, [0, 0]))
        if reconstructed_trajectory_df is None:
            reconstructed_trajectory_df = vehicle_df.copy()
        else:
            reconstructed_trajectory_df = pd.concat((reconstructed_trajectory_df, vehicle_df))

        # Transition to following vehicle
        subdf = trajectory_df[trajectory_df["Proceeding"] == vehicle_id].copy()
        follower_id = subdf["Vehicle_ID"].unique()[0]
        del subdf
        remaining_vehicles = remaining_vehicles - set([vehicle_id])
        vehicle_id = follower_id
        prec_vehicle_df = vehicle_df
        prec_vehicle_length = prec_vehicle_df["v_Length"].mean()

    print("Trajectory reconstruction FINISHED.")
    reconstructed_trajectory_df = reconstructed_trajectory_df.reset_index().drop(columns=["index"])
    if trajectory_df["Vehicle_ID"].nunique() > 1:
        reconstructed_trajectory_df = _recompute_headways(reconstructed_trajectory_df)
    if relax_accel_cnst:
        return reconstructed_trajectory_df, np.min(relax_1_records), np.max(relax_2_records)
    else:
        return reconstructed_trajectory_df


# #############################################################################
# METHODS FOR PUNZO ET AL. (2015)
# #############################################################################
def _recompute_headways(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    # av_radius = np.median(trajectory_df["Polar_Y"])
    total_space_hdwys = trajectory_df.groupby(by=["Frame_ID"])["Space_Hdwy"].sum()

    begin_df = trajectory_df[trajectory_df["Frame_ID"] == 0].copy()
    idx = begin_df["Lane_X"].idxmin()
    first_vehicle = begin_df.loc[idx, "Vehicle_ID"]

    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    mod_trajectory_df = None
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])
        
        prec_vehicle_id = vehicle_df["Proceeding"].unique()[0]        
        prec_vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == prec_vehicle_id].copy()
        if not prec_vehicle_df["Frame_ID"].is_monotonic_increasing:
            prec_vehicle_df = prec_vehicle_df.sort_values(by=["Frame_ID"], ascending=True).reset_index()
        prec_vehicle_df = prec_vehicle_df.reset_index().drop(columns=["index"])
        
        old_space_hdwy = vehicle_df["Space_Hdwy"].to_numpy()
        vehicle_df["Space_Hdwy"] = prec_vehicle_df["Lane_X"] - vehicle_df["Lane_X"]
        if prec_vehicle_id == first_vehicle:
            vehicle_df["Space_Hdwy"] += total_space_hdwys.values # 2 * np.pi * av_radius
        vehicle_df["Time_Hdwy"] = vehicle_df["Space_Hdwy"] / vehicle_df["v_Vel"]

        print(
            vehicle_id, 
            ": Max Abs Error = ", np.round(np.amax(abs(old_space_hdwy - vehicle_df["Space_Hdwy"].to_numpy())), 4), 
            " m , Mean Abs Error = ", np.round(np.mean(abs(old_space_hdwy - vehicle_df["Space_Hdwy"].to_numpy())), 4), " m"
        )

        if mod_trajectory_df is None:
            mod_trajectory_df = vehicle_df.copy()
        else:
            mod_trajectory_df = pd.concat((mod_trajectory_df, vehicle_df))
    mod_trajectory_df = mod_trajectory_df.reset_index().drop(columns=["index"])
    return mod_trajectory_df


def _remove_acceleration_outliers(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    min_accel, max_accel = -8.0, 5.0

    filtered_trajectory_df = None
    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])
        
        vehicle_df["v_Vel"] = vehicle_df["Lane_X"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
        vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY

        vehicle_df.loc[vehicle_df["v_Accel"] > max_accel, "Lane_X"] = np.nan
        vehicle_df.loc[vehicle_df["v_Accel"] < min_accel, "Lane_X"] = np.nan
        vehicle_df.loc[vehicle_df["v_Accel"] > max_accel, "v_Accel"] = np.nan
        vehicle_df.loc[vehicle_df["v_Accel"] < min_accel, "v_Accel"] = np.nan
        time = vehicle_df["Global_Time"].to_numpy()
        x_lane = vehicle_df["Lane_X"].to_numpy()
        nan_idxs = np.where(np.isnan(x_lane))[0]
        for i in nan_idxs:
            start_idx = i - FILTERING_SAMPLING_FREQUENCY if i - FILTERING_SAMPLING_FREQUENCY >= 0 else 0
            end_idx = i + FILTERING_SAMPLING_FREQUENCY if i + FILTERING_SAMPLING_FREQUENCY <= len(time) else len(time)
            local_time = time[int(start_idx):int(end_idx)]
            local_x_lane = x_lane[int(start_idx):int(end_idx)]
            local_nan_idxs = np.where(np.isnan(local_x_lane))[0]
            local_time = np.delete(local_time, local_nan_idxs)
            local_x_lane = np.delete(local_x_lane, local_nan_idxs)
            cs = CubicSpline(local_time, local_x_lane, bc_type="natural")
            x_lane[i] = cs(time[i])
        vehicle_df["Lane_X"] = x_lane
        if not vehicle_df["Lane_X"].is_monotonic_increasing:
            res = isotonic_regression(vehicle_df["Lane_X"].to_numpy(), increasing=True)
            vehicle_df["Lane_X"] = res.x
        vehicle_df["v_Vel"] = vehicle_df["Lane_X"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
        vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY

        if filtered_trajectory_df is None:
            filtered_trajectory_df = vehicle_df.copy()
        else:
            filtered_trajectory_df = pd.concat((filtered_trajectory_df, vehicle_df))
    filtered_trajectory_df = filtered_trajectory_df.reset_index().drop(columns=["index"])
    return filtered_trajectory_df


def _filter_speed_butterworth(trajectory_df: pd.DataFrame, cutoff_freq: float = 0.75) -> pd.DataFrame:
    sos = signal.butter(N=1, Wn=cutoff_freq, btype='lowpass', fs=FILTERING_SAMPLING_FREQUENCY, output='sos')

    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    filtered_trajectory_df = None
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])
        
        #fig, axs = plt.subplots(1, 3)
        #axs[0].plot(vehicle_df["Global_Time"], vehicle_df["Lane_X"], label="Before", color="black")
        #axs[1].plot(vehicle_df["Global_Time"], vehicle_df["v_Vel"], label="Before", color="black")
        #axs[2].plot(vehicle_df["Global_Time"], vehicle_df["v_Accel"], label="Before", color="black")

        vehicle_df["v_Vel"] = signal.sosfilt(sos, vehicle_df["v_Vel"].to_numpy())
        vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY

        speed = vehicle_df["v_Vel"].to_numpy()
        position = vehicle_df["Lane_X"].to_numpy()
        position = np.hstack((position[0], speed[:-1] / FILTERING_SAMPLING_FREQUENCY)).cumsum()
        vehicle_df["Lane_X"] = position
        if not vehicle_df["Lane_X"].is_monotonic_increasing:
            res = isotonic_regression(vehicle_df["Lane_X"].to_numpy(), increasing=True)
            vehicle_df["Lane_X"] = res.x
            vehicle_df["v_Vel"] = vehicle_df["Lane_X"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
            vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY

        #axs[0].plot(vehicle_df["Global_Time"], vehicle_df["Lane_X"], label="After", color="red", alpha=0.5)
        #axs[1].plot(vehicle_df["Global_Time"], vehicle_df["v_Vel"], label="After", color="red", alpha=0.5)
        #axs[2].plot(vehicle_df["Global_Time"], vehicle_df["v_Accel"], label="After", color="red", alpha=0.5)

        #plt.legend()
        #plt.show()

        if filtered_trajectory_df is None:
            filtered_trajectory_df = vehicle_df.copy()
        else:
            filtered_trajectory_df = pd.concat((filtered_trajectory_df, vehicle_df))
        
    filtered_trajectory_df = filtered_trajectory_df.reset_index().drop(columns=["index"])
    return filtered_trajectory_df


def _reconstruct_trajectories(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    reconstructed_trajectory_df = None

    begin_df = trajectory_df[trajectory_df["Frame_ID"] == 0].copy()
    idx = begin_df["Lane_X"].idxmin()
    first_vehicle = begin_df.loc[idx, "Vehicle_ID"]
    idx = begin_df["Lane_X"].idxmax()
    last_vehicle = begin_df.loc[idx, "Vehicle_ID"]
    del idx, begin_df

    remaining_vehicles = set(trajectory_df["Vehicle_ID"].unique())
    vehicle_id = last_vehicle
    while len(remaining_vehicles) > 0:
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])

        prec_vehicle_id = vehicle_df["Proceeding"].unique()[0]
        prec_vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == prec_vehicle_id].copy()
        if not prec_vehicle_df["Frame_ID"].is_monotonic_increasing:
            prec_vehicle_df = prec_vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        prec_vehicle_df = prec_vehicle_df.reset_index().drop(columns=["index"])
        prec_vehicle_length = prec_vehicle_df["v_Length"].mean()

        vehicle_df["Mean_Speed"] = vehicle_df["Lane_X"].diff(1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
        vehicle_df["Mean_Accel"] = vehicle_df["Mean_Speed"].diff(1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
        vehicle_df["Accel_Outlier"] = False
        a_min, a_max, gamma, s_min = -3.0, 2.0, int(0.5*FILTERING_SAMPLING_FREQUENCY), 1.0
        vehicle_df.loc[vehicle_df["Mean_Accel"] > a_max, "Accel_Outlier"] = True
        vehicle_df.loc[vehicle_df["Mean_Accel"] < a_min, "Accel_Outlier"] = True
        vehicle_subdf = vehicle_df[vehicle_df["Accel_Outlier"] == True]

        # Local Reconstruction Window Calculation
        vehicle_df["Recon_Window"] = pd.NA
        for idx, _ in vehicle_subdf.iterrows():
            speed = vehicle_df.loc[idx:, "Mean_Speed"].to_numpy()
            accel = vehicle_df.loc[idx:, "Mean_Accel"].to_numpy()
            x = vehicle_df.loc[idx:, "Lane_X"].to_numpy()
            x_lead = prec_vehicle_df.loc[idx:, "Lane_X"].to_numpy()
            kappa = 1
            while kappa <= min(50, len(speed)-gamma):
                av_accel_kappa = np.mean(accel[1:kappa+1])
                if av_accel_kappa < a_min or av_accel_kappa > a_max:
                    kappa = kappa + 1
                    continue
                av_accel_gamma = np.mean(accel[kappa:kappa+gamma+1])
                if av_accel_gamma < a_min or av_accel_gamma > a_max:
                    kappa = kappa + 1
                    continue
                if vehicle_id == first_vehicle:
                    print("Handle first-last vehicle case!!")
                    sys.exit(1)
                elif vehicle_id != last_vehicle:
                    if np.all(x_lead[kappa:kappa+gamma+1] - prec_vehicle_length - x[kappa:kappa+gamma+1] >= s_min):
                        break
                    else:
                        kappa = kappa + 1
            # reached optimal kappa
            vehicle_df.loc[idx, "Recon_Window"] = kappa
        
        # Local Reconstruction
        
        sys.exit(1)

        # Transition to following vehicle
        subdf = trajectory_df[trajectory_df["Proceeding"] == vehicle_id].copy()
        follower_id = subdf["Vehicle_ID"].unique()[0]
        del subdf
        remaining_vehicles = remaining_vehicles - set(vehicle_id)
        vehicle_id = follower_id
        
        



    return reconstructed_trajectory_df


def filter_and_reconstruct_trajectories(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    filtered_trajectory_df = _remove_acceleration_outliers(trajectory_df)
    filtered_trajectory_df = _filter_speed_butterworth(filtered_trajectory_df, cutoff_freq=0.55)
    #filtered_trajectory_df = _reconstruct_trajectories(filtered_trajectory_df)
    filtered_trajectory_df = _recompute_headways(filtered_trajectory_df)
    return filtered_trajectory_df



# #############################################################################
# METHODS FOR MAKRIDIS & KOUVELAS (2020)
# #############################################################################
def _label_speed_acceleration_region(trajectory_df: pd.DataFrame, common_driving_style: float = 0.5) -> pd.DataFrame:
    with open('../data_trajectories/5_vehicle_information/vehicle_dynamics/accel_capacity_interpolator.pkl', 'rb') as f:
        accel_max_spl = pickle.load(f)
    with open('../data_trajectories/5_vehicle_information/vehicle_dynamics/decel_capacity_interpolator.pkl', 'rb') as f:
        decel_min_spl = pickle.load(f)
    
    trajectory_df["Region"] = pd.NA
    for idx, row in trajectory_df.iterrows():
        speed = row["v_Vel"]
        accel = row["v_Accel"]
        a_max = accel_max_spl(speed)
        d_min = decel_min_spl(speed)
        if accel > a_max:
            trajectory_df.loc[idx, "Region"] = "A"
        elif accel >= common_driving_style*a_max:
            trajectory_df.loc[idx, "Region"] = "B"
        elif accel >= d_min:
            trajectory_df.loc[idx, "Region"] = "C"
        else:
            trajectory_df.loc[idx, "Region"] = "D"
    
    return trajectory_df


def _vehicle_dynamics_constraint(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    with open('../data_trajectories/5_vehicle_information/vehicle_dynamics/accel_capacity_interpolator.pkl', 'rb') as f:
        accel_max_spl = pickle.load(f)
    with open('../data_trajectories/5_vehicle_information/vehicle_dynamics/decel_capacity_interpolator.pkl', 'rb') as f:
        decel_min_spl = pickle.load(f)
    
    if "Region" not in trajectory_df.columns:
        trajectory_df = _label_speed_acceleration_region(trajectory_df)
    
    subdf = trajectory_df[(trajectory_df["Region"] == "A") | (trajectory_df["Region"] == "D")]
    modified_ratio = len(subdf) / len(trajectory_df)
    for idx, row in subdf.iterrows():
        speed = row["v_Vel"]
        region = row["Region"]
        if region == "A":
            trajectory_df.loc[idx, "v_Accel"] = accel_max_spl(speed)
            trajectory_df.loc[idx, "Region"] = "B"
        elif region == "D":
            trajectory_df.loc[idx, "v_Accel"] = decel_min_spl(speed)
            trajectory_df.loc[idx, "Region"] = "C"
    
    return trajectory_df, modified_ratio
    """
    filtered_trajectory_df = None
    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])

        accel = vehicle_df["v_Accel"].to_numpy()
        speed = vehicle_df["v_Vel"].to_numpy()
        #speed = np.hstack((speed[0], accel[:-1] / FILTERING_SAMPLING_FREQUENCY)).cumsum()
        speed = cumulative_simpson(accel, dx=1/FILTERING_SAMPLING_FREQUENCY, initial=speed[0])
        position = vehicle_df["Lane_X"].to_numpy()
        #position = np.hstack((position[0], speed[:-1] / FILTERING_SAMPLING_FREQUENCY)).cumsum()
        position = cumulative_simpson(speed, dx=1/FILTERING_SAMPLING_FREQUENCY, initial=position[0])
        vehicle_df["Lane_X"] = position
        if not vehicle_df["Lane_X"].is_monotonic_increasing:
            res = isotonic_regression(vehicle_df["Lane_X"].to_numpy(), increasing=True)
            vehicle_df["Lane_X"] = res.x
            vehicle_df["v_Vel"] = vehicle_df["Lane_X"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
            vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
        else:
            vehicle_df["v_Vel"] = speed
            vehicle_df["v_Accel"] = accel
        
        if filtered_trajectory_df is None:
            filtered_trajectory_df = vehicle_df.copy()
        else:
            filtered_trajectory_df = pd.concat((filtered_trajectory_df, vehicle_df))

    filtered_trajectory_df = filtered_trajectory_df.reset_index().drop(columns=["index"])
    return filtered_trajectory_df, modified_ratio
    """


def _driver_dynamics_compliance(trajectory_df: pd.DataFrame, time_window: float = 1.5):
    if "Region" not in trajectory_df.columns:
        trajectory_df = _label_speed_acceleration_region(trajectory_df)
    subdf = trajectory_df[trajectory_df["Region"] == "C"]
    region_c_ratio = len(subdf) / len(trajectory_df)

    w_var = int(time_window * FILTERING_SAMPLING_FREQUENCY)
    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    median_local_accel_std = np.zeros(shape=(len(unique_vehicles),))
    for i in range(len(unique_vehicles)):
        vehicle_id = unique_vehicles[i]
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])
        accel = vehicle_df["v_Accel"].to_numpy()
        accel = np.concatenate((np.zeros(shape=(w_var,)), accel, np.zeros(shape=(w_var,))))
        windowed_accel = sliding_window_view(accel, 2*w_var+1)
        median_local_accel_std[i] = np.median(np.std(windowed_accel[w_var:-w_var, :], axis=1))
 
    return region_c_ratio, median_local_accel_std


def _noise_reduction_butterworth_filter(trajectory_df: pd.DataFrame, cutoff_freq: float) -> pd.DataFrame:
    sos = signal.butter(N=1, Wn=cutoff_freq, btype='lowpass', fs=FILTERING_SAMPLING_FREQUENCY, output='sos')

    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    filtered_trajectory_df = None
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        vehicle_df = vehicle_df.reset_index().drop(columns=["index"])

        vehicle_df["v_Vel"] = signal.sosfilt(sos, vehicle_df["v_Vel"].to_numpy())
        vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
        speed = vehicle_df["v_Vel"].to_numpy()
        position = vehicle_df["Lane_X"].to_numpy()
        #position = np.hstack((position[0], speed[:-1] / FILTERING_SAMPLING_FREQUENCY)).cumsum()
        position = cumulative_simpson(y=speed, dx=1/FILTERING_SAMPLING_FREQUENCY, initial=position[0])
        vehicle_df["Lane_X"] = position
        if not vehicle_df["Lane_X"].is_monotonic_increasing:
            res = isotonic_regression(position, increasing=True)
            vehicle_df["Lane_X"] = res.x
            vehicle_df["v_Vel"] = vehicle_df["Lane_X"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY
            vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) * FILTERING_SAMPLING_FREQUENCY

        if filtered_trajectory_df is None:
            filtered_trajectory_df = vehicle_df.copy()
        else:
            filtered_trajectory_df = pd.concat((filtered_trajectory_df, vehicle_df))
        
    filtered_trajectory_df = filtered_trajectory_df.reset_index().drop(columns=["index"])
    return filtered_trajectory_df


def apply_physics_informed_butterworth_filter(trajectory_df: pd.DataFrame, 
                                              init_cutoff_freq = 0.9, max_filter_strength = 0.05, cutoff_freq_step: float = 0.05,
                                              region_c_threshold: float = 0.95) -> pd.DataFrame:
    filtered_trajectory_df = _label_speed_acceleration_region(trajectory_df)
    filtered_trajectory_df, modified_ratio = _vehicle_dynamics_constraint(filtered_trajectory_df)
    region_c_ratio, median_local_accel_std = _driver_dynamics_compliance(filtered_trajectory_df)
    print(region_c_ratio, modified_ratio)

    if region_c_ratio >= region_c_threshold:
        print("Done Iteration 0")
        filtered_trajectory_df = filtered_trajectory_df.drop(columns=["Region"])
        filtered_trajectory_df = _recompute_headways(filtered_trajectory_df)
        return filtered_trajectory_df
    
    prev_median_local_accel_std = median_local_accel_std
    prev_fw = None
    cutoff_freq = init_cutoff_freq
    for iter in range(20):
        if cutoff_freq < max_filter_strength:
            print("Reached maximum filter strength!")
            break

        filtered_trajectory_df = _noise_reduction_butterworth_filter(trajectory_df, cutoff_freq=cutoff_freq)
        filtered_trajectory_df = _label_speed_acceleration_region(filtered_trajectory_df)
        filtered_trajectory_df, modified_ratio = _vehicle_dynamics_constraint(filtered_trajectory_df)
        region_c_ratio, median_local_accel_std = _driver_dynamics_compliance(filtered_trajectory_df)
        fw = (prev_median_local_accel_std - median_local_accel_std) / median_local_accel_std

        if prev_fw is not None and region_c_ratio >= region_c_threshold and np.all(fw >= prev_fw):
            print(f'Done at iteration {iter} with optimal frequency {cutoff_freq}')
            print(region_c_ratio, modified_ratio)
            break

        prev_fw = fw
        prev_median_local_accel_std = median_local_accel_std
        cutoff_freq = init_cutoff_freq - (iter+1)*cutoff_freq_step
    
    filtered_trajectory_df = filtered_trajectory_df.drop(columns=["Region"])
    filtered_trajectory_df = _recompute_headways(filtered_trajectory_df)
    return filtered_trajectory_df