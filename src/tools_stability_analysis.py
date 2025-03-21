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
import mplcursors
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cvxpy as cp
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional, Tuple


# #############################################################################
# METHODS
# #############################################################################
"""
def filter_trajectories_ema(trajectory_df:pd.DataFrame, alpha: float) -> pd.DataFrame:
    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    mod_trajectory_df = None
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"])
        sampling_interval = vehicle_df["Global_Time"].diff(1).mean()
        vehicle_df["v_Vel"] = vehicle_df["Lane_X"].diff(1).shift(-1).fillna(0) / sampling_interval
        vehicle_df["v_Vel_ema"] = vehicle_df["v_Vel"].ewm(alpha=alpha, adjust=False).mean()
        vehicle_df["v_Vel"] = vehicle_df["v_Vel_ema"]
        vehicle_df = vehicle_df.drop(columns=["v_Vel_ema"])
        vehicle_df["Time_Hdwy"] = vehicle_df["Space_Hdwy"] / vehicle_df["v_Vel"]
        if mod_trajectory_df is None:
            mod_trajectory_df = vehicle_df.copy()
        else:
            mod_trajectory_df = pd.concat((mod_trajectory_df, vehicle_df))
    return mod_trajectory_df
"""

def speed_standard_deviation(trajectory_df: pd.DataFrame, start_frame: Optional[int] = 0) -> pd.DataFrame:
    first_vehicle_idx = trajectory_df.loc[trajectory_df["Frame_ID"] == start_frame, "Lane_X"].idxmin()
    vehicle_id = trajectory_df.loc[first_vehicle_idx, "Vehicle_ID"]
    v_rank =  0

    num_vehicles = trajectory_df["Vehicle_ID"].nunique()
    remaining_vehicles = set(trajectory_df["Vehicle_ID"].unique())
    results = []
    while len(remaining_vehicles) > 0:
        vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
        v_std = vehicle_df["v_Vel"].std()
        results.append([vehicle_id, v_rank, v_std])
        remaining_vehicles = remaining_vehicles - set([vehicle_id])
        vehicle_id = vehicle_df["Proceeding"].unique()[0]
        v_rank = v_rank - 1 if v_rank > 0 else v_rank - 1 + num_vehicles

    speed_std_df = pd.DataFrame(results, columns=["Vehicle_ID", "Vehicle_Rank", "Speed_Std_Dev"])
    speed_std_df = speed_std_df.sort_values(by=["Vehicle_Rank"])

    """
    plt.figure()
    sns.scatterplot(data=speed_std_df, x="Vehicle_Rank", y="Speed_Std_Dev")
    p = np.polyfit(speed_std_df["Vehicle_Rank"].to_numpy(), speed_std_df["Speed_Std_Dev"].to_numpy(), deg=2)
    vals = np.poly1d(p)(speed_std_df["Vehicle_Rank"].to_numpy())
    plt.plot(speed_std_df["Vehicle_Rank"].to_numpy(), vals, "r--")
    plt.show()
    """
    
    return speed_std_df


def speed_dft(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    start_frame, end_frame = trajectory_df["Frame_ID"].min(), trajectory_df["Frame_ID"].max()
    num_frames = end_frame - start_frame + 1
    end_frame -= int(0.1 * num_frames)
    start_frame += int(0.1 * num_frames)

    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    results = []
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[(trajectory_df["Vehicle_ID"] == vehicle_id) & (trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
        sampling_interval = vehicle_df["Global_Time"].diff(1).mean()
        num_pts = len(vehicle_df)

        freqs = np.fft.fftfreq(num_pts, d=sampling_interval)
        p = np.polyfit(vehicle_df["Global_Time"].to_numpy(), vehicle_df["v_Vel"].to_numpy(), deg=1)
        Vel_Trend = np.poly1d(p)(vehicle_df["Global_Time"].to_numpy())
        Vel_DeTrended = vehicle_df["v_Vel"].to_numpy() - Vel_Trend
        y_fft = np.fft.fft(Vel_DeTrended * np.hanning(num_pts))

        index = np.argmax(np.abs(y_fft))
        magnitude = np.abs(y_fft[index]) * 2.0/num_pts
        freq_max = abs(freqs[index])

        results.append([vehicle_id, magnitude, freq_max, 1/(2*np.pi*freq_max)])
        #print(f"{vehicle_id}: magnitude = {magnitude}, freq_max = {freq_max} Hz, period = {1 / (2*np.pi*freq_max)} s")
        #plt.figure()
        #plt.scatter(freqs[0:num_pts//2], 2.0/num_pts * np.abs(y_fft[0:num_pts//2]))
        #plt.show()
    
    dft_df = pd.DataFrame(results, columns=["Vehicle_ID", "Max_Amplitude", "Frequency", "Period"])
    return dft_df


def _get_equilibrium_velocity(leader_velocity: np.ndarray, sampling_interval: Optional[float] = 0.04) -> np.ndarray:
    time_arr = np.arange(0, len(leader_velocity)*sampling_interval, sampling_interval)
    if len(time_arr) > len(leader_velocity):
        time_arr = time_arr[:-1]
    p = np.polyfit(time_arr, leader_velocity, deg=1)
    Vel_Trend = np.poly1d(p)(time_arr)
    Vel_DeTrended = leader_velocity - Vel_Trend
    v_Eq = np.zeros_like(leader_velocity)
    stepsize = int(50.0 / sampling_interval)
    for i in range(0, len(leader_velocity), stepsize):
        if i+stepsize >= len(leader_velocity):
            v_Eq[i:] = Vel_Trend[i:] + np.median(Vel_DeTrended[i:])
        else:
            v_Eq[i:i+stepsize] = Vel_Trend[i:i+stepsize] + np.median(Vel_DeTrended[i:i+stepsize])
    return v_Eq
    """
    v_Eq = np.zeros_like(leader_velocity)
    stepsize = int(20.0 / sampling_interval)
    for i in range(0, len(leader_velocity), stepsize):
        if i+stepsize >= len(leader_velocity):
            v_Eq[i:] = np.median(leader_velocity[i:])
        else:
            v_Eq[i:i+stepsize] = np.median(leader_velocity[i:i+stepsize])
    return v_Eq
    """


def _get_correlation_matrix(X: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    N = X.shape[0]
    Tmat = np.zeros(shape=(N+m-1, m))
    for j in range(m):
        Tmat[j:j+N, j] = X
    Rmat = Tmat.T @ Tmat / N
    return Tmat, Rmat


def estimate_L2gain_CTHpolicy(trajectory_df: pd.DataFrame, start_frame: Optional[int] = 0, end_frame: Optional[int] = None):
    if end_frame is None:
        end_frame = trajectory_df["Frame_ID"].max()-1
    
    standstill_distance = 2.0
    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    gammaSquaredVel, gammaSquaredSpHdwy = {}, {}
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[(trajectory_df["Vehicle_ID"] == vehicle_id) & (trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)

        prec_vehicle_id = vehicle_df["Proceeding"].unique()[0]
        prec_vehicle_df = trajectory_df[(trajectory_df["Vehicle_ID"] == prec_vehicle_id) & (trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
        if not prec_vehicle_df["Frame_ID"].is_monotonic_increasing:
            prec_vehicle_df = prec_vehicle_df.sort_values(by=["Frame_ID"], ascending=True)

        prec_prec_vehicle_id = prec_vehicle_df["Proceeding"].unique()[0]
        prec_prec_vehicle_df = trajectory_df[(trajectory_df["Vehicle_ID"] == prec_prec_vehicle_id) & (trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
        if not prec_prec_vehicle_df["Frame_ID"].is_monotonic_increasing:
            prec_prec_vehicle_df = prec_prec_vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        
        Leader_Velocity = prec_vehicle_df["v_Vel"].to_numpy()
        Ego_Velocity = vehicle_df["v_Vel"].to_numpy()
        Space_Hdwy = vehicle_df["Space_Hdwy"].to_numpy()
        Prec_Space_Hdwy = prec_vehicle_df["Space_Hdwy"].to_numpy()
        Prec_Leader_Velocity = prec_prec_vehicle_df["v_Vel"].to_numpy()

        Vel_Eq = _get_equilibrium_velocity(Leader_Velocity)
        Prec_Vel_Eq = _get_equilibrium_velocity(Prec_Leader_Velocity)

        vehicle_length = vehicle_df["v_Length"].unique()[0]
        Time_Gap = (Space_Hdwy - vehicle_length - standstill_distance) / Ego_Velocity
        timeGap_Eq = np.median(Time_Gap)
        Space_Hdwy_Eq = vehicle_length + standstill_distance + timeGap_Eq * Vel_Eq

        prec_vehicle_length = prec_vehicle_df["v_Length"].unique()[0]
        Prec_Time_Gap = (Prec_Space_Hdwy - prec_vehicle_length - standstill_distance) / Leader_Velocity
        prec_timeGap_Eq = np.median(Prec_Time_Gap)
        Prec_Space_Hdwy_Eq = prec_vehicle_length + standstill_distance + prec_timeGap_Eq * Prec_Vel_Eq

        m = int(0.001*len(Leader_Velocity))
        gain_est = cp.Variable()
        _, Ru = _get_correlation_matrix(Leader_Velocity-Vel_Eq, m)
        _, Ry = _get_correlation_matrix(Ego_Velocity-Vel_Eq, m)
        cnst = [
            Ry - gain_est*Ru <= 0,
            gain_est >= 0
        ]
        obj = gain_est
        prob = cp.Problem(cp.Minimize(obj), cnst)
        prob.solve(solver=cp.MOSEK)
        gammaSquaredVel[vehicle_id] = gain_est.value.item()

        gain_est = cp.Variable()
        _, Ru = _get_correlation_matrix(Prec_Space_Hdwy-Prec_Space_Hdwy_Eq, m)
        _, Ry = _get_correlation_matrix(Space_Hdwy-Space_Hdwy_Eq, m)
        cnst = [
            Ry - gain_est*Ru <= 0,
            gain_est >= 0
        ]
        obj = gain_est
        prob = cp.Problem(cp.Minimize(obj), cnst)
        prob.solve(solver=cp.MOSEK)
        gammaSquaredSpHdwy[vehicle_id] = gain_est.value.item()
    
    gammaSquaredVel = pd.DataFrame(gammaSquaredVel.items(), columns=["Vehicle_ID", "gammaSquared_Speed"])
    gammaSquaredVel["L2gain_Speed"] = np.sqrt(gammaSquaredVel["gammaSquared_Speed"])

    gammaSquaredSpHdwy = pd.DataFrame(gammaSquaredSpHdwy.items(), columns=["Vehicle_ID", "gammaSquared_SpaceHdwy"])
    gammaSquaredSpHdwy["L2gain_SpaceHdwy"] = np.sqrt(gammaSquaredSpHdwy["gammaSquared_SpaceHdwy"])

    gammaSquared = gammaSquaredVel.merge(gammaSquaredSpHdwy, on=["Vehicle_ID"], how="left")
    return gammaSquared


def estimate_LinfinityGain_CTHpolicy(trajectory_df: pd.DataFrame, start_frame: Optional[int] = 0, end_frame: Optional[int] = None):
    if end_frame is None:
        end_frame = trajectory_df["Frame_ID"].max()-1
    
    standstill_distance = 2.0
    unique_vehicles = trajectory_df["Vehicle_ID"].unique()
    LinfinityGain = {}
    for vehicle_id in unique_vehicles:
        vehicle_df = trajectory_df[(trajectory_df["Vehicle_ID"] == vehicle_id) & (trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
        if not vehicle_df["Frame_ID"].is_monotonic_increasing:
            vehicle_df = vehicle_df.sort_values(by=["Frame_ID"], ascending=True)

        prec_vehicle_id = vehicle_df["Proceeding"].unique()[0]
        prec_vehicle_df = trajectory_df[(trajectory_df["Vehicle_ID"] == prec_vehicle_id) & (trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
        if not prec_vehicle_df["Frame_ID"].is_monotonic_increasing:
            prec_vehicle_df = prec_vehicle_df.sort_values(by=["Frame_ID"], ascending=True)
        
        Leader_Velocity = prec_vehicle_df["v_Vel"].to_numpy()
        Ego_Velocity = vehicle_df["v_Vel"].to_numpy()
        Space_Hdwy = vehicle_df["Space_Hdwy"].to_numpy()

        Vel_Eq = _get_equilibrium_velocity(Leader_Velocity)

        vehicle_length = vehicle_df["v_Length"].unique()[0]
        Time_Gap = (Space_Hdwy - vehicle_length - standstill_distance) / Ego_Velocity
        timeGap_Eq = np.median(Time_Gap)
        Space_Hdwy_Eq = vehicle_length + standstill_distance + timeGap_Eq * Vel_Eq

        eta = np.amax(Ego_Velocity-Vel_Eq) / np.amax(Leader_Velocity-Vel_Eq)
        LinfinityGain[vehicle_id] = eta
    
    LinfinityGain = pd.DataFrame(LinfinityGain.items(), columns=["Vehicle_ID", "LinfGain"])
    return LinfinityGain