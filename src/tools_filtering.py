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
from tools_filtering_angle import _estimateBestAngle, _nextBestGuess_45DEGREE_Step, boundAngleListPositive, _takeBestAngle
from tools_filtering_kalman import kalmanFilterTrajectory
import _constants as cs
import numpy as np
import pandas as pd




# #############################################################################
# METHODS
# #############################################################################

# ########## PREPROCESSING METHODS

def _rollingMA_limHorizon(df_orig, av_field, ma_field):
    lst_ma_val = []
    for frame_nr in df_orig["frame_nr"].tolist():
        df_sub = df_orig[df_orig["frame_nr"]<=frame_nr+int(cs.VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)/2]
        df_sub = df_sub[df_sub["frame_nr"]>=frame_nr-int(cs.VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH)/2]
        lst_ma_val.append(np.nanmean(df_sub[av_field]))
    df_orig[ma_field] = lst_ma_val
    return df_orig

def _deltaForward(df, col, new_col, speed_delta):
    new_vals = []
    vals = df[col].tolist()
    frams = df["frame_nr"].tolist()
    for idx in range(0, len(vals)):
        if idx>=speed_delta:
            this_frame = frams[idx]
            other_frame = frams[idx-speed_delta]
            if not abs(this_frame-other_frame)>speed_delta:                
                new_vals.append(vals[idx]-vals[idx-speed_delta])
            else:
                foundOj = -1
                for oj in range(speed_delta, 0, -1):
                    other_frame = frams[idx-oj]
                    if not abs(this_frame-other_frame)>speed_delta:                
                        foundOj = oj
                        break
                if foundOj==-1:
                    new_vals.append(np.nan)
                else:
                    new_vals.append(vals[idx]-vals[idx-foundOj])
        else:
            new_vals.append(np.nan)
    df[new_col] = new_vals
    return df        

def _deltaBackward(df, col, new_col, speed_delta):
    new_vals = []
    vals = df[col].tolist()
    frams = df["frame_nr"].tolist()
    for idx in range(0, len(vals)):
        if idx<=len(vals)-speed_delta-1:
            this_frame = frams[idx]
            other_frame = frams[idx+speed_delta]
            if not abs(this_frame-other_frame)>speed_delta:       
                new_vals.append(vals[idx]-vals[idx+speed_delta])
            else:
                foundOj = -1
                for oj in range(speed_delta, 0, -1):
                    other_frame = frams[idx+oj]
                    if not abs(this_frame-other_frame)>speed_delta:                
                        foundOj = oj
                        break
                if foundOj==-1:
                    new_vals.append(np.nan)
                else:
                    new_vals.append(vals[idx]-vals[idx+foundOj])
        else:
            new_vals.append(np.nan)
    df[new_col] = new_vals
    return df       

def featureCalculation(veh_annotations: pd.DataFrame, video_frames_per_second: int, obb=False):
    """
    This method prepares the vehicle annotations for Kalman filtering, by calculating various features.
    These features include time, moving average filtered positions, velocities, and pre-processed angle information.
    
    Parameters
    ----------
    veh_annotations : pd.DataFrame
        The vehicle annotations.
    VIDEO_FRAMES_PER_SECOND : int
        The frames per second.
    obb: bool
        Whether OBB annotations were provided (including angular information) or not. Default: False
        
    Returns
    -------
    featured_veh_annotations : pd.DataFrame
        The featured vehicle annotations.
    """
    speed_delta = cs.SPEED_ESTIMATION_TIME_HORIZON
    featured_veh_annotations = veh_annotations.copy()
    # Calculate Time Features
    featured_veh_annotations["time"] = featured_veh_annotations["frame_nr"]*(1/video_frames_per_second)
    featured_veh_annotations["delta_time"] = featured_veh_annotations["time"].diff()
    featured_veh_annotations["delta_timeX"] = featured_veh_annotations["time"].diff(speed_delta)
    featured_veh_annotations = _deltaForward(featured_veh_annotations, "time", "delta_time_forward", speed_delta)
    featured_veh_annotations = _deltaBackward(featured_veh_annotations, "time", "delta_time_backward", speed_delta)
    # Calculate Moving Average Features from Annotation
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "w", "w_ma")
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "h", "h_ma")
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "x", "x_ma")
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "y", "y_ma")
    # Calculate Speed Features
    featured_veh_annotations = _deltaForward(featured_veh_annotations,  "x_ma", "delta_x_forward", speed_delta)
    featured_veh_annotations = _deltaForward(featured_veh_annotations,  "y_ma", "delta_y_forward", speed_delta)
    featured_veh_annotations = _deltaBackward(featured_veh_annotations, "x_ma", "delta_x_backward", speed_delta)
    featured_veh_annotations = _deltaBackward(featured_veh_annotations, "y_ma", "delta_y_backward", speed_delta)
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "delta_x_forward", "delta_x_forward")
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "delta_y_forward", "delta_y_forward")
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "delta_x_backward", "delta_x_backward")
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "delta_y_backward", "delta_y_backward")
    featured_veh_annotations["v_x_forward"] = featured_veh_annotations["delta_x_forward"] / featured_veh_annotations["delta_time_forward"]
    featured_veh_annotations["v_y_forward"] = featured_veh_annotations["delta_y_forward"] / featured_veh_annotations["delta_time_forward"]
    featured_veh_annotations["v_forward"] = np.sqrt(featured_veh_annotations["v_x_forward"]*featured_veh_annotations["v_x_forward"] + featured_veh_annotations["v_y_forward"]*featured_veh_annotations["v_y_forward"])
    featured_veh_annotations["v_x_backward"] = featured_veh_annotations["delta_x_backward"] / featured_veh_annotations["delta_time_backward"]
    featured_veh_annotations["v_y_backward"] = featured_veh_annotations["delta_y_backward"] / featured_veh_annotations["delta_time_backward"]
    featured_veh_annotations["v_backward"] = np.sqrt(featured_veh_annotations["v_x_backward"]*featured_veh_annotations["v_x_backward"] + featured_veh_annotations["v_y_backward"]*featured_veh_annotations["v_y_backward"])
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "v_backward", "v_backward_ma")
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "v_forward", "v_forward_ma")
    featured_veh_annotations["v_estimation_backward"] = -featured_veh_annotations["v_backward_ma"]
    featured_veh_annotations["v_estimation_forward"] = featured_veh_annotations["v_forward_ma"]
    # Calculate Angle Features
        # Estimated Angle Based on Trajectory - Forward
    featured_veh_annotations["angle_estim1_forward"] = np.arctan(featured_veh_annotations["v_y_forward"]/featured_veh_annotations["v_x_forward"]) 
    featured_veh_annotations["angle_estim1_forward"] = boundAngleListPositive(featured_veh_annotations["angle_estim1_forward"], "rad")
    featured_veh_annotations["angle_estim2_forward"] = np.arcsin(featured_veh_annotations["v_y_forward"]/featured_veh_annotations["v_forward"]) 
    featured_veh_annotations["angle_estim2_forward"] = boundAngleListPositive(featured_veh_annotations["angle_estim2_forward"], "rad")
    featured_veh_annotations["angle_estim3_forward"] = np.arccos(featured_veh_annotations["v_x_forward"]/featured_veh_annotations["v_forward"]) 
    featured_veh_annotations["angle_estim3_forward"] = boundAngleListPositive(featured_veh_annotations["angle_estim3_forward"], "rad")
    featured_veh_annotations["angle_estim_final_forward"] = _estimateBestAngle(featured_veh_annotations["angle_estim3_forward"], featured_veh_annotations["angle_estim2_forward"])
    featured_veh_annotations["angle_estimation_forward"] = featured_veh_annotations["angle_estim_final_forward"]
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "angle_estimation_forward", "angle_estimation_forward")
    featured_veh_annotations = _deltaForward(featured_veh_annotations,  "angle_estimation_forward", "angle_vel_estimation_forward", speed_delta)
    featured_veh_annotations["angle_vel_estimation_forward"] = [angle if abs(angle)<cs.ANGLE_VELOCITY_THRESHOLD else cs.ANGLE_VELOCITY_THRESHOLD for angle in featured_veh_annotations["angle_vel_estimation_forward"]]
    featured_veh_annotations["angle_vel_estimation_forward"] = featured_veh_annotations["angle_vel_estimation_forward"]/featured_veh_annotations["delta_time_forward"]  
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "angle_vel_estimation_forward", "angle_vel_estimation_forward")
        # Estimated Angle Based on Trajectory - Backward
    featured_veh_annotations["angle_estim1_backward"] = np.arctan(featured_veh_annotations["v_y_backward"]/featured_veh_annotations["v_x_backward"]) 
    featured_veh_annotations["angle_estim1_backward"] = boundAngleListPositive(featured_veh_annotations["angle_estim1_backward"], "rad")
    featured_veh_annotations["angle_estim2_backward"] = np.arcsin(featured_veh_annotations["v_y_backward"]/featured_veh_annotations["v_backward"]) 
    featured_veh_annotations["angle_estim2_backward"] = boundAngleListPositive(featured_veh_annotations["angle_estim2_backward"], "rad")
    featured_veh_annotations["angle_estim3_backward"] = np.arccos(featured_veh_annotations["v_x_backward"]/featured_veh_annotations["v_backward"]) 
    featured_veh_annotations["angle_estim3_backward"] = boundAngleListPositive(featured_veh_annotations["angle_estim3_backward"], "rad")
    featured_veh_annotations["angle_estim_final_backward"] = _estimateBestAngle(featured_veh_annotations["angle_estim3_backward"], featured_veh_annotations["angle_estim2_backward"])
    featured_veh_annotations["angle_estimation_backward"] = featured_veh_annotations["angle_estim_final_backward"]
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "angle_estimation_backward", "angle_estimation_backward")
    featured_veh_annotations = _deltaForward(featured_veh_annotations,  "angle_estimation_backward", "angle_vel_estimation_backward", speed_delta)
    featured_veh_annotations["angle_vel_estimation_backward"] = [angle if abs(angle)<cs.ANGLE_VELOCITY_THRESHOLD else cs.ANGLE_VELOCITY_THRESHOLD for angle in featured_veh_annotations["angle_vel_estimation_backward"]]
    featured_veh_annotations["angle_vel_estimation_backward"] = featured_veh_annotations["angle_vel_estimation_backward"]/featured_veh_annotations["delta_time_backward"]  
    featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "angle_vel_estimation_backward", "angle_vel_estimation_backward")
        # Observation Angle
    if obb:
        featured_veh_annotations["angle_observation"] = featured_veh_annotations["angle_rad"]
        featured_veh_annotations["angle_observation"] = boundAngleListPositive(featured_veh_annotations["angle_observation"], "rad")
        featured_veh_annotations["angle_observation"] = 2*np.pi-featured_veh_annotations["angle_observation"]
        featured_veh_annotations["angle_observation_DEG"] = featured_veh_annotations["angle_observation"] / (2*np.pi)*360
        featured_veh_annotations["angle_estimation_forward_DEG"] = featured_veh_annotations["angle_estimation_forward"] / (2*np.pi)*360
        featured_veh_annotations["angle_estimation_backward_DEG"] = featured_veh_annotations["angle_estimation_backward"] / (2*np.pi)*360
        featured_veh_annotations["angle_observation_forward_DEG"] = _nextBestGuess_45DEGREE_Step(featured_veh_annotations["angle_estimation_forward_DEG"], featured_veh_annotations["angle_observation_DEG"])
        featured_veh_annotations["angle_observation_backward_DEG"] = _nextBestGuess_45DEGREE_Step(featured_veh_annotations["angle_estimation_backward_DEG"], featured_veh_annotations["angle_observation_DEG"])
        featured_veh_annotations["angle_observation_forward"] = featured_veh_annotations["angle_observation_forward_DEG"] / 360 *(2*np.pi)
        featured_veh_annotations["angle_observation_backward"] = featured_veh_annotations["angle_observation_backward_DEG"] / 360 *(2*np.pi)
        featured_veh_annotations = _takeBestAngle(featured_veh_annotations, "forward")
        featured_veh_annotations = _takeBestAngle(featured_veh_annotations, "backward")       
        featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "angle_observation_forward", "angle_observation_forward")
        featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "angle_observation_backward", "angle_observation_backward")
        featured_veh_annotations = _deltaForward(featured_veh_annotations,  "angle_observation_forward", "angle_vel_observation_forward", speed_delta)
        featured_veh_annotations = _deltaForward(featured_veh_annotations,  "angle_observation_backward", "angle_vel_observation_backward", speed_delta)
        featured_veh_annotations["angle_vel_observation_forward"] = [angle if abs(angle)<cs.ANGLE_VELOCITY_THRESHOLD else cs.ANGLE_VELOCITY_THRESHOLD for angle in featured_veh_annotations["angle_vel_observation_forward"]]
        featured_veh_annotations["angle_vel_observation_backward"] = [angle if abs(angle)<cs.ANGLE_VELOCITY_THRESHOLD else cs.ANGLE_VELOCITY_THRESHOLD for angle in featured_veh_annotations["angle_vel_observation_backward"]]
        featured_veh_annotations["angle_vel_observation_forward"] = featured_veh_annotations["angle_vel_observation_forward"]/featured_veh_annotations["delta_time_forward"]  
        featured_veh_annotations["angle_vel_observation_backward"] = featured_veh_annotations["angle_vel_observation_backward"]/featured_veh_annotations["delta_time_backward"]  
        featured_veh_annotations = _rollingMA_limHorizon(featured_veh_annotations, "angle_vel_observation_forward", "angle_vel_observation_forward")
        featured_veh_annotations["angle_vel_observation_backward"] = -featured_veh_annotations["angle_vel_observation_backward"]
    return featured_veh_annotations

def _determineTrajectoryFusionWeights(df):
    weights_backward = []
    weights_forward = []
    dat = df["measurement_available"].tolist()
    for i in range(0, len(dat)):
        if i<cs.KALMAN_TRANSIENT_PERIOD:
            weights_forward.append(10000)
        else:
            weights_forward.append(1)
        if i>len(dat)-cs.KALMAN_TRANSIENT_PERIOD:
            weights_backward.append(10000)
        else:
            weights_backward.append(1)
    df["weight_forward"] = weights_forward 
    df["weight_backward"] = weights_backward
    return df

def calculateKalmanFilteredTrajectory(veh_trajectory_raw: pd.DataFrame, Q_k: np.array, R_k: np.array, first_frame: int, last_frame: int, video_frames_per_second: int, obb=False):
    """
    This method calculates the Kalman filtered vehicle trajectory.
    
    Parameters
    ----------
    veh_trajectory_raw : pd.DataFrame
        The raw vehicle trajectory A.
    Q_k : np.array
        The Kalmam Q matrix. Whats this used for? Estimate what?
    R_k : np.array
        The Kalmam R matrix. Whats this used for? Estimate what?
    first_frame : int
        The first frame the Kalman filtering should start from.
    last_frame : int
        The last frame the Kalman filtering should go to.
    video_frames_per_second : int
        The video frame rate (frames per second).
    obb : bool
        Whether a OBB vehicle annotation was provided (including angular information). Default: False.
    
    Returns
    -------
    kalman_filtered_trajectory_rts : pd.DataFrame
        The RTS-Kalman filtered vehicle trajectory. (what was RTS again ruch taubel striehl filter or something?)
    """
    kalman_filtered_trajectory_forward_hbb = kalmanFilterTrajectory(veh_trajectory_raw, Q_k, R_k, first_frame, last_frame, video_frames_per_second, obb)
    kalman_filtered_trajectory_backward_hbb = kalmanFilterTrajectory(veh_trajectory_raw, Q_k, R_k, first_frame, last_frame, video_frames_per_second, obb, rev=True)
    kalman_filtered_trajectory_rts = kalman_filtered_trajectory_forward_hbb.copy()
    kalman_filtered_trajectory_rts = kalman_filtered_trajectory_rts.merge(kalman_filtered_trajectory_backward_hbb, on="frame_nr", how="left")
    kalman_filtered_trajectory_rts["time"] = kalman_filtered_trajectory_rts["time_x"]
    kalman_filtered_trajectory_rts["measurement_available"] = kalman_filtered_trajectory_rts["measurement_available_x"]
    kalman_filtered_trajectory_rts = _determineTrajectoryFusionWeights(kalman_filtered_trajectory_rts)
    kalman_filtered_trajectory_rts["weight_sum"] = kalman_filtered_trajectory_rts[["weight_forward", "weight_backward"]].sum(axis=1)
    kalman_filtered_trajectory_rts["weight_forward_final"] = (kalman_filtered_trajectory_rts["weight_sum"]-kalman_filtered_trajectory_rts["weight_forward"])/kalman_filtered_trajectory_rts["weight_sum"]
    kalman_filtered_trajectory_rts["weight_backward_final"] = (kalman_filtered_trajectory_rts["weight_sum"]-kalman_filtered_trajectory_rts["weight_backward"])/kalman_filtered_trajectory_rts["weight_sum"]
    for col in ["x", "y", "state1", "state2", "state3", "state4", "state5"]:
        kalman_filtered_trajectory_rts[col] = (kalman_filtered_trajectory_rts[col+"_x"]*kalman_filtered_trajectory_rts["weight_forward_final"]+kalman_filtered_trajectory_rts[col+"_y"]*kalman_filtered_trajectory_rts["weight_backward_final"])
        kalman_filtered_trajectory_rts[col] = kalman_filtered_trajectory_rts[col].rolling(window=20, center=True, min_periods=1).mean()
    kalman_filtered_trajectory_rts = kalman_filtered_trajectory_rts[["frame_nr", "time", "measurement_available", "x", "y", "state1", "state2", "state3", "state4", "state5"]]
    return kalman_filtered_trajectory_rts

def alignTrajectories(trajectoryA: pd.DataFrame, trajectoryB: pd.DataFrame):
    """
    This method aligns trajectoryA to trajectoryB. If trajectoryA deviates to much from trajectoryB, then trajectoryB values are used.
    
    Parameters
    ----------
    trajectoryA : pd.DataFrame
        The vehicle trajectory A.
    trajectoryB : pd.DataFrame
        The vehicle trajectory B.
        
    Returns
    -------
    trajectoryA_aligned : pd.DataFrame
        The vehicle trajectory A aligned to B.
    """
    trajectoryA_aligned = trajectoryA.copy()
    trajectoryA_aligned = trajectoryA_aligned.merge(trajectoryB, on=["frame_nr", "time", "measurement_available"], how="left")
    trajectoryA_aligned["dist_x"] = trajectoryA_aligned["x_x"] - trajectoryA_aligned["x_y"]
    trajectoryA_aligned["dist_y"] = trajectoryA_aligned["x_x"] - trajectoryA_aligned["x_y"]
    trajectoryA_aligned["dist"] = np.sqrt(trajectoryA_aligned["dist_y"]*trajectoryA_aligned["dist_y"] + trajectoryA_aligned["dist_x"]*trajectoryA_aligned["dist_x"]) 
    trajectoryA_aligned["use_hbb"] = trajectoryA_aligned["dist"]>cs.MAX_ALLOWED_OBB_DEVIATON_M
    for col in ["x", "y", "state1", "state2", "state3", "state4", "state5"]:
        trajectoryA_aligned.loc[  trajectoryA_aligned["use_hbb"], col ] = trajectoryA_aligned[col+"_y"]
        trajectoryA_aligned.loc[ ~trajectoryA_aligned["use_hbb"], col ] = trajectoryA_aligned[col+"_x"]
    trajectoryA_aligned = trajectoryA_aligned[["frame_nr", "time", "measurement_available", "use_hbb", "x", "y", "state1", "state2", "state3", "state4", "state5"]]
    trajectoryA_aligned = _rollingMA_limHorizon(trajectoryA_aligned, "x", "x")
    trajectoryA_aligned = _rollingMA_limHorizon(trajectoryA_aligned, "y", "y")
    return trajectoryA_aligned