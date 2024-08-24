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
import numpy as np
import _constants as cs
from tools_filtering_angle import boundAnglePositive
from scipy.optimize import minimize
import pandas as pd




# #############################################################################
# METHODS
# #############################################################################


def getInitialEstimate(veh_annotations, obb=False, rev=False):
    # select relevant columns
    if not obb:
        if not rev:
            cols  = ["x", "y", "angle_estimation_forward",  "v_estimation_forward",  "angle_vel_estimation_forward"]
        else:
            cols  = ["x", "y", "angle_estimation_backward", "v_estimation_backward", "angle_vel_estimation_backward"]
        colsE = ["x_ma", "y_ma", "angle_estimation_forward", "v_estimation_forward", "angle_vel_estimation_forward", "x", "y", "angle_estimation_backward", "v_estimation_backward", "angle_vel_estimation_backward"]
    else:    
        if not rev:
            cols  = ["x", "y", "angle_observation_forward",  "v_estimation_forward",  "angle_vel_observation_forward"]
        else:
            cols  = ["x", "y", "angle_observation_backward", "v_estimation_backward", "angle_vel_observation_backward"]
        colsE = ["x_ma", "y_ma", "angle_observation_forward", "v_estimation_forward", "angle_vel_observation_forward", "x", "y", "angle_observation_backward", "v_estimation_backward", "angle_vel_observation_backward"]
    # initial estimate for state
    if not rev:
        x_0 = [np.nanmean(veh_annotations[ cols[0] ].tolist()[cs.SPEED_ESTIMATION_TIME_HORIZON:cs.SPEED_ESTIMATION_TIME_HORIZON+cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]), 
               np.nanmean(veh_annotations[ cols[1] ].tolist()[cs.SPEED_ESTIMATION_TIME_HORIZON:cs.SPEED_ESTIMATION_TIME_HORIZON+cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]),
               np.nanmean(veh_annotations[ cols[2] ].tolist()[cs.SPEED_ESTIMATION_TIME_HORIZON:cs.SPEED_ESTIMATION_TIME_HORIZON+cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]), 
               np.nanmean(veh_annotations[ cols[3] ].tolist()[cs.SPEED_ESTIMATION_TIME_HORIZON:cs.SPEED_ESTIMATION_TIME_HORIZON+cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]),
               np.nanmean(veh_annotations[ cols[4] ].tolist()[cs.SPEED_ESTIMATION_TIME_HORIZON:cs.SPEED_ESTIMATION_TIME_HORIZON+cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH]), 
               ]
    else:
        x_0 = [np.nanmean(veh_annotations[ cols[0] ].tolist()[-cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-cs.SPEED_ESTIMATION_TIME_HORIZON:-cs.SPEED_ESTIMATION_TIME_HORIZON]), 
               np.nanmean(veh_annotations[ cols[1] ].tolist()[-cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-cs.SPEED_ESTIMATION_TIME_HORIZON:-cs.SPEED_ESTIMATION_TIME_HORIZON]),
               np.nanmean(veh_annotations[ cols[2] ].tolist()[-cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-cs.SPEED_ESTIMATION_TIME_HORIZON:-cs.SPEED_ESTIMATION_TIME_HORIZON]), 
               np.nanmean(veh_annotations[ cols[3] ].tolist()[-cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-cs.SPEED_ESTIMATION_TIME_HORIZON:-cs.SPEED_ESTIMATION_TIME_HORIZON]),
               np.nanmean(veh_annotations[ cols[4] ].tolist()[-cs.KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH-cs.SPEED_ESTIMATION_TIME_HORIZON:-cs.SPEED_ESTIMATION_TIME_HORIZON]), 
               ]
    # initial estimate for state error
    state_error = veh_annotations[[*colsE]]
    state_error["x_err"]  = veh_annotations[ colsE[0] ] - veh_annotations[ colsE[5] ]
    state_error["y_err"]  = veh_annotations[ colsE[1] ] - veh_annotations[ colsE[6] ]
    state_error["a_err"]  = veh_annotations[ colsE[2] ] - veh_annotations[ colsE[7] ]
    state_error["v_err"]  = veh_annotations[ colsE[3] ] - veh_annotations[ colsE[8] ]
    state_error["av_err"] = veh_annotations[ colsE[4] ] - veh_annotations[ colsE[9] ]
    state_error = np.asarray(state_error[["x_err", "y_err", "a_err", "v_err", "av_err"]])
    state_error_valid = np.sum(np.isnan(state_error), axis=1)
    P_0 = np.cov(state_error[state_error_valid==0].transpose())
    return x_0, P_0

def generateKalmanTrajectoryDataFromVehicleAnnotations(veh_annotations, obb=False, rev=False):
    # New Description For Kalman Filtering
    data = veh_annotations.copy()
    data["y_1_m"] = data["x"]
    data["y_2_m"] = data["y"]
    if not obb:
        if not rev:
            data["y_3_m"] = data["angle_estimation_forward"]
            data["y_4_m"] = data["v_estimation_forward"]
            data["y_5_m"] = data["angle_vel_estimation_forward"]
        else:
            data["y_3_m"] = data["angle_estimation_backward"]
            data["y_4_m"] = data["v_estimation_backward"]
            data["y_5_m"] = data["angle_vel_estimation_backward"]
    else:
        if not rev:
            data["y_3_m"] = data["angle_observation_forward"]
            data["y_4_m"] = data["v_estimation_forward"]
            data["y_5_m"] = data["angle_vel_observation_forward"]
        else:
            data["y_3_m"] = data["angle_observation_backward"]
            data["y_4_m"] = data["v_estimation_backward"]
            data["y_5_m"] = data["angle_vel_observation_backward"]
    data = data[["frame_nr", "time", "y_1_m", "y_2_m", "y_3_m", "y_4_m", "y_5_m"]]
    return data


# Linearized Matrices 
# system state matrix:    x+1 = f(x,u)    ->      A = df/dx
# output state matrix:    y+1 = h(x,u)    ->      C = dh/dx
# state = [x, y, angle,  v, angle_vel]
# units = [m, m, rad,  m/s, rad/s]
def f_func(x, u, video_frames_per_second):
    delta_time = 1/video_frames_per_second
    x_new = [x[0] + np.cos(x[2])*x[3]*delta_time,
             x[1] - np.sin(x[2])*x[3]*delta_time,
             boundAnglePositive(x[2] + x[4]*delta_time, "rad"),
             x[3],
             x[4]]
    return np.asarray(x_new)

def h_func(x, u):
    return np.asarray([x[0], x[1], x[2], x[3], x[4]])

def getLinearizedMatrix_A(last_x_c, delta_time):
    A = np.asarray([
        [1,0,-np.sin(last_x_c[2])*last_x_c[3]*delta_time,+np.cos(last_x_c[2])*delta_time, 0], 
        [0,1,-np.cos(last_x_c[2])*last_x_c[3]*delta_time,-np.sin(last_x_c[2])*delta_time, 0],
        [0,0,1,0,delta_time],
        [0,0,0,1,0],
        [0,0,0,0,1]
        ])
    return A

C = np.asarray([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]) # system output matrix
I = np.eye(5)
sel_columns = ["y_1_m", "y_2_m", "y_3_m", "y_4_m", "y_5_m"]

def determineNextAvailableFrame(df, a,b,c):
    next_frame = -1
    for frame_nr in range(a, b, c):
        if len(df[df["frame_nr"]==frame_nr])==1:
            next_frame = frame_nr
            break
    return next_frame

def predictEvaluateGuess(state_start, state_target, frame_start, frame_end, frame_steps, vel, angle_vel, crit, video_frames_per_second):
    last_x_p = state_start.copy()
    last_x_p[3] = vel
    last_x_p[4] = angle_vel
    distance_travelled = []
    for frame_nr in range(frame_start, frame_end, frame_steps):
        next_x_p = f_func( last_x_p, [], video_frames_per_second )
        distance_travelled.append(np.linalg.norm(np.asarray([last_x_p[0], last_x_p[1]]) - np.asarray([state_target[0], state_target[1]])))    
        last_x_p = next_x_p
    distance_travelled.append(np.linalg.norm(np.asarray([last_x_p[0], last_x_p[1]]) - np.asarray([state_target[0], state_target[1]])))    
    weights = np.arange(len(distance_travelled))*np.arange(len(distance_travelled))
    weights = weights/np.sum(weights)
    pos_actual = np.asarray([last_x_p[0], last_x_p[1]])
    pos_target = np.asarray([state_target[0],state_target[1]])
    angle_diff = abs(last_x_p[2]-state_target[2])
    if crit:
        return np.linalg.norm(pos_actual-pos_target)
    else:
        return 10*np.linalg.norm(pos_actual-pos_target) + angle_diff

def to_optimize_function(x, state_start, state_target, frame_start, frame_end, frame_steps, crit, video_frames_per_second):
    return predictEvaluateGuess(state_start, state_target, frame_start, frame_end, frame_steps, x[0], x[1], crit, video_frames_per_second)

def kalmanFilterTrajectory(veh_trajectory_raw, Q_k, R_k, first_frame, last_frame, video_frames_per_second, obb=False, rev=False):
    # Determine Initial Estimates
    x_0, P_0 = getInitialEstimate(veh_trajectory_raw, obb, rev)
    # Prepare Observation Data from Vehicle Trajectory
    kalman_data = generateKalmanTrajectoryDataFromVehicleAnnotations(veh_trajectory_raw, obb, rev)
    kalman_data = kalman_data.dropna()
    last_x_p = x_0.copy()
    last_x_c = x_0.copy()
    last_P_p = P_0.copy()
    last_P_c = P_0.copy()
    last_u = None
    state_kalman = []
    last_frame_had_measurement=False
    a = first_frame
    b = last_frame
    c = 1
    if rev:
        a = last_frame
        b = first_frame
        c = -1
    skip_counter = 0
    # For each Frame conduct Kalman Filter Step
    for frame_nr in range(a, b, c):   
        # Skip Kalman Iterations if very large gap was filled by inference approach
        if skip_counter > 0:
            skip_counter -= 1
            continue
        time = frame_nr*(1/video_frames_per_second)
        # Determine measurement availability
        measurement_available = len(kalman_data[kalman_data["frame_nr"]==frame_nr])==1
        if not rev:
            next_available_frame = determineNextAvailableFrame(kalman_data, frame_nr+1, b, c)
        else:
            next_available_frame = determineNextAvailableFrame(kalman_data, frame_nr-1, b, c)
        # Assess whether observation gap to large, whether too apply inference approach
        if measurement_available:
            if next_available_frame!=-1 and abs(next_available_frame - frame_nr) >= cs.SKIP_KALMAN_FILTERING_MAX_GAP:
                # skip following frames
                if not rev:
                    skip_counter = next_available_frame - frame_nr - 1
                else:
                    skip_counter = frame_nr - next_available_frame - 1
                # Determine Speed and Angular Velocity based on optimization, so that arc hits next observation best
                target = np.asarray(kalman_data[kalman_data["frame_nr"]==next_available_frame][[*sel_columns]].iloc[0])                 
                x_initial_guess = [last_x_p[3], last_x_p[4]]
                res = minimize(to_optimize_function, x_initial_guess, method="nelder-mead",
                               args=(last_x_p, target, frame_nr, next_available_frame, c, True, video_frames_per_second),
                               # (x, state_start, state_target, frame_start, frame_end, frame_steps, crit)
                               options={'xatol': 1e-8, 'disp': False})
                last_x_p[3] = res.x[0]
                last_x_p[4] = res.x[1]
                x_initial_guess = [last_x_p[3], last_x_p[4]]
                res = minimize(to_optimize_function, x_initial_guess, method="nelder-mead",
                               args=(last_x_p, target, frame_nr, next_available_frame, c, False, video_frames_per_second),
                               # (x, state_start, state_target, frame_start, frame_end, frame_steps, crit)
                               options={'xatol': 1e-8, 'disp': False})
                last_x_p[3] = res.x[0]
                last_x_p[4] = res.x[1]                
                # Fill Gaps of Series with Inference Approach
                for frame_nr2 in range(frame_nr, next_available_frame, c):
                    time = frame_nr2*(1/video_frames_per_second)
                    # prediction step
                    u_measured = []
                    last_u = u_measured
                    delta_time = 1/video_frames_per_second
                    A = getLinearizedMatrix_A(last_x_p, delta_time)
                    next_x_p = f_func( last_x_p, u_measured, video_frames_per_second)
                    next_P_p = ( A @ last_P_p @ A.T ) + Q_k
                    state_kalman.append([frame_nr2, time, 2, next_x_p[0], next_x_p[1],   next_x_p[0],next_x_p[1],next_x_p[2],next_x_p[3],next_x_p[4]   ])
                    last_x_p = next_x_p
                    last_P_p = next_P_p
                continue 
        # Prediction Step
        if measurement_available:
            u_measured = [] #np.asarray(kalman_data[kalman_data["frame_nr"]==frame_nr][["u_1_m", "u_2_m"]].iloc[0]) 
            last_u = u_measured
        else:
            u_measured = last_u
        delta_time = 1/video_frames_per_second
        if last_frame_had_measurement: 
            A = getLinearizedMatrix_A(last_x_c, delta_time)
            next_x_p = f_func( last_x_c, u_measured, video_frames_per_second)
            next_P_p = ( A @ last_P_c @ A.T ) + Q_k
        else:
            A = getLinearizedMatrix_A(last_x_p, delta_time)
            next_x_p = f_func( last_x_p, u_measured, video_frames_per_second)
            next_P_p = ( A @ last_P_p @ A.T ) + Q_k
        # Correction Step
        if measurement_available:
            y_measured = np.asarray(kalman_data[kalman_data["frame_nr"]==frame_nr][[*sel_columns]].iloc[0]) 
            K_c = last_P_p @ C.T @ np.linalg.inv( C @ last_P_p @ C.T + R_k )
            next_x_c = last_x_p + K_c @ ( y_measured - h_func( last_x_p, u_measured ) )
            next_P_c = (I - K_c @ C) @ last_P_c
        # Update Data & Variables
        if measurement_available:
            state_kalman.append([frame_nr, time, measurement_available, next_x_c[0], next_x_c[1],   next_x_c[0],next_x_c[1],next_x_c[2],next_x_c[3],next_x_c[4]   ])
        else:
            state_kalman.append([frame_nr, time, measurement_available, next_x_p[0], next_x_p[1],   next_x_p[0],next_x_p[1],next_x_p[2],next_x_p[3],next_x_p[4]   ])
        last_x_p = next_x_p
        last_frame_had_measurement = False
        if measurement_available:
            last_x_c = next_x_c
            last_P_c = next_P_c
            last_frame_had_measurement = True
        last_P_p = next_P_p
    state_kalman = np.asarray(state_kalman)
    kalman_filtered_trajectory = pd.DataFrame(state_kalman, columns=["frame_nr", "time", "measurement_available", "x", "y", "state1", "state2", "state3", "state4", "state5"])
    return kalman_filtered_trajectory