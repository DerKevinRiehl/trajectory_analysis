"""
Consistent Vehicle Trajectory Extraction From Aerial Recordings Using Oriented Object Detection
-------------------------------------------
Authors:        Kevin Riehl, Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024 - 2025
Submitted to:   Scientific Reports
-------------------------------------------
"""


# #############################################################################
# IMPORTS
# #############################################################################
import numpy as np
import _constants as cs




# #############################################################################
# METHODS
# #############################################################################

def boundAnglePositive(angle, angle_format="rad"):
    """
    This method filters any angle into the positive range (0 - 360°) resp. (0 - 2*PI).

    Parameters
    ----------
    angle : float
        The angle to be bounded.
    angle_format : str
        The format of the angle. Options are "deg" (degree) or "rad" (radians). Default is "deg".
        
    Returns
    -------
    angle_bounded : float
        The bounded angle.
    """
    angle_bounded = angle
    if angle_format=="rad":
        while angle_bounded <0:
            angle_bounded += 2*np.pi
        while angle_bounded > 2*np.pi:
            angle_bounded -= 2*np.pi
    elif angle_format=="deg":
        while angle_bounded <0:
            angle_bounded += 360
        while angle_bounded > 360:
            angle_bounded -= 360
    else:
        raise Exception("Invalid angle_format '"+str(angle_format)+"'. Supported Formats are 'rad' and 'deg'!")
    return angle_bounded

def boundAngleListPositive(lst_angles, angle_format="deg"):
    """
    This method filters any list of angles into the positive range (0 - 360°) resp. (0 - 2*PI).

    Parameters
    ----------
    lst_angle : List[float]
        The angles to be bounded.
    angle_format : str
        The format of the angle. Options are "deg" (degree) or "rad" (radians). Default is "deg".
        
    Returns
    -------
    angle_bounded : List[float]
        The bounded angles.
    """
    lst_angles_bounded = []
    for angle in lst_angles:
        lst_angles_bounded.append(boundAnglePositive(angle, angle_format))
    return lst_angles_bounded


def _estimateBestAngle(lst_angle_cos, lst_angle_sin):
    lst_angle_cos = lst_angle_cos.tolist()
    lst_angle_sin = lst_angle_sin.tolist()
    lst_new = []
    for idx in range(0, len(lst_angle_cos)):
        if lst_angle_sin[idx] < np.pi:
            lst_new.append(np.pi + (np.pi - lst_angle_cos[idx]) )
        else:
            lst_new.append(lst_angle_cos[idx])

    return lst_new

def _nextBestGuess_45DEGREE_Step(lst_angle_estim, lst_angle_real):
    lst_angle_estim = lst_angle_estim.tolist()
    lst_angle_real = lst_angle_real.tolist()
    lst_new = []
    for idx in range(0, len(lst_angle_estim)):
        angle_real = lst_angle_real[idx]
        angle_estim = lst_angle_estim[idx]

        angle_realA = boundAnglePositive(angle_real+90*0, "deg")
        angle_realB = boundAnglePositive(angle_real+90*1, "deg")
        angle_realC = boundAnglePositive(angle_real+90*2, "deg")
        angle_realD = boundAnglePositive(angle_real+90*3, "deg")
        
        angle_diffA = abs(angle_estim - angle_realA)
        angle_diffB = abs(angle_estim - angle_realB)
        angle_diffC = abs(angle_estim - angle_realC)
        angle_diffD = abs(angle_estim - angle_realD)
        
        argmin_idx = np.argmin([angle_diffA, angle_diffB, angle_diffC, angle_diffD])
        lst_angle_candidates = [angle_realA, angle_realB, angle_realC, angle_realD]
        lst_new.append(lst_angle_candidates[argmin_idx])
    return lst_new

def _takeBestAngle(df, label):
    angle_observation = df["angle_observation_"+label].tolist()
    angle_estimation = df["angle_estimation_"+label].tolist()
    angle_new = []
    for i in range(0, len(angle_observation)):
        if angle_estimation[i]==np.nan:
            angle_new.append(np.nan)
        elif abs(angle_observation[i]-angle_estimation[i]) < cs.ANGLE_ESTIMATION_DEVIATION_TOLLERANCE:
            angle_new.append(angle_observation[i])
        else:
            angle_new.append(angle_estimation[i])
    df["angle_observation_"+label] = angle_new
    return df
