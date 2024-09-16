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
from shapely.geometry.polygon import Polygon
import numpy as np




# #############################################################################
# CONSTANTS
# #############################################################################




######################################
# PATHS
videos = ["DJI_0933.MOV", "DJI_0934.MOV", "DJI_0939.MOV", "DJI_0940.MOV", "DJI_0943.MOV", "DJI_0944.MOV"]
video_path                    = "R:/Riehl/VehicleTrajectoryProject/1_DAT_VIDEO/"
inference_annotations_path    = "../data/1_annotations/"
homography_path               = "../data/1_homography/"
frame_processed_path          = "../data/2_frame_processed/"
trajectorized_unlabelled_path = "../data/3_A_trajectorized_unlabelled/"
trajectorized_labelled_path   = "../data/3_B_trajectorized_vehiclized/"
kalman_filtered_path          = "../data/4_kalman_filtered/"





######################################
# INDICES

# ANNOTATIONS
INDEX_ANNOTATION = {
    "ANN_TYPE": 0,
    "POS_X": 1,
    "POS_Y": 2,
    "WIDTH": 3,
    "HEIGHT": 4,
    "ANGLE": 5,
    "CONFIDENCE": 6,
    "LABEL": 7,
    "TRAJECTORY": 8,
}






######################################
# SINGLE FRAME FILTERING

# ROI = region of interest in first frame, with homography it will be transformed for 
# each frame to account for movement changes
REGION_OF_INTEREST = { 
    "DJI_0933.MOV": {
        "INCLUDING": Polygon([
            (330,  1500),    
            (550,  2050),
            (2700, 2050),
            (3370, 1060),
            (2850, 260),
            (2400, 40),
            (1860, 100),
        ]),
        "EXCLUDING": Polygon([
            (520,  1550), 
            (620,  1750),
            (1510, 1805),
            (1130, 990),
            (560, 1550),
        ])
    },
    "DJI_0934.MOV": {
        "INCLUDING": Polygon([
            (330,  1500),    
            (550,  2050),
            (2700, 2050),
            (3370, 1060),
            (2850, 260),
            (2400, 40),
            (1860, 100),
        ]),
        "EXCLUDING": Polygon([
            (520,  1550), 
            (620,  1750),
            (1510, 1805),
            (1130, 990),
            (560, 1550),
        ])
    },
    "DJI_0939.MOV": {
        "INCLUDING": Polygon([
            (215,  1650),    
            (550,  2030),
            (2610, 1970),
            (3150, 1050),
            (2850, 260),
            (2400, 40),
            (1600, 100),
        ]),
        "EXCLUDING": Polygon([
            (490,  1650), 
            (600,  1850),
            (1350, 1800),
            (1050, 1000),
            (490, 1650),
        ])
    },
    "DJI_0940.MOV": {
        "INCLUDING": Polygon([
            (215,  1650),    
            (550,  2030),
            (2610, 1990),
            (3150, 1050),
            (2850, 260),
            (2400, 40),
            (1600, 100),
        ]),
        "EXCLUDING": Polygon([
            (490,  1650), 
            (600,  1850),
            (1350, 1800),
            (1050, 1000),
            (490, 1650),
        ])
    },
    "DJI_0943.MOV": {
        "INCLUDING": Polygon([
            (390,  1600),    
            (550,  2050),
            (2650, 1950),
            (3340, 1085),
            (2890, 190),
            (2400, 40),
            (1750, 15),
        ]),
        "EXCLUDING": Polygon([
            (645,  1730), 
            (740,  1820),
            (1510, 1760),
            (1140, 1015),
            (640, 1610),
        ])
    },
    "DJI_0944.MOV": {
        "INCLUDING": Polygon([
            (360,  1580),    
            (550,  2050),
            (2630, 1910),
            (3340, 1085),
            (2880, 145),
            (2400, 40),
            (1730, 10),
        ]),
        "EXCLUDING": Polygon([
            (630,  1735), 
            (740,  1820),
            (1510, 1760),
            (1140, 1015),
            (640, 1610),
        ])
    },
}

MIN_VEHICLE_WIDTH    = 1.8 # [m]
MAX_VEHICLE_WIDTH    = 8.0 # [m]
MIN_VEHICLE_HEIGHT   = 0.4 # [m]
MAX_VEHICLE_HEIGHT   = 8.0 # [m]
CLUSTER_MIN_DISTANCE = 3.0 # [m]




######################################
# TRAJECTORY BUILDING
MAX_MATCHING_DISTANCE = 0.5 # [m]
MAX_FRAME_GAP         = 5   # [frames]



######################################
# KALMAN FILTERING
VEHICLE_DIMENSION_MOVING_AVERAGE_WINDOW_LENGTH = 10
VEHICLE_DYNAMICS_MOVING_AVERAGE_WINDOW_LENGTH = 10
KALMAN_INITIAL_ESTIMATION_WINDOW_LENGTH = 30
KALMAN_TRANSIENT_PERIOD = 60
ANGLE_VELOCITY_THRESHOLD = 10/360*2*np.pi
SPEED_ESTIMATION_TIME_HORIZON = 15
SKIP_KALMAN_FILTERING_MAX_GAP = 25
ANGLE_ESTIMATION_DEVIATION_TOLLERANCE = 5/360*2*np.pi
MAX_ALLOWED_OBB_DEVIATON_M = 0.5



######################################
# TRAJECTORY PROCESSING
PROCESSING_INTEGRATION_CORRECTION_THRESHOLD = 0.00001
PROCESSING_INTEGRATION_CORRECTION_REPETITIONS = 50



######################################
# VISUALIZATION

default_drawing_settings = {
    "homography": {
        "draw_line": True,
        "draw_fill": False,
        "draw_alpha": 0.5,
        "line_width": 5,
        "line_style": "--",
        "line_color": "cyan",
        "fill_color": "blue",
        "fill_hatch": None,
    },
    "region_of_interest": {
        "draw_line": True,
        "draw_fill": True,
        "draw_alpha": 0.1,
        "line_width": 5,
        "line_style": "--",
        "line_color": "cyan",
        "fill_color_in": "green",
        "fill_color_ex": "red",
        "fill_hatch": None,
    },
    "vehicle_annotations": {
        "draw_alpha": 0.5,
        "line_width": 1,
        "line_color": "red",
    },
    "labelled_vehicle_annnotations": {
        "draw_alpha": 0.5,
        "circle_radius": 100,
        "line_width": 5,
        "line_color": "white",
        "line_style": "--",
        "font_color": "white",
        "font_size": 20,
    },
}