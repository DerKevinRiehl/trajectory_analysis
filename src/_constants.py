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


# #############################################################################
# CONSTANTS
# #############################################################################




######################################
# PATHS
videos = ["DJI_0933.MOV", "DJI_0934.MOV", "DJI_0939.MOV", "DJI_0940.MOV", "DJI_0943.MOV", "DJI_0944.MOV"]
video_path                    = "C:/VIDEO_ETH/"
inference_annotations_path    = "../data/1_annotations/"
homography_path               = "../data/1_homography/"
frame_processed_path          = "../data/2_frame_processed/"
trajectorized_unlabelled_path = "../data/3_A_trajectorized_unlabelled/"
trajectorized_labelled_path   = "../data/3_B_trajectorized_labelled/"
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
    }
}

MIN_VEHICLE_WIDTH    = 1.8 # [m]
MAX_VEHICLE_WIDTH    = 8.0 # [m]
MIN_VEHICLE_HEIGHT   = 0.4 # [m]
MAX_VEHICLE_HEIGHT   = 8.0 # [m]
CLUSTER_MIN_DISTANCE = 3.0 # [m]





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
}