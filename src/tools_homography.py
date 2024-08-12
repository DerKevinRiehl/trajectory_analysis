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
import pandas as pd
from shapely.geometry.polygon import Polygon
import _constants as cs


# #############################################################################
# CONSTANTS
# #############################################################################
CIRCLE_DIAMETER = 65.17 # [m]




# #############################################################################
# METHODS
# #############################################################################

def loadHomography(homography_file: str):
    """
    This method loads the homography information for each frame. In this 
    project we employ the large circle in the middle of the circular road as 
    the characteristic pattern.

    Parameters
    ----------
    homography_file : str
        The path to the homography file.
        
    Returns
    -------
    df_circles: pd.DataFrame
        A dataframe, including the columns: "frame_nr", "x", "y", "r"
        This information can be used to transform the pixel coordinate system
        to Cartesian coordinates, knowing the real radius of the circle.
    """
    df_homography = []
    file = open(homography_file, "r")
    line = file.readline()
    while line!="":
        parts = line.replace("["," ").replace("]", " ").replace("\n", "").replace("\t", " ").split(" ")
        while "" in parts:
            parts.remove("")
        frame_nr = parts[0]
        x = parts[1]
        y = parts[2]
        r = parts[3]
        frame_nr = int(frame_nr)
        x = float(x)
        y = float(y)
        r = float(r)
        df_homography.append([frame_nr,x,y,r])
        line = file.readline()
    file.close()
    df_homography= pd.DataFrame(df_homography, columns=["frame_nr", "x", "y", "r"])
    return df_homography
    
def getFrameHomography(df_homography: pd.DataFrame, frame_nr: int):
    """
    This method loads the homography information for each frame. In this 
    project we employ the large circle in the middle of the circular road as 
    the characteristic pattern.

    Parameters
    ----------
    df_homography : pd.DataFrame
        The loaded homography dataframe.
    frame_nr: int
        The frame number that the homograhy should be loaded for.
        
    Returns
    -------
    frame_homography : list[int]
        The frame-specific homography list of three integers: "x", "y", "r"
    """
    df_selection = df_homography[df_homography["frame_nr"]==frame_nr]
    frame_homography = [df_selection["x"].tolist()[0], 
                        df_selection["y"].tolist()[0], 
                        df_selection["r"].tolist()[0]]
    return frame_homography

def transformPointFrom_PIX_2_CARTESIAN(point: list, frame_homography: list):
    """
    Transform point from pixel to Cartesian coordinates.

    Parameters
    ----------
    point : list[float]
        The coordinates of the point: "x", "y" (pixel coordinates)
    frame_homography: list[int]
        The homography list of three integers: "x", "y", "r"
        
    Returns
    -------
    point_new : list[float]
        The coordinates of the point: "x", "y" (Cartesian coordinates)
    """
    scale_factor_meter_per_pixel = CIRCLE_DIAMETER / (2*frame_homography[2])
    point_new = point.copy()
    point_new[0] =  (point[0] - frame_homography[0]) * scale_factor_meter_per_pixel
    point_new[1] =  (point[1] - frame_homography[1]) * scale_factor_meter_per_pixel
    return point_new

def transformPointFrom_CARTESIAN_2_PIX(point: list, frame_homography: list):
    """
    Transform point from Cartesian to pixel coordinates.

    Parameters
    ----------
    point : list[float]
        The coordinates of the point: "x", "y" (Cartesian coordinates)
    frame_homography: list[int]
        The homography list of three integers: "x", "y", "r"
        
    Returns
    -------
    point_new : list[float]
        The coordinates of the point: "x", "y" (pixel coordinates)
    """
    scale_factor_meter_per_pixel = CIRCLE_DIAMETER / (2*frame_homography[2])
    point_new = point.copy()
    point_new[0] = point[0] / scale_factor_meter_per_pixel + frame_homography[0]
    point_new[1] = point[1] / scale_factor_meter_per_pixel + frame_homography[1]
    return point_new

def transformAnnotations_PIX_2_CARTESIAN(frame_annotations: list, frame_homography: list):
    """
    Transform annotations from pixel to Cartesian coordinates.

    Parameters
    ----------
    frame_annotations : list
        The frame-specific annotations (coordinates in pixels). 
    frame_homography: list[int]
        The homography list of three integers: "x", "y", "r".
        
    Returns
    -------
    transformed_annotations : list
        The coordinates of the point: "x", "y" (Cartesian coordinates)
    """
    scale_factor_meter_per_pixel = CIRCLE_DIAMETER / (2*frame_homography[2])
    transformed_annotations = []
    for annotation in frame_annotations:
        new_annotation = []
        new_annotation.append(annotation[cs.INDEX_ANNOTATION["ANN_TYPE"]])
        point_new = transformPointFrom_PIX_2_CARTESIAN([annotation[cs.INDEX_ANNOTATION["POS_X"]], 
                                                        annotation[cs.INDEX_ANNOTATION["POS_Y"]]], 
                                                       frame_homography)
        x_rel = point_new[0]
        y_rel = point_new[1]
        new_annotation.append(x_rel)
        new_annotation.append(y_rel)
        w_rel = annotation[cs.INDEX_ANNOTATION["WIDTH"]] * scale_factor_meter_per_pixel
        h_rel = annotation[cs.INDEX_ANNOTATION["HEIGHT"]] * scale_factor_meter_per_pixel
        new_annotation.append(w_rel)
        new_annotation.append(h_rel)
        for idx in range(5, len(annotation)):
            new_annotation.append(annotation[idx])
        transformed_annotations.append(new_annotation)
    return transformed_annotations

def transformAnnotations_CARTESIAN_2_PIX(frame_annotations: list, frame_homography: list):
    """
    Transform annotations from Cartesian to pixel coordinates.

    Parameters
    ----------
    frame_annotations : list
        The frame-specific annotations (Cartesian coordinates). 
    frame_homography: list[int]
        The homography list of three integers: "x", "y", "r".
        
    Returns
    -------
    transformed_annotations : list
        The coordinates of the point: "x", "y" (Coordinates in pixels)
    """
    scale_factor_meter_per_pixel = CIRCLE_DIAMETER / (2*frame_homography[2])
    transformed_annotations = []
    for annotation in frame_annotations:
        new_annotation = []
        new_annotation.append(annotation[cs.INDEX_ANNOTATION["ANN_TYPE"]])
        point_new = transformPointFrom_CARTESIAN_2_PIX([annotation[cs.INDEX_ANNOTATION["POS_X"]], 
                                                        annotation[cs.INDEX_ANNOTATION["POS_Y"]]], 
                                                       frame_homography)
        x_rel = point_new[0]
        y_rel = point_new[1]
        new_annotation.append(x_rel)
        new_annotation.append(y_rel)
        w_rel = annotation[cs.INDEX_ANNOTATION["WIDTH"]] / scale_factor_meter_per_pixel
        h_rel = annotation[cs.INDEX_ANNOTATION["HEIGHT"]] / scale_factor_meter_per_pixel
        new_annotation.append(w_rel)
        new_annotation.append(h_rel)
        for idx in range(5, len(annotation)):
            new_annotation.append(annotation[idx])
        transformed_annotations.append(new_annotation)
    return transformed_annotations

def _transform_roi_shape(shape, transform_func, frame_homography):
    x_coords = shape.exterior.coords.xy[0]
    y_coords = shape.exterior.coords.xy[1]
    points = []
    for idx in range(0, len(x_coords)):
        points.append([x_coords[idx], y_coords[idx]])    
    for idx in range(0, len(x_coords)):
        points[idx] = transform_func(points[idx], frame_homography)
    new_polygon = Polygon([*points])
    return new_polygon

def getTransformedRegionOfInterest(region_of_interest: dict, df_homography: pd.DataFrame, frame_nr: int):
    """
    This method transforms the region of interest, so that it fits to a specific frame.
    The region of interest is defined for the first frame. Using the homography, it is
    updated by this function to account for perspective movements.

    Parameters
    ----------
    region_of_interest : dict
        The region of interest.
    df_homography : pd.DataFrame
        The loaded homography dataframe.
    frame_nr: int
        The frame number that the homograhy should be loaded for.
        
    Returns
    -------
    new_region_of_interest : dict
        The updated version of the region_of_interest for a specific frame.
    """
    new_region_of_interest = region_of_interest.copy()
    region_area_1 = new_region_of_interest["INCLUDING"]
    region_area_2 = new_region_of_interest["EXCLUDING"]
    first_homography = getFrameHomography(df_homography, 0)
    second_homography = getFrameHomography(df_homography, frame_nr)
    # Step 1: convert to Cartesian coordinates using homography of frame 1
    region_area_1 = _transform_roi_shape(region_area_1, transformPointFrom_PIX_2_CARTESIAN, first_homography)
    region_area_2 = _transform_roi_shape(region_area_2, transformPointFrom_PIX_2_CARTESIAN, first_homography)
    # Step 2: convert back to Pixel coordinates using homography of specific frame
    region_area_1 = _transform_roi_shape(region_area_1, transformPointFrom_CARTESIAN_2_PIX, second_homography)
    region_area_2 = _transform_roi_shape(region_area_2, transformPointFrom_CARTESIAN_2_PIX, second_homography)
    # Return results
    new_region_of_interest["INCLUDING"] = region_area_1
    new_region_of_interest["EXCLUDING"] = region_area_2
    return new_region_of_interest
    