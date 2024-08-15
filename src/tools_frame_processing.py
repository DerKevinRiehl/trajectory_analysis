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
import math
import numpy as np
from shapely.geometry.polygon import Point
from tools_homography import transformAnnotations_PIX_2_CARTESIAN
import _constants as cs




# #############################################################################
# METHODS
# #############################################################################
def _filterAnnotationsInRegionOfInterest(frame_annotations, region_of_interest):
    passed_annotations = []
    for ann in frame_annotations:
        filtered_out = False
        point = Point(ann[cs.INDEX_ANNOTATION["POS_X"]], ann[cs.INDEX_ANNOTATION["POS_Y"]])
        if region_of_interest["INCLUDING"] is not None:
            if not region_of_interest["INCLUDING"].contains(point):
                filtered_out = True
        if region_of_interest["EXCLUDING"] is not None:
            if region_of_interest["EXCLUDING"].contains(point):
                filtered_out = True
        if not filtered_out:
            passed_annotations.append(ann)
    return passed_annotations

def _filterAnnotationsBySize(frame_annotations):
    passed_annotations = []
    for ann in frame_annotations:
        filtered_out = False
        if ann[cs.INDEX_ANNOTATION["WIDTH"]] > cs.MAX_VEHICLE_WIDTH:
            filtered_out = True
        if ann[cs.INDEX_ANNOTATION["WIDTH"]] < cs.MIN_VEHICLE_WIDTH:
            filtered_out = True
        if ann[cs.INDEX_ANNOTATION["HEIGHT"]] > cs.MAX_VEHICLE_HEIGHT:
            filtered_out = True
        if ann[cs.INDEX_ANNOTATION["HEIGHT"]] < cs.MIN_VEHICLE_HEIGHT:
            filtered_out = True
        if not filtered_out:
            passed_annotations.append(ann)
    return passed_annotations

def _clusterizeAnnotations(frame_annotations):
    # Cluster Annotations
        # Calculate Annotation Distances
    distances = []
    for ann1 in frame_annotations:
        lst_dist = []
        for ann2 in frame_annotations:
            if not ann1 is ann2:
                lst_dist.append(math.hypot(ann2[cs.INDEX_ANNOTATION["POS_X"]]-ann1[cs.INDEX_ANNOTATION["POS_X"]], 
                                           ann2[cs.INDEX_ANNOTATION["POS_Y"]]-ann1[cs.INDEX_ANNOTATION["POS_Y"]]))
            else:
                lst_dist.append(-1)
        distances.append(lst_dist)
    distances = np.asarray(distances)
    distances[distances>cs.CLUSTER_MIN_DISTANCE] = -1
        # Identify Clusters
    matched_clusters = []
    for it1 in range(0, distances.shape[0]):
        cluster_partners = [it1]
        for it2 in range(0, it1):
            if distances[it1][it2] != -1:
                cluster_partners.append(it2)
        matched_clusters.append(cluster_partners)
    matched_clusters = sorted(matched_clusters, key=len)
    matched_clusters.reverse()
        # Determine Most Confident Annotation per Cluster
    matched_annotations = []
    used_idx = []
    for clust in matched_clusters:
        cluster_annotation = []
        for idx in clust:
            if idx not in used_idx:
                cluster_annotation.append(frame_annotations[idx])
                used_idx.append(idx)
        if len(cluster_annotation) > 0:
            cluster_annotation = np.asarray(cluster_annotation)
            most_confident_idx = np.argmax(cluster_annotation[:,-1])
            matched_annotations.append(cluster_annotation[most_confident_idx].tolist())
    return matched_annotations

def processFrameAnnotations(frame_annotations: list, region_of_interest: dict, frame_homography: list):
    """
    This method single-frame-processes the frame-specific annotations. This includes steps:
        (1) Filter annotations based on Region-Of-Interest
        (2) Transform from Pixel to Cartesian coordinate system (using homography)
        (3) Filter annotations based on Size (dimensions of annotations)
        (4) Cluster annotations and select mostconfident annotation

    Parameters
    ----------
    frame_annotations : list
        The frame-specific annotations.
    region_of_interest : dict
        The region of interest.
    frame_homography: list[int]
        The homography list of three integers: "x", "y", "r"
        
    Returns
    -------
    processed_annotations_4 : list
        The single-frame-processed annotations.
    """
    # Step 0: Raw Annotations
    # Step 1: Filtering by ROI
    processed_frame_annotations_1 = _filterAnnotationsInRegionOfInterest(frame_annotations, region_of_interest)
    # Step 2: Convert from PIXEL to 2D_METER coordinates
    transformed_annotations_2 = transformAnnotations_PIX_2_CARTESIAN(processed_frame_annotations_1, frame_homography)
    # Step 3: Filtering by Size
    processed_annotations_3 = _filterAnnotationsBySize(transformed_annotations_2)
    # Step 4: Clustering And Take Most Confident
    processed_annotations_4 = _clusterizeAnnotations(processed_annotations_3)
    return processed_annotations_4