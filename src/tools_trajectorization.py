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
import tools_hungarian
import json




# #############################################################################
# METHODS
# #############################################################################

def generateTrajectories(annotations: dict, print_status=False):
    """
    This function matches single frame annotations on a frame-to-frame basis to
    coherent trajectories were possible, using Hungarian matching.

    Parameters
    ----------
    annotations: dict
        The annotations. Each key represents a frame number.
    print_status: bool
        If set to True, console printing takes place to inform about the process of trajectorization.
        Default: False, so no printing takes place.
        
    Returns
    -------
    trajectorized_annotations : dict
        The annotations. Each key represents a frame number. 
        Each annotation contains a new column with a trajectory label.
    """
    trajectorized_annotations = {}
    current_trajectories = {}
    trajectory_counter = 0
    num_frames = list(annotations.keys())[-1]
    for frame in annotations:
        if print_status:
            print(frame, "/", num_frames, ", Trajectories: ", trajectory_counter, len(annotations[frame]), len(current_trajectories))
        current_frame_annotations = annotations[frame]
        # Initial Setup if no trajectories known
        if len(current_trajectories)==0:
            for annotation in current_frame_annotations:
                traj_name = "Trajectory_"+str(trajectory_counter)
                annotation.append(traj_name)
                trajectory_counter += 1
                current_trajectories[traj_name] = {}
                current_trajectories[traj_name]["last_frame"] = frame
                current_trajectories[traj_name]["last_position"] = np.asarray([annotation[1], annotation[2]])
        # Hungarian Matching Approach
        else:
            # Create 2D Distance Matrix
            dist_mat = []
            for annotation in current_frame_annotations:
                dist_vec = []
                for trajectory in current_trajectories:
                    pos_a = current_trajectories[trajectory]["last_position"]
                    pos_b = np.asarray([annotation[1], annotation[2]])
                    distance = np.linalg.norm(pos_a-pos_b)
                    dist_vec.append(distance)
                dist_mat.append(dist_vec)
            dist_mat = np.asarray(dist_mat)
            # Hungarian Matching
            result = None
            try:
                hun = tools_hungarian.Hungarian(dist_mat)
                hun.calculate()
                result = hun.get_results()
            except:
                print("Hungarian Error")
            if result is None:
                continue
            # Update Current Trajectories & Annotation
            trajectory_labels = [trajectory for trajectory in current_trajectories]
            for pair in result:
                annotation_idx = pair[0]
                trajectory_idx = pair[1]
                trajectory_name = trajectory_labels[trajectory_idx]
                annotation = current_frame_annotations[annotation_idx]
                pos_a = current_trajectories[trajectory_name]["last_position"]
                pos_b = np.asarray([annotation[1], annotation[2]])
                distance = np.linalg.norm(pos_a-pos_b)
                if distance < cs.MAX_MATCHING_DISTANCE:
                    current_trajectories[trajectory_name]["last_frame"] = frame
                    current_trajectories[trajectory_name]["last_position"] = np.asarray([annotation[1], annotation[2]])
                    annotation.append(trajectory_name)
            # Register New Trajectories for Unmatched Annotations
            for annotation in current_frame_annotations:
                if len(annotation) == 7: # Label missing
                    traj_name = "Trajectory_"+str(trajectory_counter)
                    annotation.append(traj_name)
                    trajectory_counter += 1
                    current_trajectories[traj_name] = {}
                    current_trajectories[traj_name]["last_frame"] = frame
                    current_trajectories[traj_name]["last_position"] = np.asarray([annotation[1], annotation[2]])
        # Store results in processed
        trajectorized_annotations[frame] = []  
        for annotation in current_frame_annotations:
            trajectorized_annotations[frame].append(annotation)
        # Clean Unmatched Too Long Out
        to_delete_keys = []
        for trajectory in current_trajectories:
            if frame-current_trajectories[trajectory]["last_frame"] > cs.MAX_FRAME_GAP:
                to_delete_keys.append(trajectory)
        to_delete_keys = list(set(to_delete_keys))
        for key in to_delete_keys:
            del current_trajectories[key]
    # Return
    return trajectorized_annotations


def determineUniqueTrajectoryLabels(trajectorized_annotations: dict):
    """
    This function matches single frame annotations on a frame-to-frame basis to
    coherent trajectories were possible, using Hungarian matching.

    Parameters
    ----------
    trajectorized_annotations: dict
        The annotations. Each key represents a frame number.
        Each annotation contains the last column with a trajectory label.
    print_status: bool
        If set to True, console printing takes place to inform about the process of trajectorization.
        Default: False, so no printing takes place.
        
    Returns
    -------
    unique_trajectory_labels : list[str]
        A list of unique trajectory labels.
    """
    unique_trajectory_labels = []
    for frame_nr in trajectorized_annotations:
        for annotation in trajectorized_annotations[frame_nr]:
            unique_trajectory_labels.append(annotation[-1])
    unique_trajectory_labels = list(set(unique_trajectory_labels))
    return unique_trajectory_labels



def generateEmptyTrajectoryLabelVehicleMap(map_file: str, unique_trajectory_labels: list):
    """
    This function generates an empty trajectory label  - vehicle map and stores it into a file for manual editing.
    
    Parameters
    ----------
    map_file: str
        The file where the map shall be stored.
    unique_trajectory_labels : list[str]
        A list of unique trajectory labels.
    """
    file_writer = open(map_file, "w+")
    file_writer.write("{\n")
    for trajectory_label_idx in range(0, len(unique_trajectory_labels)):
        trajectory_label = unique_trajectory_labels[trajectory_label_idx]
        file_writer.write("\t\"")
        file_writer.write(trajectory_label)
        if not trajectory_label_idx == len(unique_trajectory_labels)-1:
            file_writer.write("\": \"UNDEFINED\",\n")
        else:
            file_writer.write("\": \"UNDEFINED\"\n")
    file_writer.write("}\n")
    file_writer.close()

def loadTrajectoryLabelVehicleMap(map_file: str):
    """
    This function loads the trajectory label  - vehicle map from a given file.
    
    Parameters
    ----------
    map_file: str
        The file where the map is stored.
        
    Returns
    -------
    trajectory_vehicle_map: dict
        The trajectory label - vehicle map, where the keys are trajectory labels,
        and the values are vehicle labels.
    """
    file_reader = open(map_file, "r")
    json_string = file_reader.read()
    file_reader.close()
    trajectory_vehicle_map = json.loads(json_string)
    return trajectory_vehicle_map