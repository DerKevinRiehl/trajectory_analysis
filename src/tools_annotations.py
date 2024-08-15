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




# #############################################################################
# METHODS
# #############################################################################
def loadAnnotations(annotation_file: str):
    """
    This method loads all annotations from a given file.

    Parameters
    ----------
    annotation_file : str
        The path to the annotation file.
        
    Returns
    -------
    annotations: dict
        The annotations. Each key represents a frame number. Each value is a list
        of single annotations. Each single annotation is a list of elements,
        including: [annotation_type, position_x, position_y, width, height, angle, confidence].
        The annotation_type depends on the DOTA dataset classification scheme.
        The position, width and height are coordinate-system specific, e.g.
        pixel coordinates, Cartesian coordinates, or lane coordinates.
        The angle is in radians. The confidence is in %, and output of the neural
        network used for annotation.
    """
    if annotation_file.endswith(".zip"):
        df = pd.read_csv(annotation_file, compression="zip", sep="\t", header=None)
    else:
        df = pd.read_csv(annotation_file, sep="\t", header=None)
    if df.iloc[:,df.shape[1]-1].dropna().shape[0]==0:
        df = df.iloc[:,:-1]
    columns = df.shape[1]
    annotations = {}
    for idx, row in df.iterrows():
        frame_no = int(row[0])
        if frame_no not in annotations:
            annotations[frame_no] = []
        values = []
        for x in range(1, columns):
            values.append(row[x])
        annotations[frame_no].append(values)
    return annotations

def saveAnnotations(annotation_file, annotations):
    """
    This method saves all annotations to a given file.

    Parameters
    ----------
    annotation_file : str
        The path to the annotation file.
    annotations: dict
        The annotations to be stored.
    """
    fWriter = open(annotation_file, "w+")
    for frame_nr in annotations:
        frame_annotations = annotations[frame_nr]
        for annotation in frame_annotations:
            fWriter.write(str(frame_nr))
            fWriter.write("\t")
            for valIdx in range(0, len(annotation)):
                val = annotation[valIdx]
                fWriter.write(str(val))
                if not valIdx==len(annotation)-1:
                    fWriter.write("\t")
            fWriter.write("\n")
    fWriter.close()
