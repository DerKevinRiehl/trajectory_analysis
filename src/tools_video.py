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
import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as tr
# import matplotlib
import numpy as np
from tools_homography import getFrameHomography, getTransformedRegionOfInterest
import gc



# #############################################################################
# METHODS
# #############################################################################

def getNumberOfFramesFromVideo(video_file_path: str):
    """
    This method determines the number of frames in a given video file.

    Parameters
    ----------
    video_file_path : str
        The path to the video file.

    Returns
    -------
    num_frames : int
        The number of frames in the video file.
    """
    vidcap = cv2.VideoCapture(video_file_path)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    return num_frames

def extractFrameFromVideo(video_file_path: str, frame_nr: int):
    """
    This method loads a specific frame from a given video file.

    Parameters
    ----------
    video_file_path : str
        The path to the video file.
    frame_nr : int
        The frame number. First frame is 0.
        
    Returns
    -------
    success: bool
        Whether the loading was successful.
    frame : uint8 Array [HEIGHTxWIDTHx3]
        The frame as uint8 Array in RGB format.
    """
    try:
        vidcap = cv2.VideoCapture(video_file_path)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr-1)
        success, frame = vidcap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = None
        return success, frame
    except:
        return False, None

def getVideoResolution(video_file_path: str):
    """
    This method determines the resolution of a given video file.

    Parameters
    ----------
    video_file_path : str
        The path to the video file.

    Returns
    -------
    width: int
        Width-resolution of video in pixels.
    height: int
        Height-resolution of video in pixels.
    fps: int
        Temporal-resolution (frames-per-second) of video.
    """
    vidcap = cv2.VideoCapture(video_file_path)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = vidcap.read()
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = vidcap.get(cv2.CAP_PROP_FPS)
    if success:
        return frame.shape[1], frame.shape[0], fps
    return None

def renderAnnotatedFrame(frame, elements: dict, design: dict):
    """
    This method renders a frame with elements drawn on it (e.g. the homography 
    pattern, vehicle annotations, region_of_interest)

    Parameters
    ----------
    frame : uint8 Array [HEIGHTxWIDTHx3]
        The path to the video file.
    elements: dict
        A dictionary containing elements. Possible keys are: "homography", 
        "vehicle_annotations", "labelled_vehicle_annnotations", and 
        "region_of_interest".
    design: dict
        A dictionary containing instructions for drawing. This can include
        colors, line sizes, and labelling styles.
        
    Returns
    -------
    frame_out : uint8 Array [HEIGHTxWIDTHx3]
        The number of frames in the video file.
    """
    # generate figure
    # last_backend = matplotlib.get_backend()
    # matplotlib.use('Agg') # make a user that does not exist so window doesnt popup
    plt.ioff()
    fig = plt.figure(frameon=False)
    fig.set_size_inches(frame.shape[1]/100,frame.shape[0]/100)
    # dedicated ax so picture is same size as canvas
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)
    fig.add_axes(ax)
    # render frame
    ax.imshow(frame, aspect='auto')
    # draw homography
    if "homography" in elements:
        homography_circle = plt.Circle((elements["homography"][0], elements["homography"][1]), elements["homography"][2], 
                                       # general
                                       fill = design["homography"]["draw_fill"], 
                                       alpha = design["homography"]["draw_alpha"], 
                                       # Line Specific
                                       edgecolor = design["homography"]["line_color"],
                                       linestyle = design["homography"]["line_style"],
                                       linewidth = design["homography"]["line_width"],
                                       # Area / Fill Specific
                                       facecolor = design["homography"]["fill_color"],
                                       hatch = design["homography"]["fill_hatch"]
                                       )
        ax.add_patch(homography_circle)
    # draw region of interest
    if "region_of_interest" in elements:
        x_i, y_i = elements["region_of_interest"]["INCLUDING"].exterior.xy
        x_e, y_e = elements["region_of_interest"]["EXCLUDING"].exterior.xy
        if design["region_of_interest"]["draw_line"]:
            ax.plot(x_i, y_i, color=design["region_of_interest"]["line_color"], 
                     linestyle=design["region_of_interest"]["line_style"], 
                     linewidth=design["region_of_interest"]["line_width"], 
                     alpha=design["region_of_interest"]["draw_alpha"])
            ax.plot(x_e, y_e, color=design["region_of_interest"]["line_color"], 
                     linestyle=design["region_of_interest"]["line_style"], 
                     linewidth=design["region_of_interest"]["line_width"], 
                     alpha=design["region_of_interest"]["draw_alpha"], )
        if design["region_of_interest"]["draw_fill"]:
            ax.fill(x_i, y_i, color=design["region_of_interest"]["fill_color_in"],
                     hatch=design["region_of_interest"]["fill_hatch"],
                     alpha=design["region_of_interest"]["draw_alpha"])
            ax.fill(x_e, y_e, color=design["region_of_interest"]["fill_color_ex"], 
                     hatch=design["region_of_interest"]["fill_hatch"],
                     alpha=design["region_of_interest"]["draw_alpha"], )
    # draw annotation
    if "vehicle_annotations" in elements:
        for annotation in elements["vehicle_annotations"]:
            angle_deg = annotation[5]*360/2/3.14159
            WIDTH = annotation[3]
            HEIGHT = annotation[4]
            CENTER = (annotation[1] - WIDTH/2, annotation[2] - HEIGHT/2)
            rect_transform = tr.Affine2D().rotate_deg_around(annotation[1], annotation[2], angle_deg)  + plt.gca().transData
            rect_patch = plt.Rectangle(CENTER, WIDTH, HEIGHT, fill=False, 
                                       ec=design["vehicle_annotations"]["line_color"], 
                                       lw=design["vehicle_annotations"]["line_width"], 
                                       alpha=design["vehicle_annotations"]["draw_alpha"],
                                       transform=rect_transform)
            ax.add_patch(rect_patch)
    # convert matplot canvas back to array
    fig.canvas.draw()
    frame_out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_out = frame_out.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    gc.collect()
    # reset matplotlib
    # matplotlib.use(last_backend)
    plt.ion()
    return frame_out

def renderAnnotatedVideo(video_file_path_source: str, 
                         video_file_path_destination: str, 
                         elements: dict, design:dict, 
                         max_num_frames=None, print_status=False):
    """
    This method renders a video with elements drawn on it (e.g. the homography 
    pattern, vehicle annotations, region_of_interest).

    Parameters
    ----------
    video_file_path_source : str
        The path to the source video file.
    video_file_path_destination : str
        The path to the destination video file.
    elements: dict
        A dictionary containing elements. Possible keys are: "homography", 
        "vehicle_annotations", "labelled_vehicle_annnotations", and 
        "region_of_interest".
    design: dict
        A dictionary containing instructions for drawing. This can include
        colors, line sizes, and labelling styles.
    max_num_frames: int
        If set, the video is only rendered until a specific frame (not all frames).
        Default: None, so the whole video is rendered per default.
    print_status: bool
        If set to True, console printing takes place to inform about the process of video generation.
        Default: False, so no printing takes place.
    """
    # Open Video Reader & Writer
    vidcap = cv2.VideoCapture(video_file_path_source)
    res_width, res_height, res_fps = getVideoResolution(video_file_path_source)
    vidformat_fourcc = cv2.VideoWriter_fourcc(*'MJPG') # cv2.VideoWriter_fourcc("MJPG")
    video_writer = cv2.VideoWriter(video_file_path_destination, vidformat_fourcc, res_fps, (res_width, res_height), True)
    # Start Processing Each Frame
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_num_frames is not None:
        num_frames = max_num_frames
    for frame_counter in range(0, num_frames):
        # Load Next Frame
        success, image = vidcap.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Draw On Frame
        frame_elements = {}
        if "vehicle_annotations" in elements:
            frame_elements["vehicle_annotations"] = elements["vehicle_annotations"][frame_counter]
        if "homography" in elements:
            frame_elements["homography"] = getFrameHomography(elements["homography"], frame_counter)
        if "region_of_interest" in elements:
            frame_elements["region_of_interest"] = getTransformedRegionOfInterest(elements["region_of_interest"], elements["homography"], frame_counter)      
        edited_frame = renderAnnotatedFrame(image, frame_elements, design)
        edited_frame = cv2.cvtColor(edited_frame, cv2.COLOR_RGB2BGR)
        # Save Image to File
        video_writer.write(edited_frame)
        del image
        del edited_frame
        # Print Status
        if print_status:
            print("Video Processing...\t", frame_counter, "\t", "/", num_frames)
    # Close Video Writer
    video_writer.release()
