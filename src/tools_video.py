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
import numpy as np
from tools_homography import getFrameHomography, getTransformedRegionOfInterest, transformPointFrom_CARTESIAN_2_PIX
import gc
import matplotlib.transforms as mtransforms
import matplotlib.patches as patches
import time



HISTORY_LENGTH = 25*4

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
        The frame as uint8 Array in RGB format.
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
        The annotated frame as uint8 Array in RGB format.
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

    # labelled annotation
    if "labelled_vehicle_annnotations" in elements:
        for annotation in elements["labelled_vehicle_annnotations"]:
            WIDTH = annotation[3]
            HEIGHT = annotation[4]
            CENTER = (annotation[1] - WIDTH/2, annotation[2] - HEIGHT/2)
            circle_patch = plt.Circle([annotation[1], annotation[2]], design["labelled_vehicle_annnotations"]["circle_radius"], fill=False, 
                                        ec=design["labelled_vehicle_annnotations"]["line_color"], 
                                        lw=design["labelled_vehicle_annnotations"]["line_width"], 
                                        alpha=design["labelled_vehicle_annnotations"]["draw_alpha"],
                                        linestyle=design["labelled_vehicle_annnotations"]["line_style"],)
            ax.add_patch(circle_patch)
            ax.scatter(annotation[1], annotation[2], color="white", s=20)
            ax.text(annotation[1], annotation[2]-design["labelled_vehicle_annnotations"]["circle_radius"]-design["labelled_vehicle_annnotations"]["font_size"], 
                      str(annotation[-1]), 
                      horizontalalignment='center',
                      color=design["labelled_vehicle_annnotations"]["font_color"],
                      fontsize=design["labelled_vehicle_annnotations"]["font_size"])
    
    # final trajectories
    if "final_trajectory" in elements:
        for idx, row in elements["final_trajectory"].iterrows():
            WIDTH = row["v_Width"]
            HEIGHT = row["v_Length"]
            CENTER = (row["x_pixel"] - WIDTH/2, row["y_pixel"] - HEIGHT/2)
            circle_patch = plt.Circle([row["x_pixel"], row["y_pixel"]], design["labelled_vehicle_annnotations"]["circle_radius"], fill=False, 
                                        ec=design["labelled_vehicle_annnotations"]["line_color"], 
                                        lw=design["labelled_vehicle_annnotations"]["line_width"], 
                                        alpha=design["labelled_vehicle_annnotations"]["draw_alpha"],
                                        linestyle=design["labelled_vehicle_annnotations"]["line_style"],)
            ax.add_patch(circle_patch)
            ax.scatter(row["x_pixel"], row["y_pixel"], color="white", s=20)
            ax.text(row["x_pixel"], row["y_pixel"]-design["labelled_vehicle_annnotations"]["circle_radius"]-design["labelled_vehicle_annnotations"]["font_size"], 
                      str(row["Vehicle_ID"]), 
                      horizontalalignment='center',
                      color=design["labelled_vehicle_annnotations"]["font_color"],
                      fontsize=design["labelled_vehicle_annnotations"]["font_size"])
    if "final_trajectory_history" in elements:
        counter = 0
        actual_history_length = len(elements["final_trajectory_history"])
        for val in elements["final_trajectory_history"]:
            if counter<10:
                counter+=1
                continue
            for idx, row in val.iterrows():
                WIDTH = row["v_Width"]
                HEIGHT = row["v_Length"]
                ax.scatter(row["x_pixel"], row["y_pixel"], color="blue", s=20, alpha=(1.0/actual_history_length)*(actual_history_length-counter))
            counter += 1
    
    # convert matplot canvas back to array
    fig.canvas.draw()
    frame_out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_out = frame_out.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    gc.collect()
    # reset matplotlib
    plt.ion()
    return frame_out

def renderTransformedFrame(frame, transformation):
    """
    This method renders a frame with elements drawn on it (e.g. the homography 
    pattern, vehicle annotations, region_of_interest)

    Parameters
    ----------
    frame : uint8 Array [HEIGHTxWIDTHx3]
        The frame as uint8 Array in RGB format.
    transformation: mtransforms.Affine2D
        A transformation function.
        
    Returns
    -------
    frame_out : uint8 Array [HEIGHTxWIDTHx3]
        The transformed frame as uint8 Array in RGB format.
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
    image_context = ax.imshow(frame, aspect='auto')
    image_context.set_transform(transformation)
    # convert matplot canvas back to array
    fig.canvas.draw()
    frame_out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_out = frame_out.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    gc.collect()
    # reset matplotlib
    # matplotlib.use(last_backend)
    plt.ion()
    frame_out = np.flip(frame_out, axis=0)
    return frame_out

def getVehicleEgoPerspectiveTransformation(frame: np.array, kalman_frame_coordinates: list, kalman_frame_angle: float, zoom_factor=5):
    """
    This method calculate a transformation function to display a frame with a vehicle in its center, zoomed in, and rotated by the vehicle angle.

    Parameters
    ----------
    frame : uint8 Array [HEIGHTxWIDTHx3]
        The frame as uint8 Array in RGB format.
    kalman_frame_coordinates: list[float]
        The annotation coordinates (row) from a Kalman filtered trajectory file.
    kalman_frame_angle: float
        The angle of the vehicle in radians from a Kalman filtered trajectory file.
        
    Returns
    -------
    transformation : mtransforms.Affine2D
        The transformation function.
    """
    transformation = mtransforms.Affine2D().translate(-kalman_frame_coordinates[0], -kalman_frame_coordinates[1]).scale(
        zoom_factor, zoom_factor).rotate(kalman_frame_angle-np.pi/2).translate(frame.shape[1]/2, frame.shape[0]/2)
    return transformation

def renderAnnotatedVideo(video_file_path_source: str, 
                         video_file_path_destination: str, 
                         elements: dict, design:dict, 
                         start_frame=None,
                         end_frame=None, print_status=False):
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
    start_frame: int
        If set, the video is rendered beginning from this frame.
        Default: None, so the video starts from beginning.
    end_frame: int
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
    if end_frame is not None:
        num_frames = end_frame
    start_frame_used = 0
    if start_frame is not None:
        start_frame_used = start_frame
        for i in range(0, start_frame):
            success, image = vidcap.read()
    for frame_counter in range(start_frame_used, num_frames):
        # Load Next Frame
        success, image = vidcap.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Draw On Frame
        frame_elements = {}
        if "vehicle_annotations" in elements:
            frame_elements["vehicle_annotations"] = elements["vehicle_annotations"][frame_counter]
        if "labelled_vehicle_annnotations" in elements:
            frame_elements["labelled_vehicle_annnotations"] = elements["labelled_vehicle_annnotations"][frame_counter]
        if "homography" in elements:
            frame_elements["homography"] = getFrameHomography(elements["homography"], frame_counter)
        if "region_of_interest" in elements:
            frame_elements["region_of_interest"] = getTransformedRegionOfInterest(elements["region_of_interest"], elements["homography"], frame_counter)      
        if "labelled_final_trajectory" in elements:
            df_final_trajectory = elements["labelled_final_trajectory"]
            lst_df_history = []
            for t_frame_counter in range(frame_counter, frame_counter-HISTORY_LENGTH, -1):
                if t_frame_counter<0:
                    continue
                df_frame = df_final_trajectory[df_final_trajectory["Frame_ID"]==t_frame_counter]
                positions_transformed_x = []
                positions_transformed_y = []
                for idx, row in df_frame.iterrows():
                    pos = [row["Cartesian_X"], row["Cartesian_Y"]]
                    posT = transformPointFrom_CARTESIAN_2_PIX(pos, frame_elements["homography"])
                    positions_transformed_x.append(posT[0])
                    positions_transformed_y.append(posT[1])
                df_frame["x_pixel"] = positions_transformed_x
                df_frame["y_pixel"] = positions_transformed_y
                lst_df_history.append(df_frame)
            frame_elements["final_trajectory"] = lst_df_history[0]
            frame_elements["final_trajectory_history"] = lst_df_history
        edited_frame = renderAnnotatedFrame(image, frame_elements, design)
        if "transformation" in elements:
            df_kalman_vehicle_data = elements["transformation"]["kalman"]
            df_vehicle_frame = df_kalman_vehicle_data[df_kalman_vehicle_data["frame_nr"]==frame_counter]
            kalman_annotations = df_vehicle_frame.iloc[0].tolist()
            kalman_coordinate = kalman_annotations[4:5+1]
            kalman_frame_coordinates = transformPointFrom_CARTESIAN_2_PIX(kalman_coordinate, frame_elements["homography"])
            kalman_frame_angle = kalman_annotations[8]
            if "norotation" in elements["transformation"]:
                kalman_frame_angle = 0
            transformation = getVehicleEgoPerspectiveTransformation(edited_frame, kalman_frame_coordinates, kalman_frame_angle, elements["transformation"]["zoom"][frame_counter])
            edited_frame = renderTransformedFrame(edited_frame, transformation)
        if "hud" in elements:
            df_space_headway = elements["hud"]["headway"]
            space_headway = df_space_headway[df_space_headway["Frame_ID"]==frame_counter]["Space_Hdwy"].iloc[0]
            df_trajectory_data = elements["hud"]["history"]
            df_history = df_trajectory_data[df_trajectory_data["Frame_ID"]<=frame_counter]
            df_history = df_history[df_history["Frame_ID"]>=frame_counter-elements["hud"]["horizon"]]
            positions_transformed_x = []
            positions_transformed_y = []
            for idx, row in df_history.iterrows():
                pos = [row["Cartesian_X"], row["Cartesian_Y"]]
                posT = transformPointFrom_CARTESIAN_2_PIX(pos, frame_elements["homography"])
                positions_transformed_x.append(posT[0])
                positions_transformed_y.append(posT[1])
            df_history["x_pixel"] = positions_transformed_x
            df_history["y_pixel"] = positions_transformed_y
            edited_frame = renderHUDAnnotatedFrame(edited_frame, kalman_annotations, df_history, space_headway, elements["hud"]["horizon"])
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


def renderHUDAnnotatedFrame(frame, kalman_annotations, df_history, space_headway, history_horizon):
    """
    This method renders a frame with elements drawn on it (e.g. the homography 
    pattern, vehicle annotations, region_of_interest)

    Parameters
    ----------
    frame : uint8 Array [HEIGHTxWIDTHx3]
        The frame as uint8 Array in RGB format.
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
        The annotated frame as uint8 Array in RGB format.
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

    # render statistics text
    rect = patches.Rectangle((100-40, 50), 1600, 550, linewidth=1, edgecolor='none', facecolor='gray', alpha=0.5)
    ax.add_patch(rect)

    sep=20
    fsize=50
    lblx = 100
    valx = 900
    ystrt = 250
    xoffst = 500
    unitxofst=20
    ax.text(lblx, ystrt-(fsize+sep)*2, "Time",             color="white", fontsize=fsize)
    ax.text(lblx, ystrt+(fsize+sep)*0, "Position (x, y)",  color="white", fontsize=fsize)
    ax.text(lblx, ystrt+(fsize+sep)*1, "Angle",            color="white", fontsize=fsize)
    ax.text(lblx, ystrt+(fsize+sep)*2, "Velocity",         color="white", fontsize=fsize)
    ax.text(lblx, ystrt+(fsize+sep)*3, "Ang. Velocity",    color="white", fontsize=fsize)
    ax.text(lblx, ystrt+(fsize+sep)*4, "Space Headway:",       color="white", fontsize=fsize)

    time_frames = kalman_annotations[0]
    time_seconds = time_frames/25
    time_formatted = time.strftime('%M:%S', time.gmtime(time_seconds))
    ax.text(valx, ystrt-(fsize+sep)*2, time_formatted, color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+xoffst+100, ystrt-(fsize+sep)*2, str(int(time_frames))+" Frames", color="white", fontsize=fsize, horizontalalignment='right')

    ax.text(valx, ystrt+(fsize+sep)*0, "{:.2f}".format(kalman_annotations[6+0]), color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+xoffst, ystrt+(fsize+sep)*0, "{:.2f}".format(kalman_annotations[6+1]), color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+unitxofst, ystrt+(fsize+sep)*0, "m", color="white", fontsize=fsize, horizontalalignment='left')
    ax.text(valx+xoffst+unitxofst, ystrt+(fsize+sep)*0, "m", color="white", fontsize=fsize, horizontalalignment='left')
    
    ax.text(valx, ystrt+(fsize+sep)*1, "{:.2f}".format(kalman_annotations[6+2]), color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+xoffst, ystrt+(fsize+sep)*1, "{:.2f}".format(360*kalman_annotations[6+2]/(2*np.pi)), color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+unitxofst, ystrt+(fsize+sep)*1, "'", color="white", fontsize=fsize, horizontalalignment='left')
    ax.text(valx+xoffst+unitxofst, ystrt+(fsize+sep)*1, "°", color="white", fontsize=fsize, horizontalalignment='left')
    
    ax.text(valx, ystrt+(fsize+sep)*2, "{:.2f}".format(abs(kalman_annotations[6+3]*25*10)), color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+xoffst, ystrt+(fsize+sep)*2, "{:.2f}".format(abs(kalman_annotations[6+3]*3.6*25*10)), color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+unitxofst, ystrt+(fsize+sep)*2, "m/s", color="white", fontsize=fsize, horizontalalignment='left')
    ax.text(valx+xoffst+unitxofst, ystrt+(fsize+sep)*2, "km/h", color="white", fontsize=fsize, horizontalalignment='left')
    
    ax.text(valx, ystrt+(fsize+sep)*3, "{:.2f}".format(kalman_annotations[6+4]), color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+xoffst, ystrt+(fsize+sep)*3, "{:.2f}".format(360*kalman_annotations[6+4]/(2*np.pi)), color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+unitxofst, ystrt+(fsize+sep)*3, "'/s", color="white", fontsize=fsize, horizontalalignment='left')
    ax.text(valx+xoffst+unitxofst, ystrt+(fsize+sep)*3, "°/s", color="white", fontsize=fsize, horizontalalignment='left')

    ax.text(valx, ystrt+(fsize+sep)*4, "{:.2f}".format(space_headway), color="white", fontsize=fsize, horizontalalignment='right')
    ax.text(valx+unitxofst, ystrt+(fsize+sep)*4, "m", color="white", fontsize=fsize, horizontalalignment='left')

    # draw history trajectory   
    xstrt = 100
    ystrt = 1100
    hstH = 500
    historyLbl = "Space_Hdwy"
    historyMax = np.nanmax(df_history[historyLbl])
    historyMin = np.nanmin(df_history[historyLbl])
    df_history["val"] = df_history[historyLbl]-historyMin
    df_history["val"] = df_history["val"]/(historyMax-historyMin)
    
    rect = patches.Rectangle((100-40, ystrt-250), 1000, 800, linewidth=1, edgecolor='none', facecolor='gray', alpha=0.5)
    ax.add_patch(rect)
    ax.text(lblx, ystrt-100-(fsize+sep)*1, "Historical Space Headway", color="white", fontsize=fsize)
    yscatter = (ystrt+(1-df_history["val"])*hstH).tolist()
    xscatter = xstrt+np.arange(len(yscatter))*8
    if len(xscatter)==len(yscatter):
        ax.scatter(xscatter, yscatter, color="white", s=100)
        ax.scatter(xscatter[-1], yscatter[-1], color="cyan", s=400)
        ax.text(xscatter[-1], yscatter[-1]-fsize/2, "{:.2f}".format(space_headway), color="white", fontsize=fsize, horizontalalignment='center')
        ax.plot([xscatter[0], xscatter[-1]], [ystrt, ystrt], "--", color="white")
        ax.plot([xscatter[0], xscatter[-1]], [ystrt+hstH, ystrt+hstH], "--", color="white")
        ax.text(np.nanmean(xscatter), ystrt+fsize*1.5, "{:.2f}".format(historyMax), color="white", fontsize=fsize, horizontalalignment='center')
        ax.text(np.nanmean(xscatter), ystrt+hstH-fsize/2, "{:.2f}".format(historyMin), color="white", fontsize=fsize, horizontalalignment='center')

    # convert matplot canvas back to array
    fig.canvas.draw()
    frame_out = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_out = frame_out.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    gc.collect()
    # reset matplotlib
    plt.ion()
    return frame_out