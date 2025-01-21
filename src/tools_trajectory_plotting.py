"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish, Kevin Riehl
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import sys
import mplcursors
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional

from _constants import default_plotting_settings


# #############################################################################
# METHODS
# #############################################################################
def plot_time_space_diagram(trajectory_df: pd.DataFrame, start_frame: Optional[int] = 0, end_frame: Optional[int] = None,
                            vehicle_labels_in_plot: Optional[bool] = False):
    """
    This methods plots the time space diagram showing the x-lane coordinates of all vehicles across time.

    Parameters
    ----------
    trajectory_df: pd.DataFrame
        A dataframe containing the trajectory information for all vehicles.
    start_frame: Optional[int] = 0
        The starting frame to begin plotting from.
    end_frame: Optional[int] = None
        The ending frame to end plotting at.
    vehicle_labels_in_plot: Optional[bool] = False
        Flag to add text annptations with vehicle IDs next to the trajectory end inside the plot.

    Returns
    -------
    None
    """
    if end_frame is None:
        end_frame = trajectory_df["Frame_ID"].max()
    plot_df = trajectory_df[(trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
    plot_df[["vehicleMeaningless","vehicleIndexNumber"]] = plot_df["Vehicle_ID"].str.split("_", n=1, expand=True)
    plot_df = plot_df.drop(columns=["vehicleMeaningless"])
    plot_df["vehicleIndexNumber"] = plot_df["vehicleIndexNumber"].astype(int)
    plot_df = plot_df.sort_values(by=["Frame_ID", "vehicleIndexNumber"])

    palette = sns.color_palette(default_plotting_settings["color_palette"])
    plt.figure(figsize=default_plotting_settings["figure_size"])
    ax = sns.lineplot(plot_df, x="Global_Time", y="Lane_X", hue="vehicleIndexNumber", 
                 palette=palette, linewidth=default_plotting_settings["line_width"])
    plt.grid(alpha=default_plotting_settings["grid_alpha"], linestyle=default_plotting_settings["grid_line_style"])
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=7, title="Vehicle ID", frameon=True,
    )

    if vehicle_labels_in_plot:
        unique_vehicles = plot_df["vehicleIndexNumber"].unique()
        for vehicle_idx in unique_vehicles:
            vehicle_df = plot_df[plot_df["vehicleIndexNumber"] == vehicle_idx]
            idx = vehicle_df["Frame_ID"].idxmax()
            plt.text(x=vehicle_df.loc[idx, "Global_Time"].item()+0.2, y=vehicle_df.loc[idx, "Lane_X"].item(), s=f"V_{vehicle_idx}", 
                    color=palette[vehicle_idx-1], fontsize=10, weight="bold")
    plt.xlim([-1, plot_df["Global_Time"].max()+5])


def plot_oblique_trajectories(trajectory_df: pd.DataFrame, start_frame: Optional[int] = 0, end_frame: Optional[int] = None,
                              vehicle_labels_in_plot: Optional[bool] = False, active_cursor: Optional[bool] = False) -> None:
    """
    This methods plots the time space diagram showing the oblique x-lane coordinates of all vehicles across time.
    Oblique x-lane coordinates are extracted using a transformation in order to rotate the trajectories to be horizontal in the plot.
    Shear transformation: [t_transformed, x_transformed] = [[1, 0], [sin(theta), cos(theta)]] * [t, x]

    Parameters
    ----------
    trajectory_df: pd.DataFrame
        A dataframe containing the trajectory information for all vehicles.
    start_frame: Optional[int] = 0
        The starting frame to begin plotting from.
    end_frame: Optional[int] = None
        The ending frame to end plotting at.
    vehicle_labels_in_plot: Optional[bool] = False
        Flag to add text annptations with vehicle IDs next to the trajectory end inside the plot.
    active_cursor: Optional[bool] = False
        Flag to activate mplcursors to show trajectory data when hovering.
    
    Returns
    -------
    None
    """
    def _show_annotations(sel):
        try:
            row = plot_df.iloc[sel.index]
            sel.annotation.set_text(f"{row['Vehicle_ID']}\nLane_X={row['Lane_X']:.2f}\nSpace_Hdwy={row['Space_Hdwy']:.2f}")
        except (KeyError, TypeError) as e:
            pass
    
    if end_frame is None:
        end_frame = trajectory_df["Frame_ID"].max()
    plot_df = trajectory_df[(trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
    plot_df[["vehicleMeaningless","vehicleIndexNumber"]] = plot_df["Vehicle_ID"].str.split("_", n=1, expand=True)
    plot_df = plot_df.drop(columns=["vehicleMeaningless"])
    plot_df["vehicleIndexNumber"] = plot_df["vehicleIndexNumber"].astype(int)
    plot_df = plot_df.sort_values(by=["Frame_ID", "vehicleIndexNumber"])

    vehicle_id = np.random.choice(plot_df["Vehicle_ID"].unique(), size=1)[0]
    vehicle_df = plot_df[plot_df["Vehicle_ID"] == vehicle_id].copy()
    vehicle_start_pos = vehicle_df.loc[vehicle_df["Frame_ID"] == start_frame, "Lane_X"].item()
    vehicle_end_pos = vehicle_df.loc[vehicle_df["Frame_ID"] == end_frame, "Lane_X"].item()
    start_time = vehicle_df.loc[vehicle_df["Frame_ID"] == start_frame, "Global_Time"].item()
    end_time = vehicle_df.loc[vehicle_df["Frame_ID"] == end_frame, "Global_Time"].item()
    rot_angle = -np.arctan2(vehicle_end_pos-vehicle_start_pos, end_time-start_time)
    plot_df["Oblique_Lane_X"] = plot_df["Global_Time"] * np.sin(rot_angle) + plot_df["Lane_X"] * np.cos(rot_angle)

    palette = sns.color_palette(default_plotting_settings["color_palette"])
    plt.figure(figsize=default_plotting_settings["figure_size"])
    ax = sns.lineplot(plot_df, x="Global_Time", y="Oblique_Lane_X", hue="vehicleIndexNumber", 
                 palette=palette, linewidth=default_plotting_settings["ObliqueTrajectories"]["line_width"])
    h, l = ax.get_legend_handles_labels()

    if active_cursor:
        tsd = sns.scatterplot(data=plot_df, x="Global_Time", y="Oblique_Lane_X", hue="vehicleIndexNumber", 
                              palette=palette, alpha=0.05)
        cursor1 = mplcursors.cursor(tsd, hover=True)
        cursor1.connect('add', _show_annotations)
    
    # Define xlim and ylim
    min_t, max_t = -1, end_time+5
    min_x, max_x = plot_df["Oblique_Lane_X"].min()-2, plot_df["Oblique_Lane_X"].max()+20

    # plot oblique grid lines
    oblique_grid_times = np.linspace(start_time, end_time, default_plotting_settings["ObliqueTrajectories"]["oblique_grid_num_ticks"])
    for t in oblique_grid_times:
        d = vehicle_df.loc[vehicle_df["Global_Time"] == t, "Lane_X"].item()
        oblique_d_min_t = min_t * np.sin(rot_angle) + d * np.cos(rot_angle)
        t_min_x = (min_x - d * np.cos(rot_angle)) / np.sin(rot_angle)
        plt.plot([min_t, t_min_x], [oblique_d_min_t, min_x], 
                 linewidth=default_plotting_settings["ObliqueTrajectories"]["oblique_grid_line_width"], 
                 linestyle=default_plotting_settings["ObliqueTrajectories"]["oblique_grid_line_style"], 
                 color=default_plotting_settings["ObliqueTrajectories"]["oblique_grid_line_color"],
                 alpha=default_plotting_settings["ObliqueTrajectories"]["oblique_grid_alpha"])
        t_text = (max_x-10 - d * np.cos(rot_angle)) / np.sin(rot_angle)
        if t_text < min_t:
            continue
        plt.text(x=t_text-2.0, y=max_x-10, s=f"{d:.1f} m",
                 color=default_plotting_settings["ObliqueTrajectories"]["oblique_grid_line_color"],
                 fontsize=6, bbox=dict(facecolor='white', edgecolor="white"))

    plt.grid(alpha=default_plotting_settings["grid_alpha"], linestyle=default_plotting_settings["grid_line_style"])
    ax.legend(h, l, bbox_to_anchor=(.5, 1), loc="lower center", ncol=7, title="Vehicle ID", frameon=True)

    if vehicle_labels_in_plot:
        unique_vehicles = plot_df["vehicleIndexNumber"].unique()
        for vehicle_idx in unique_vehicles:
            vehicle_df = plot_df[plot_df["vehicleIndexNumber"] == vehicle_idx]
            idx = vehicle_df["Frame_ID"].idxmax()
            plt.text(x=vehicle_df.loc[idx, "Global_Time"].item()+0.2, y=vehicle_df.loc[idx, "Oblique_Lane_X"].item(), s=f"V_{vehicle_idx}", 
                    color=palette[vehicle_idx-1], fontsize=10, weight="bold")
    plt.xlim([min_t, max_t])
    plt.ylim([min_x, max_x])


def plot_velocities_and_headways(trajectory_df: pd.DataFrame, start_frame: Optional[int] = 0, end_frame: Optional[int] = None) -> None:
    """
    This methods plots the velocities, time headways and space headways across time for all vehicles.

    Parameters
    ----------
    trajectory_df: pd.DataFrame
        A dataframe containing the trajectory information for all vehicles.
    start_frame: Optional[int] = 0
        The starting frame to begin plotting from.
    end_frame: Optional[int] = None
        The ending frame to end plotting at.
    
    Returns
    -------
    None
    """
    if end_frame is None:
        end_frame = trajectory_df["Frame_ID"].max()

    plot_df = trajectory_df[(trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
    plot_df[["vehicleMeaningless","vehicleIndexNumber"]] = plot_df["Vehicle_ID"].str.split("_", n=1, expand=True)
    plot_df = plot_df.drop(columns=["vehicleMeaningless"])
    plot_df["vehicleIndexNumber"] = plot_df["vehicleIndexNumber"].astype(int)
    plot_df = plot_df.sort_values(by=["Frame_ID", "vehicleIndexNumber"])
    
    palette = sns.color_palette(default_plotting_settings["color_palette"])
    fig, axs = plt.subplots(1, 3, figsize=[default_plotting_settings["figure_size"][0]*3, default_plotting_settings["figure_size"][1]])
    p = sns.lineplot(data=plot_df, x="Global_Time", y="v_Vel", hue="vehicleIndexNumber", ax=axs[0],
                 palette=palette, linewidth=default_plotting_settings["ObliqueTrajectories"]["line_width"])
    h, l = p.get_legend_handles_labels()
    try:
        sns.lineplot(data=plot_df, x="Global_Time", y="Space_Hdwy", hue="vehicleIndexNumber", ax=axs[1],
                 palette=palette, linewidth=default_plotting_settings["line_width"])
    except:
        print(plot_df.head())
        sys.exit(1)
    sns.lineplot(data=plot_df, x="Global_Time", y="Time_Hdwy", hue="vehicleIndexNumber", ax=axs[2],
                 palette=palette, linewidth=default_plotting_settings["line_width"])
    
    axs[0].grid(alpha=default_plotting_settings["grid_alpha"], linestyle=default_plotting_settings["grid_line_style"])
    axs[1].grid(alpha=default_plotting_settings["grid_alpha"], linestyle=default_plotting_settings["grid_line_style"])
    axs[2].grid(alpha=default_plotting_settings["grid_alpha"], linestyle=default_plotting_settings["grid_line_style"])

    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[2].get_legend().remove()
    fig.legend(h, l, loc="upper center", ncol=plot_df["Vehicle_ID"].nunique(), title="Vehicle ID", frameon=True)

    return trajectory_df


def plot_accelerations(trajectory_df: pd.DataFrame, start_frame: Optional[int] = 0, end_frame: Optional[int] = None) -> pd.DataFrame:
    if not "v_Accel" in trajectory_df.columns:
        unique_vehicles = trajectory_df["Vehicle_ID"].unique()
        accel_df = None
        for vehicle_id in unique_vehicles:
            vehicle_df = trajectory_df[trajectory_df["Vehicle_ID"] == vehicle_id].copy()
            if not vehicle_df["Frame_ID"].is_monotonic_increasing:
                vehicle_df = vehicle_df.sort_values(by=["Frame_ID"])
            sampling_interval = vehicle_df["Global_Time"].diff(1).mean()
            vehicle_df["v_Accel"] = vehicle_df["v_Vel"].diff(1).shift(-1).fillna(0) / sampling_interval
            vehicle_df = vehicle_df[["Frame_ID", "Vehicle_ID", "v_Accel"]]
            if accel_df is None:
                accel_df = vehicle_df.copy()
            else:
                accel_df = pd.concat((accel_df, vehicle_df))
        trajectory_df = trajectory_df.merge(accel_df, on=["Frame_ID", "Vehicle_ID"], how="left")

    if end_frame is None:
        end_frame = trajectory_df["Frame_ID"].max()

    plot_df = trajectory_df[(trajectory_df["Frame_ID"] >= start_frame) & (trajectory_df["Frame_ID"] <= end_frame)].copy()
    plot_df[["vehicleMeaningless","vehicleIndexNumber"]] = plot_df["Vehicle_ID"].str.split("_", n=1, expand=True)
    plot_df = plot_df.drop(columns=["vehicleMeaningless"])
    plot_df["vehicleIndexNumber"] = plot_df["vehicleIndexNumber"].astype(int)
    plot_df = plot_df.sort_values(by=["Frame_ID", "vehicleIndexNumber"])

    plt.figure()
    palette = sns.color_palette(default_plotting_settings["color_palette"])
    ax = sns.lineplot(data=plot_df, x="Global_Time", y="v_Accel", hue="vehicleIndexNumber", 
                      palette=palette, linewidth=default_plotting_settings["line_width"])
    plt.grid(alpha=default_plotting_settings["grid_alpha"], linestyle=default_plotting_settings["grid_line_style"])
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=7, title="Vehicle ID", frameon=True,
    )

    return trajectory_df