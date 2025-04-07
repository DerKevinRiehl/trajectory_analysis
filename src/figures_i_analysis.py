import os
import sys
import pickle

import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.patches import Rectangle
from _constants import VEHICLE_INFO_PATH, FILTERING_SAMPLING_FREQUENCY
from tools_trajectory_plotting import plot_accelerations
from tools_trajectory_plotting import plot_time_space_diagram


# Root Directories
DATA_ROOT = "../data_trajectories/6_final_trajectories/"
RCSN_ROOT = "../data_trajectories/7_final_trajectories_reconstructed/"

ALL_VIDEOS = [
    #"DJI_0933.MOV", "DJI_0934.MOV", "DJI_0939.MOV", "DJI_0940.MOV", 
    "DJI_0943.MOV", "DJI_0944.MOV"
]

# RELEVANT_VIDEO = "DJI_0943.MOV"
RELEVANT_VIDEO = "DJI_0944.MOV"

# Plot settings
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 


###################################################################################
# Main: Figure 10
###################################################################################
# Load
RELEVANT_VIDEO = "DJI_0943.MOV"
df_43 = pd.read_csv(RCSN_ROOT + RELEVANT_VIDEO + "_norelax.txt", sep=",")
df_43_info = pd.read_csv(VEHICLE_INFO_PATH + RELEVANT_VIDEO + ".txt", sep="\t")
df_43 = plot_accelerations(df_43)
plt.close()
ts_43, te_43 = 50, 100 # DJI_0943

RELEVANT_VIDEO = "DJI_0944.MOV"
df_44 = pd.read_csv(RCSN_ROOT + RELEVANT_VIDEO + "_norelax.txt", sep=",")
df_44_info = pd.read_csv(VEHICLE_INFO_PATH + RELEVANT_VIDEO + ".txt", sep="\t")
df_44 = plot_accelerations(df_44)
plt.close()
ts_44, te_44 = 40, 100 # DJI_0944

fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=100)

plot_time_space_diagram(df_43, start_frame=ts_43*FILTERING_SAMPLING_FREQUENCY, end_frame=te_43*FILTERING_SAMPLING_FREQUENCY, axis=axs[0])
handles, labels = axs[1].get_legend_handles_labels()
axs[0].get_legend().remove()
axs[0].set_xlim([ts_43, te_43])
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Lane-Coordinate Position [m]")
axs[0].set_title("(a) DJI_0943", fontweight="bold")

plot_time_space_diagram(df_44, start_frame=ts_44*FILTERING_SAMPLING_FREQUENCY, end_frame=te_44*FILTERING_SAMPLING_FREQUENCY, axis=axs[1])
handles, labels = axs[1].get_legend_handles_labels()
axs[1].get_legend().remove()
axs[1].set_xlim([ts_44, te_44])
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Lane-Coordinate Position [m]")
axs[1].set_title("(b) DJI_0944", fontweight="bold")

#labels = [f"V_{l}" for l in labels]
#fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -0.05), ncol=15, fontsize="small")
fig.tight_layout()
#fig.savefig(f"./fig_i_TSDsnippet.pdf", bbox_inches='tight', dpi=100)
plt.show()
sys.exit(1)

###################################################################################
# Main: Figure 11
###################################################################################
from tools_stability_analysis import estimate_L2gain_CTHpolicy, estimate_LinfinityGain_CTHpolicy
from tools_stability_analysis import speed_standard_deviation, speed_dft
from _constants import default_plotting_settings

RELEVANT_VIDEO = "DJI_0943.MOV"
start_frame, end_frame = int(ts_43*FILTERING_SAMPLING_FREQUENCY), int(te_43*FILTERING_SAMPLING_FREQUENCY)
L2gains_df = estimate_L2gain_CTHpolicy(df_43, start_frame=start_frame, end_frame=end_frame)
LinfGains_df = estimate_LinfinityGain_CTHpolicy(df_43, start_frame=int(ts_43*FILTERING_SAMPLING_FREQUENCY), end_frame=int(te_43*FILTERING_SAMPLING_FREQUENCY))
speed_std_df = speed_standard_deviation(df_43[(df_43["Frame_ID"] >= start_frame) & (df_43["Frame_ID"] <= end_frame)], start_frame=start_frame)
dft_df = speed_dft(df_43[(df_43["Frame_ID"] >= start_frame) & (df_43["Frame_ID"] <= end_frame)])

exp_43_df = L2gains_df.merge(speed_std_df, on=["Vehicle_ID"], how="left")
exp_43_df = exp_43_df.merge(LinfGains_df, on=["Vehicle_ID"], how="left")
exp_43_df = exp_43_df.merge(dft_df, on=["Vehicle_ID"], how="left")
exp_43_df = exp_43_df.merge(df_43_info, on=["Vehicle_ID"], how="left")
exp_43_df["Video"] = RELEVANT_VIDEO
exp_43_df = exp_43_df.sort_values(by=["Vehicle_ID"]).reset_index()
analysis_df = exp_43_df.copy()

RELEVANT_VIDEO = "DJI_0944.MOV"
start_frame, end_frame = int(ts_44*FILTERING_SAMPLING_FREQUENCY), int(te_44*FILTERING_SAMPLING_FREQUENCY)
L2gains_df = estimate_L2gain_CTHpolicy(df_44, start_frame=start_frame, end_frame=end_frame)
LinfGains_df = estimate_LinfinityGain_CTHpolicy(df_44, start_frame=int(ts_44*FILTERING_SAMPLING_FREQUENCY), end_frame=int(te_44*FILTERING_SAMPLING_FREQUENCY))
speed_std_df = speed_standard_deviation(df_44[(df_44["Frame_ID"] >= start_frame) & (df_44["Frame_ID"] <= end_frame)], start_frame=start_frame)
dft_df = speed_dft(df_44[(df_44["Frame_ID"] >= start_frame) & (df_44["Frame_ID"] <= end_frame)])

exp_44_df = L2gains_df.merge(speed_std_df, on=["Vehicle_ID"], how="left")
exp_44_df = exp_44_df.merge(LinfGains_df, on=["Vehicle_ID"], how="left")
exp_44_df = exp_44_df.merge(dft_df, on=["Vehicle_ID"], how="left")
exp_44_df = exp_44_df.merge(df_44_info, on=["Vehicle_ID"], how="left")
exp_44_df["Video"] = RELEVANT_VIDEO
exp_44_df = exp_44_df.sort_values(by=["Vehicle_ID"]).reset_index()
analysis_df = pd.concat((analysis_df, exp_44_df))
print(analysis_df.columns)


fig, axs = plt.subplots(1, 4, figsize=(12, 3), dpi=100)
sns.violinplot(data=exp_43_df, x="Powertrain", y="L2gain_Speed", 
               palette=default_plotting_settings["color_palette"], ax=axs[0], cut=0.1)
axs[0].set_xlabel("Engine Type")
axs[0].set_ylabel("$\gamma_v$")

sns.violinplot(data=exp_43_df[exp_43_df["L2gain_SpaceHdwy"] >= 0.1], x="Powertrain", y="L2gain_SpaceHdwy", 
               palette=default_plotting_settings["color_palette"], ax=axs[1], cut=0.1)
axs[1].set_xlabel("Engine Type")
axs[1].set_ylabel("$\gamma_s$")

sns.violinplot(data=exp_43_df[exp_43_df["Powertrain"] == "ICES"], x="Gearbox", y="L2gain_Speed", 
               palette=default_plotting_settings["color_palette"], ax=axs[2], cut=0.1)
axs[2].set_xlabel("Gearbox Type")
axs[2].set_ylabel("$\gamma_v$")

sns.violinplot(data=exp_43_df[(exp_43_df["Powertrain"] == "ICES") & (exp_43_df["L2gain_SpaceHdwy"] >= 0.1)], x="Gearbox", y="L2gain_SpaceHdwy", 
               palette=default_plotting_settings["color_palette"], ax=axs[3], cut=0.1)
axs[3].set_xlabel("Gearbox Type")
axs[3].set_ylabel("$\gamma_s$")
fig.tight_layout()


fig, axs = plt.subplots(1, 4, figsize=(12, 3), dpi=100)
sns.violinplot(data=exp_44_df, x="Powertrain", y="L2gain_Speed", 
               palette=default_plotting_settings["color_palette"], ax=axs[0], cut=0.1)
axs[0].set_xlabel("Engine Type")
axs[0].set_ylabel("$\gamma_v$")

sns.violinplot(data=exp_44_df[exp_44_df["L2gain_SpaceHdwy"] >= 0.1], x="Powertrain", y="L2gain_SpaceHdwy", 
               palette=default_plotting_settings["color_palette"], ax=axs[1], cut=0.1)
axs[1].set_xlabel("Engine Type")
axs[1].set_ylabel("$\gamma_s$")

sns.violinplot(data=exp_44_df[exp_44_df["Powertrain"] == "ICES"], x="Gearbox", y="L2gain_Speed", 
               palette=default_plotting_settings["color_palette"], ax=axs[2], cut=0.1)
axs[2].set_xlabel("Gearbox Type")
axs[2].set_ylabel("$\gamma_v$")

sns.violinplot(data=exp_44_df[(exp_44_df["Powertrain"] == "ICES") & (exp_44_df["L2gain_SpaceHdwy"] >= 0.1)], x="Gearbox", y="L2gain_SpaceHdwy", 
               palette=default_plotting_settings["color_palette"], ax=axs[3], cut=0.1)
axs[3].set_xlabel("Gearbox Type")
axs[3].set_ylabel("$\gamma_s$")
fig.tight_layout()

fig, axs = plt.subplots(1, 4, figsize=(12, 3), dpi=100)
sns.violinplot(data=analysis_df, x="Powertrain", y="Speed_Std_Dev", 
               palette=default_plotting_settings["color_palette"], ax=axs[0], cut=0.1)
axs[0].set_xlabel("Engine Type")
axs[0].set_ylabel("$\sigma_v$")

sns.violinplot(data=analysis_df[analysis_df["Powertrain"] == "ICES"], x="Gearbox", y="Speed_Std_Dev", 
               palette=default_plotting_settings["color_palette"], ax=axs[1], cut=0.1)
axs[1].set_xlabel("Gearbox Type")
axs[1].set_ylabel("$\sigma_v$")

sns.violinplot(data=analysis_df, x="Powertrain", y="Max_Amplitude", 
               palette=default_plotting_settings["color_palette"], ax=axs[2], cut=0.1)
axs[2].set_xlabel("Engine Type")
axs[2].set_ylabel("Max. Amplitude of Oscillations")

sns.violinplot(data=analysis_df, x="Powertrain", y="Frequency", 
               palette=default_plotting_settings["color_palette"], ax=axs[3], cut=0.1)
axs[3].set_xlabel("Engine Type")
axs[3].set_ylabel("Frequency of Oscillations")
fig.tight_layout()

plt.show()

