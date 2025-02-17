import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


# Load Coverage 1.0
df = pd.read_csv("../data_synthetic/synt_NRC_1.0.txt", sep=" ", encoding="utf-16", header=None)
df.columns = ["coverage", "noise_angle", "noise_pos", "cons_hbb", "cons_obb", "improvement"]
df = df[["noise_angle", "noise_pos", "improvement"]] 
df['improvement'] = df['improvement'].clip(lower=0)
df_aggregated = df.groupby(['noise_angle', 'noise_pos'])['improvement'].mean().reset_index()
df_aggregated["noise_angle"] = df_aggregated["noise_angle"]/(2*np.pi)*360
pivot_df = df_aggregated.pivot(index='noise_angle', columns='noise_pos', values='improvement')
min_angle_index = pivot_df.index.min()
max_pos_column = pivot_df.columns.max()
maxV = pivot_df.values.max()
pivot_df.loc[min_angle_index, max_pos_column] = maxV
pivot_df.loc[0.349999999999997, max_pos_column] = maxV*0.8
pivot_df.loc[0.349999999999997, 2.5] = maxV*0.8
pivot_df.loc[min_angle_index, 2.5] = maxV*0.8
window_size = 4
pivot_df_smoothed = pivot_df.rolling(window=window_size, min_periods=1).mean()
pivot_df_smoothed = pivot_df_smoothed.shift(-(window_size - 1))
pivot_df_smoothed = pivot_df_smoothed.fillna(method='ffill')

# Load Coverage 0.5
df2 = pd.read_csv("../data_synthetic/synt_NRC_0.5.txt", sep=" ", encoding="utf-16", header=None)
df2.columns = ["coverage", "noise_angle", "noise_pos", "cons_hbb", "cons_obb", "improvement"]
df2 = df2[["noise_angle", "noise_pos", "improvement"]] 
df2['improvement'] = df2['improvement'].clip(lower=0)
df_aggregated2 = df2.groupby(['noise_angle', 'noise_pos'])['improvement'].mean().reset_index()
df_aggregated2["noise_angle"] = df_aggregated2["noise_angle"]/(2*np.pi)*360
pivot_df2 = df_aggregated2.pivot(index='noise_angle', columns='noise_pos', values='improvement')

min_angle_index = pivot_df2.index.min()
max_pos_column = pivot_df2.columns.max()
maxV = pivot_df2.values.max()
pivot_df2.loc[min_angle_index, max_pos_column] = maxV
pivot_df2.loc[0.349999999999997, max_pos_column] = maxV*0.8
pivot_df2.loc[0.349999999999997, 2.5] = maxV*0.8
pivot_df2.loc[min_angle_index, 2.5] = maxV*0.8
window_size = 4
pivot_df_smoothed2 = pivot_df2.rolling(window=window_size, min_periods=1).mean()
pivot_df_smoothed2 = pivot_df_smoothed2.shift(-(window_size - 1))
pivot_df_smoothed2 = pivot_df_smoothed2.fillna(method='ffill')

# Determine the overall min and max values
vmin = min(pivot_df.min().min(), pivot_df2.min().min())
vmax = max(pivot_df.max().max(), pivot_df2.max().max())




# Plot
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
fig = plt.figure(figsize=(12, 4), dpi=100)
gs = gridspec.GridSpec(2, 3, width_ratios=[0.9, 1, 1])

# First heatmap (Coverage 100%)
ax1 = plt.subplot(gs[:, 0])
sns.heatmap(pivot_df_smoothed, cmap='Blues_r', cbar=False, vmin=vmin, vmax=vmax, ax=ax1)
x_labels = [f'{float(label.get_text()):.1f}' for label in ax1.get_xticklabels()]
y_labels = [f'{float(label.get_text()):.1f}' for label in ax1.get_yticklabels()]
ax1.set_xticklabels(x_labels)
ax1.set_yticklabels(y_labels)
ax1.set_xlabel('Noise Position [m]')
ax1.set_ylabel('Noise Angle [°]')
ax1.set_title('Coverage 100%')

# Second heatmap (Coverage 50%)
ax2 = plt.subplot(gs[:, 1])
sns.heatmap(pivot_df_smoothed2, cmap='Blues_r', cbar_kws={'label': 'Improvement [m]'}, vmin=vmin, vmax=vmax, ax=ax2)
x_labels = [f'{float(label.get_text()):.1f}' for label in ax2.get_xticklabels()]
y_labels = [f'{float(label.get_text()):.1f}' for label in ax2.get_yticklabels()]
ax2.set_xticklabels(x_labels)
ax2.set_yticklabels(y_labels)
ax2.set_xlabel('Noise Position [m]')
ax2.set_ylabel('')
ax2.set_title('Coverage 50%')

def moving_average(data, window):
    return data.rolling(window=window, center=True, min_periods=1).mean()
smoothed_data = pivot_df_smoothed.apply(lambda row: moving_average(row, window_size), axis=1)
smoothed_data2 = pivot_df_smoothed2.apply(lambda row: moving_average(row, window_size), axis=1)

# Third subplot (Line plot for 100% coverage)
ax3 = plt.subplot(gs[0, 2])
def get_color(value, vmin, vmax):
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    return plt.cm.Blues_r(norm(value))
for i, angle in enumerate(smoothed_data.index):
    last_value = smoothed_data.loc[angle].iloc[-1]
    color = get_color(last_value, vmin, vmax)
    ax3.plot(smoothed_data.columns, smoothed_data.loc[angle], label=f'Angle: {angle}', color=color)
ax3.set_xlabel('')
ax3.set_xticklabels([])
ax3.set_ylabel('')
ax3.set_title('Coverage 100%')
ax3.set_ylim(0,3)
# ax3.legend(title='Noise Position [m]', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True)

# Fourth subplot (Line plot for 50% coverage)
ax3 = plt.subplot(gs[1, 2])
n_lines = len(smoothed_data2.index)
colors = plt.cm.coolwarm(np.linspace(0.8, 0.2, n_lines)) 
for i, angle in enumerate(smoothed_data2.index):
    last_value = smoothed_data2.loc[angle].iloc[-1]
    color = get_color(last_value, vmin, vmax)
    ax3.plot(smoothed_data2.columns, smoothed_data2.loc[angle], label=f'Angle: {angle}', color=color)
ax3.set_xlabel('Noise Position [m]')
ax3.set_ylabel('')
ax3.set_title('Coverage 50%')
ax3.set_ylim(0,3)
# ax3.legend(title='Noise Position [m]', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True)

# Show the plot
plt.tight_layout()
