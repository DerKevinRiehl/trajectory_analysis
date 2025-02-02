import pandas as pd
import numpy as np

df = pd.read_csv("log_synthetic_kalman_050_coverage.txt", sep=" ", header=None)
df = df.rename(columns={0:"coverage", 1:"noise_angle", 2:"noise_pos", 3:"es_avg1", 4:"es_avg2", 5:"improvement"})

# Get unique values for noise_angle and noise_pos
noise_angles = df['noise_angle'].unique()
noise_pos = df['noise_pos'].unique()

# Sort the unique values to ensure consistent ordering
noise_angles.sort()
noise_pos.sort()

# Create an empty 2D numpy array with the correct shape
improvement_matrix = np.zeros((len(noise_angles), len(noise_pos)))

# Fill the matrix with improvement values
for i, angle in enumerate(noise_angles):
    for j, pos in enumerate(noise_pos):
        # Get the improvement value for this combination of noise_angle and noise_pos
        value = df[(df['noise_angle'] == angle) & (df['noise_pos'] == pos)]['improvement'].values
        if len(value) > 0:
            improvement_matrix[i, j] = value[0]

# Now improvement_matrix is your 2D numpy array with improvement values
# Rows represent noise_angle, columns represent noise_pos

print(improvement_matrix)