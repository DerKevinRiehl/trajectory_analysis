# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# paths
# source_file = "DJI_0933.MOV_circle.txt"
# source_file = "DJI_0934.MOV_circle.txt"
# source_file = "DJI_0939.MOV_circle.txt"
# source_file = "DJI_0940.MOV_circle.txt"
# source_file = "DJI_0943.MOV_circle.txt"
source_file = "DJI_0944.MOV_circle.txt"
target_file = "proc/"+source_file




# load
def loadFile(f):
    df = []
    file = open(f, "r")
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
        df.append([frame_nr,x,y,r])
        line = file.readline()
        print(line)
    file.close()
    df = pd.DataFrame(df, columns=["frame_nr", "x", "y", "r"])
    return df

def saveFile(f, df):
    fW = open(f, "w")
    for idx, row in df.iterrows():
        fW.write(str(int(row["frame_nr"])))
        fW.write("\t")
        fW.write("[[[")
        fW.write(str(row["x"]))
        fW.write("\t")
        fW.write(str(row["y"]))
        fW.write("\t")
        fW.write(str(row["r"]))
        fW.write("]]]")
        fW.write("\n")
    fW.close()



col_of_interest = "r"



# df = loadFile("DJI_0933.MOV_circle.txt")
# plt.plot(df["frame_nr"], df[col_of_interest])



df = loadFile(source_file)
plt.plot(df["frame_nr"], df[col_of_interest])



#############################################
#### Remove Peaks ###########################
#############################################
# stinky_frames = []
# last_rad = df.iloc[0][col_of_interest]
# new_r = []
# for idx, row in df.iterrows():
#     if abs(row[col_of_interest]-last_rad)>10:
#         print(row["frame_nr"])
#         stinky_frames.append(row["frame_nr"])
#         new_r.append(last_rad)
#     else:
#         last_rad = row[col_of_interest]
#         new_r.append(row[col_of_interest])
# df[col_of_interest] = new_r
# plt.plot(df["frame_nr"], df[col_of_interest])
# saveFile(target_file, df)




#############################################
#### Smoothin Rollin ###########################
#############################################

df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).median()
df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()

# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()
# df[col_of_interest] = df[col_of_interest].rolling(10, center=True, min_periods=1).mean()

plt.plot(df["frame_nr"], df[col_of_interest])

# saveFile(target_file, df)
