"""
Consistent Vehicle Trajectory Extraction From Aerial Recordings Using Oriented Object Detection
-------------------------------------------
Authors:        Kevin Riehl, Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024 - 2025
Submitted to:   Scientific Reports
-------------------------------------------
"""
################################################################
# IMPORTS
################################################################
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from co2mpas_driver import dsp as driver # This needs a specific conda environment with mfc_env_requirements.txt
from _constants import VEHICLE_INFO_PATH

################################################################
# Constants
################################################################
DB_PATH = "C:/Users/selbaklish/Desktop/Python_Workspace/co2mpas_driver/co2mpas_driver/db/EuroSegmentCar_cleaned.csv"
HP2KW = 0.7457

################################################################
# Functions
################################################################
def determine_MFC_CarID(engine_type: str, transmission_type: str, production_year: int, max_power_hp: float, max_torque_nm: float):
    mfc_df = pd.read_csv(DB_PATH, encoding="ISO-8859-1", index_col=0)
    car_id = None
    if engine_type == "ICES":
        subdf = mfc_df[(mfc_df["General Specifications-Transmission"] == transmission_type.lower()) & (mfc_df["General Specifications-Release date"] <= production_year) & (mfc_df["General Specifications-End date"] >= str(production_year))].copy()
        subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - max_torque_nm).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - max_power_hp*HP2KW).abs() / subdf["Fuel Engine-Max power"].mean()
        car_id = subdf["ClosenessMeasure"].idxmin()

        print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
        print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} kW, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
        print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')
    elif engine_type == "Electric":
        subdf = mfc_df[(mfc_df["General Specifications-Release date"] <= production_year)].copy() # & (mfc_df["General Specifications-End date"] >= str(production_year))
        subdf["ClosenessMeasure"] = (subdf["Electric Engine-Max torque"] - max_torque_nm).abs() / subdf["Electric Engine-Max torque"].mean() + (subdf["Electric Engine-Total max power"] - max_power_hp*HP2KW).abs() / subdf["Electric Engine-Total max power"].mean()
        car_id = subdf["ClosenessMeasure"].idxmin()

        print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
        print(f' Power = {subdf.loc[car_id, "Electric Engine-Total max power"]} kW, Torque = {subdf.loc[car_id, "Electric Engine-Max torque"]} Nm')
        print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')
    else:
        raise NotImplementedError

    return car_id


def get_speed_acceleration_curve(car_id, save_interp = False):
    sol = driver(
        dict(
            vehicle_id=car_id, # A sample car id from the database
            db_path=DB_PATH,
            inputs=dict(
                inputs=dict(
                    degree=4,
                    use_linear_gs=True,
                    use_cubic=False,
                )
            )
        )
    )["outputs"]
    #print(sol.keys())
    #print(sol['vehicle_mass'], sol['vehicle_max_speed'])
    #print(type(sol['discrete_car_res_curve_force']), sol['discrete_car_res_curve_force'].shape)
    #print(type(sol['sp_bins']), sol['sp_bins'].shape)
    #plt.figure()
    #plt.plot(sol['sp_bins'], sol['discrete_car_res_curve_force'])
    
    speeds = np.linspace(0, sol['vehicle_max_speed'], 1000)
    accelerations = np.zeros_like(speeds)
    decelerations = np.zeros_like(speeds)
    all_decelerations = np.zeros(shape=(len(sol["curves_dec"]), len(speeds)))
    for i in range(len(accelerations)):
        for j in range(len(sol["curves"])):
            try:
                accelerations[i] = max(accelerations[i], sol["curves"][j](speeds[i]))
            except ValueError:
                continue
        for j in range(len(sol["curves_dec"])):
            try:
                decelerations[i] = min(decelerations[i], sol["curves_dec"][j](speeds[i]))
                all_decelerations[j, i] = sol["curves_dec"][j](speeds[i])
            except ValueError:
                continue

    a_p_spl = interpolate.Akima1DInterpolator(speeds, accelerations)
    d_p_spl = interpolate.Akima1DInterpolator(speeds, decelerations)

    #plt.figure()
    #plt.plot(sol['sp_bins'], (1.03*sol['vehicle_mass']*a_p_spl(sol['sp_bins'])+sol['discrete_car_res_curve_force'])*sol['sp_bins'])

    plt.figure()
    plt.plot(speeds, accelerations, label="a_p")
    plt.plot(speeds, a_p_spl(speeds), label="a_p spline", linestyle="--")
    plt.plot(speeds, 0.5*accelerations, label="a_d")
    plt.plot(speeds, decelerations, label="d_p")
    plt.plot(speeds, d_p_spl(speeds), label="d_p spline", linestyle="--")
    plt.legend()
    plt.show()

    if save_interp:
        with open(VEHICLE_INFO_PATH + f'ID{car_id}_AccelCapInterp.pkl', 'wb') as f:
            pickle.dump(a_p_spl, f)
        with open(VEHICLE_INFO_PATH + f'ID{car_id}_DecelCapInterp.pkl', 'wb') as f:
            pickle.dump(d_p_spl, f)


################################################################
# Main
################################################################
# NOTE: The vehicles' numbers are according to Table 4 in the Scientific Reports paper.
# NOTE: The obtained MFC_CarID for all vehicles are yet mapped to the corresponding vehicles information for each video.

# RELEVANT_VIDEO = "DJI_0933.MOV"
# RELEVANT_VIDEO = "DJI_0934.MOV"
# RELEVANT_VIDEO = "DJI_0939.MOV"
# RELEVANT_VIDEO = "DJI_0940.MOV"
# RELEVANT_VIDEO = "DJI_0943.MOV"
RELEVANT_VIDEO = "DJI_0944.MOV"

df_info = pd.read_csv(VEHICLE_INFO_PATH + RELEVANT_VIDEO + ".txt", sep="\t")
print(df_info.head())
for idx, row in df_info.iterrows():
    print("*********************************************")
    print(f"**************** {row['Vehicle_ID']} ******************")
    car_id = determine_MFC_CarID(engine_type=row['Powertrain'], transmission_type=row['Gearbox'], production_year=row['Year'], 
                                 max_power_hp=row['Max_Power_HP'], max_torque_nm=row['Max_Torque_Nm'])
    df_info.loc[idx, 'MFC_CarID'] = car_id
    get_speed_acceleration_curve(car_id,  save_interp = True)
    print()
df_info.to_csv(VEHICLE_INFO_PATH + RELEVANT_VIDEO + ".txt", sep="\t", index=False)
sys.exit(1)

df = pd.read_csv(DB_PATH, encoding="ISO-8859-1", index_col=0)
# print(df.columns)

# VEHICLE_1
vehicle_id = "VEHICLE_1"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "automatic") & (df["General Specifications-Release date"] <= 2022)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 340).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 150).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_2 (Electric)
vehicle_id = "VEHICLE_2"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Release date"] <= 2022)].copy()
subdf["ClosenessMeasure"] = (subdf["Electric Engine-Max torque"] - 266).abs() / subdf["Electric Engine-Max torque"].mean() + (subdf["Electric Engine-Total max power"] - 218).abs() / subdf["Electric Engine-Total max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Electric Engine-Total max power"]} hp, Torque = {subdf.loc[car_id, "Electric Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_3
vehicle_id = "VEHICLE_3"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "manual") & (df["General Specifications-Release date"] <= 2015)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 340).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 150).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_4
vehicle_id = "VEHICLE_4"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "automatic") & (df["General Specifications-Release date"] <= 2017)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 250).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 150).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_5
vehicle_id = "VEHICLE_5"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "automatic") & (df["General Specifications-Release date"] <= 2014)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 340).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 150).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_6
vehicle_id = "VEHICLE_6"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "automatic") & (df["General Specifications-Release date"] <= 2015)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 400).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 177).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_7
vehicle_id = "VEHICLE_7"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "manual") & (df["General Specifications-Release date"] <= 2015)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 160).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 120).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_8
vehicle_id = "VEHICLE_8"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "manual") & (df["General Specifications-Release date"] <= 2018)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 240).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 150).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_9
vehicle_id = "VEHICLE_9"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "automatic") & (df["General Specifications-Release date"] <= 2019)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 200).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 125).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_10
vehicle_id = "VEHICLE_10"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "automatic") & (df["General Specifications-Release date"] <= 2021)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 1000).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 600).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_11
vehicle_id = "VEHICLE_11"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "manual") & (df["General Specifications-Release date"] <= 2021)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 160).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 86).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_12
vehicle_id = "VEHICLE_12"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "manual") & (df["General Specifications-Release date"] <= 2015)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 330).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 270).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_13 (Electric)
vehicle_id = "VEHICLE_13"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Release date"] <= 2014)].copy()
subdf["ClosenessMeasure"] = (subdf["Electric Engine-Max torque"] - 270).abs() / subdf["Electric Engine-Max torque"].mean() + (subdf["Electric Engine-Total max power"] - 115).abs() / subdf["Electric Engine-Total max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Electric Engine-Total max power"]} hp, Torque = {subdf.loc[car_id, "Electric Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_14
vehicle_id = "VEHICLE_14"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "automatic") & (df["General Specifications-Release date"] <= 2016)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 400).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 190).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)


# VEHICLE_15
vehicle_id = "VEHICLE_15"
print("*********************************************")
print(f"**************** {vehicle_id} ******************")
subdf = df[(df["General Specifications-Transmission"] == "manual") & (df["General Specifications-Release date"] <= 2019)].copy()
subdf["ClosenessMeasure"] = (subdf["Fuel Engine-Max torque"] - 130).abs() / subdf["Fuel Engine-Max torque"].mean() + (subdf["Fuel Engine-Max power"] - 87).abs() / subdf["Fuel Engine-Max power"].mean()
car_id = subdf["ClosenessMeasure"].idxmin()

print(f'ID = {car_id}\n Release Date {subdf.loc[car_id, "General Specifications-Release date"]}, End Date = {subdf.loc[car_id, "General Specifications-End date"]}')
print(f' Power = {subdf.loc[car_id, "Fuel Engine-Max power"]} hp, Torque = {subdf.loc[car_id, "Fuel Engine-Max torque"]} Nm')
print(f' Performance: Top speed = {subdf.loc[car_id, "Performance-Top speed"]} km/h, 0-100 km/h acceleration = {subdf.loc[car_id, "Performance-Acceleration 0-100 km/h"]} m/s2 \n')

get_speed_acceleration_curve(car_id,  save_interp = True)
