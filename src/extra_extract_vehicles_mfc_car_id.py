"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish, Kevin Riehl
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2024
Submitted to:   JOURNAL
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

################################################################
# Constants
################################################################
DB_PATH = "C:/Users/selbaklish/Desktop/Python_Workspace/co2mpas_driver/co2mpas_driver/db/EuroSegmentCar_cleaned.csv"


################################################################
# Functions
################################################################
def get_speed_acceleration_curve(car_id, save_interp = False):
    sol = driver(
        dict(
            vehicle_id=car_id, # A sample car id from the database
            inputs=dict(
                inputs=dict(
                    gear_shifting_style=0.8,
                    desired_velocity=124/3.6,
                    starting_velocity=0,
                    driver_style=0.6,
                    sim_start=0,
                    sim_step=0.1,
                    duration=100,
                    degree=4,
                    use_linear_gs=True,
                    use_cubic=False,

                )
            )
        )
    )["outputs"]
    discrete_acceleration_curves = sol["discrete_acceleration_curves"]
    fig = plt.figure()
    for curve in discrete_acceleration_curves:
        sp_bins = list(curve["x"])
        acceleration = list(curve["y"])
        plt.plot(sp_bins, acceleration)
    plt.plot(sol["velocities"][1:], sol["accelerations"][1:])
    plt.xlabel("Speed", fontsize=18)
    plt.ylabel("Acceleration", fontsize=16)
    plt.legend(
        [
            "acceleration per gear 0",
            "acceleration per gear 1",
            "acceleration per gear 2",
            "acceleration per gear 3",
            "acceleration per gear 4",
            "final acceleration",
        ]
    )
    plt.grid()

    speeds = np.linspace(0, 50, 1000)
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

    plt.figure()
    plt.plot(speeds, accelerations, label="a_p")
    plt.plot(speeds, a_p_spl(speeds), label="a_p spline", linestyle="--")
    plt.plot(speeds, 0.5*accelerations, label="a_d")
    plt.plot(speeds, decelerations, label="d_p")
    plt.plot(speeds, d_p_spl(speeds), label="d_p spline", linestyle="--")
    plt.legend()
    plt.show()

    if save_interp:
        with open(f'ID{car_id}_AccelCapInterp.pkl', 'wb') as f:
            pickle.dump(a_p_spl, f)
        with open(f'ID{car_id}_DecelCapInterp.pkl', 'wb') as f:
            pickle.dump(d_p_spl, f)


################################################################
# Main
################################################################
df = pd.read_csv(DB_PATH, encoding="ISO-8859-1", index_col=0)
# print(df.columns)

# NOTE: The vehicles' numbers are according to Table 4 in the Scientific Reports paper.
# NOTE: The obtained MFC_CarID for all vehicles are yet mapped to the corresponding vehicles information for each video.

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
