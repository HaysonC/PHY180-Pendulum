import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv("pendulum1.csv")

# Print the first few rows of the data
print(df[:5])


""" 
# get max and min values
max = []
min = []
current_max = 0
current_min = 1e9
max_counter = 0
min_counter = 0
find_max = True
for i in range(len(data)):
    if find_max:
        if(data["y"][i] > current_max):
            current_max = data["y"][i]
            max_counter = 0
        else:
            max_counter += 1
            if max_counter > 10:
                max.append((data["time"][i-10],current_max))
                current_max = 0
                max_counter = 0
                find_max = False
    else:
        if (data["y"][i] < current_min):
            current_min = data["y"][i]
            min_counter = 0
        else:
            min_counter += 1
            if min_counter > 10:
                min.append((data["time"][i-10],current_min))
                current_min = 0
                min_counter = 0
                find_max = True

print(max)
print(min)
"""
start_from = "left"
# add derivative column to the data
df["derivative"] = np.zeros(len(df))

# the shae of a sine graph, we need to take the origin of the pendulum, which is (miny, mean x)
# the length of the pendulum is the distance between the origin and the bob. We need to convert pixel to real length

origin = (min(df["y"]) if start_from == "left" else max(df["y"]), df["x"].mean())

df["theta"] = np.where(
    df["y"] > origin[0],
    np.arctan((df["x"] - origin[1]) / (df["y"] - origin[0])) * 180 / np.pi,
    -np.arctan((df["x"] - origin[1]) / (df["y"] - origin[0])) * 180 / np.pi,
)

lenghtSting = 0.025  # 0.025 m is 250mm


# we need to obtain the period of the pendulum, and label sections of the pendulum and we would asisng a period number to each section
# use noise filtering to compute the deritivie of the angle, and then use the zero crossing to compute the period
last_derivative = 0
alpha = 0.5
for i in range(1, len(df)):
    df["derivative"][i] = (
        alpha * (df["y"][i] - df["y"][i - 1]) + (1 - alpha) * last_derivative
    )
    last_derivative = df["derivative"][i]
zeroCrossing = []
i = 1
while i < len(df):
    if df["derivative"][i] * df["derivative"][i - 1] <= 0 and df["derivative"][i] > 0:
        zeroCrossing.append(df["time"][i])
        i += 100
        continue
    i += 1


# for every other zero corssing we would assign a period number
period = []
for i in range(1, len(zeroCrossing)):
    period.append((zeroCrossing[i]))


# for each period we would assign a period number to each section of the dataframe
df["period"] = np.zeros(len(df))
df["period"][df["time"] < period[0]] = 1
for i in range(1, len(period)):
    df["period"][df["time"] >= period[i]] = i + 1

print(df)

# for each section record the max and min values of the angle
max = dict()
min = dict()
for i in range(1, len(period)):
    max[i] = df["theta"][df["period"] == i].max()
    min[i] = df["theta"][df["period"] == i].min()

print(max)
print(min)

# plot total time of the period vs the angle
for i in max.keys():
    plt.plot(max[i], len(df["time"][df["period"] == i]) / 120, "ro")
for i in min.keys():
    plt.plot(min[i], len(df["time"][df["period"] == i]) / 120, "ro")
# plot the angle vs the period
# scale the graph, y axis from 0 to 2
plt.xlabel("Angle (degrees)")
plt.ylabel("Period (s)")
plt.ylim(0, 2)
plt.show()

"""
ratio = lenghtSting / (max(data["y"]) - min(data["y"]))
# we need to convert the pixel to real length
data["y"] = (data["y"] - origin[0]) * ratio
data["x"] = (data["x"] - origin[1]) * ratio

 """
