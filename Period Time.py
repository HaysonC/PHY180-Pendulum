import math
import os
from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from data_analysis import *

bin_size = 0.1

# include all csv files from the direcotry runs
data = [
    data_analysis("runs/" + file)
    for file in os.listdir("runs")
    if file.endswith(".csv")
]
period_time_data = []
totalData = 0
totalPeriodData = 0
# get the total abount of datapoints
for file in os.listdir("runs"):
    if file.endswith(".csv"):
        df = pd.read_csv("runs/" + file)
        totalData += len(df)
print(f"total data: {totalData}")
for df, periods, anti_periods in data:
    # for each section record the max and min values of the angle
    max = dict()
    min = dict()
    for i in range(1, len(periods)):
        max[i] = df["theta"][df["period"] == i].max()

    # plot total time of the period vs the angle, ignore the first two period and the last two period
    # collect data for Period_Timex
    for i in range(1, len(anti_periods)):
        min[i] = df["theta"][df["anti period"] == i].min()

    for i in max.keys():
        if i == 1 or i == len(max) or i == 2 or i == len(max) - 1:
            continue
        period_time_data.append({"Angle": max[i], "Period Time": periods[i]})

    for i in min.keys():
        if i == 1 or i == len(min) or i == 2 or i == len(min) - 1:
            continue
        period_time_data.append({"Angle": min[i], "Period Time": anti_periods[i]})
    totalPeriodData += len(periods) + len(anti_periods)

print(f"total period data: {totalPeriodData}")
# sort it by angle
period_time_data = sorted(period_time_data, key=lambda x: x["Angle"])
# create DataFrame from collected data
Period_Time = pd.DataFrame(period_time_data)
# convert to radians
Period_Time["Angle"] = Period_Time["Angle"] * math.pi / 180
# bin the data by angle of 5 degrees
min_value = Period_Time["Angle"].min()  # Minimum value in the column
max_value = Period_Time["Angle"].max()  # Maximum value in the column
bin_edges = pd.interval_range(start=min_value, end=max_value, freq=bin_size)
Period_Time["Angle"] = pd.cut(Period_Time["Angle"], bins=bin_edges)
# Period_Time["Angle"] = Period_Time["Angle"].apply(
#     lambda x: int(x / bin_size) * bin_size
# )

# xerr a/4
xerr = bin_size / 4
# yerr is the standard deviation of the period time
n = Period_Time.groupby("Angle").count().reset_index()["Period Time"]
yn = Period_Time.groupby("Angle").count().reset_index()["Period Time"]
yerr = Period_Time.groupby("Angle").std().reset_index()["Period Time"] / np.sqrt(n)
Period_Time = Period_Time.groupby("Angle").mean().reset_index()


model = np.poly1d(np.polyfit(Period_Time["Angle"], Period_Time["Period Time"], 2))
a = model[2]
b = model[1]
c = model[0]


def fit_curve(t, a=a, b=b, c=c):
    return a * t**2 + b * t + c


# sigma is the standard deviation of the residuals
sigma = np.sqrt(
    np.sum((Period_Time["Period Time"] - fit_curve(Period_Time["Angle"])) ** 2)
    / (len(Period_Time) - 3)
)
# r_value is the correlation coefficient, do it from definition
r_value = np.corrcoef(Period_Time["Period Time"], fit_curve(Period_Time["Angle"]))[0, 1]

# error is sigma/sqrt(n)
n = len(Period_Time)
error = sigma / math.sqrt(n)
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2.4, 1]})
ax1.plot(Period_Time["Angle"], fit_curve(Period_Time["Angle"]), label="Regression Line")
ax1.scatter(
    Period_Time["Angle"],
    Period_Time["Period Time"],
    label="Data Points",
    s=5,
    color="red",
)
ax1.errorbar(
    Period_Time["Angle"],
    Period_Time["Period Time"],
    yerr=yerr,
    xerr=xerr,
    fmt="none",
    label="Error Bar",
    capsize=3,
    capthick=1,
    color="red",
)

ax1.set_xlabel("Angle (deg)")
ax1.set_ylabel("Period Time (s)")
# ax1.ylim([1.25, 1.75])
ax1.legend()
ax1.set_title("Period Time vs Angle")
ax1.set_ylim(bottom=1.40, top=1.60)
plt.subplots_adjust(hspace=0.5)
# ax2 plot the error between the data and the regression line
ax2.scatter(
    Period_Time["Angle"],
    Period_Time["Period Time"] - (fit_curve(Period_Time["Angle"])),
    label="Error",
    s=5,
    color="red",
)
# horzontal line at y = 0
ax2.axhline(y=0, color="black", linestyle="-")
# draw light grey -- line at +sigma and -sigma and +2sigma and -2sigma with 0.5 alpha
ax2.axhline(y=sigma, color="grey", linestyle="--", alpha=0.5)
ax2.axhline(y=-sigma, color="grey", linestyle="--", alpha=0.5)
ax2.axhline(y=2 * sigma, color="grey", linestyle="--", alpha=0.5)
ax2.axhline(y=-2 * sigma, color="grey", linestyle="--", alpha=0.5)
# in the middle of the line of sigma, overlay $+\sigma$ and $-\sigma$ and so on
largest = Period_Time["Angle"].max() + 10
ax2.text(largest, sigma, r"$+\sigma$", ha="right", color="grey")
ax2.text(largest, -sigma, r"$-\sigma$", ha="right", color="grey")
ax2.text(largest, 2 * sigma, r"$+2\sigma$", ha="right", color="grey")
ax2.text(largest, -2 * sigma, r"$-2\sigma$", ha="right", color="grey")
# make more dense y axis lelves
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.errorbar(
    Period_Time["Angle"],
    Period_Time["Period Time"] - fit_curve(Period_Time["Angle"]),
    yerr=yerr,
    xerr=xerr,
    fmt="none",
    label="Error Bar",
    capsize=3,
    capthick=1,
    color="red",
)
ax2.set_xlabel("Angle (deg)")
ax2.set_ylabel("Residual (s)")

plt.show()

# print n, sigma, r^2
print("n:", n)
print("sigma:", sigma)
print("r^2:", r_value**2)

# print a, b, c
print("a:", a)
print("b:", b)
print("c:", c)
