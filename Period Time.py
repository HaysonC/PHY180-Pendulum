import math
from cProfile import label
from xxlimited_35 import error

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from data_analysis import *

# for each section record the max and min values of the angle
max = dict()
min = dict()
for i in range(1, len(periods)):
    max[i] = df["theta"][df["period"] == i].max()

print(max)

# plot total time of the period vs the angle, ignore the first two period and the last two period
# collect data for Period_Timex
for i in range(1, len(anti_periods)):
    min[i] = df["theta"][df["anti period"] == i].min()


print(min)

period_time_data = []
for i in max.keys():
    if i == 1 or i == len(max) or i == 2 or i == len(max) - 1:
        continue
    period_time_data.append({"Angle": max[i], "Period Time": periods[i]})

for i in min.keys():
    if i == 1 or i == len(min) or i == 2 or i == len(min) - 1:
        continue
    period_time_data.append({"Angle": min[i], "Period Time": anti_periods[i]})

# sort it by angle
period_time_data = sorted(period_time_data, key=lambda x: x["Angle"])
# create DataFrame from collected data
Period_Time = pd.DataFrame(period_time_data)

model = np.poly1d(np.polyfit(Period_Time["Angle"], Period_Time["Period Time"], 2))
a = model[2]
b = model[1]
c = model[0]
print("error:", error)


def quadratic(t, a=a, b=b, c=c):
    return a * t**2 + b * t + c


# sigma is the standard deviation of the residuals
sigma = np.sqrt(
    np.sum((Period_Time["Period Time"] - quadratic(Period_Time["Angle"])) ** 2)
    / (len(Period_Time) - 3)
)
# r_value is the correlation coefficient, do it from definition
r_value = np.corrcoef(Period_Time["Period Time"], quadratic(Period_Time["Angle"]))[0, 1]

# error is sigma/sqrt(n)
n = len(Period_Time)
error = sigma / math.sqrt(n)

""" 
plt.xlabel("Angle (deg)")
plt.ylabel("Period Time (s)")
plt.title("Period Time vs Angle")
plt.legend()
plt.ylim(0.7, 0.9)
# plot ax2 at the bottom, show difference between the data and the regression line, error bar included
# refernce code from fit_black_box.py
plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.figsize'] = 10, 9
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
ax2.errorbar(Period_Time["Angle"], Period_Time["Period Time"], yerr=error, fmt=".", label="data", color="black")
ax2.plot(Period_Time["Angle"], Period_Time["Angle"] * slope + intercept, label="best fit", color="black")
ax2.legend(loc='upper right')
ax2.set_xlabel("Angle")
ax2.set_ylabel("Period Time")

plt.show()
"""
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2.4, 1]})
ax1.plot(Period_Time["Angle"], quadratic(Period_Time["Angle"]), label="Regression Line")
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
    yerr=error,
    fmt="none",
    label="Error Bar",
    capsize=3,
    capthick=1,
    color="red",
)

ax1.set_xlabel("Angle (deg)")
ax1.set_ylabel("Period Time (s)")
ax1.legend()
ax1.set_title("Period Time vs Angle")
ax1.set_ylim(bottom=0)
plt.subplots_adjust(hspace=0.5)
# ax2 plot the error between the data and the regression line
ax2.scatter(
    Period_Time["Angle"],
    Period_Time["Period Time"] - (quadratic(Period_Time["Angle"])),
    label="Error",
    s=5,
    color="black",
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
    Period_Time["Period Time"] - quadratic(Period_Time["Angle"]),
    yerr=error,
    fmt="none",
    label="Error Bar",
    capsize=3,
    capthick=1,
    color="black",
)
ax2.set_xlabel("Angle (deg)")
ax2.set_ylabel("Residual (s)")

plt.show()

# print n, sigma, r^2
print("n:", n)
print("sigma:", sigma)
print("r^2:", r_value**2)
