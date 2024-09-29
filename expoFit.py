import os

import matplotlib as plt
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit

from data_analysis import *
import numpy as np
# fit an exponential function to the sine function, general form is \theta (t) = \theta_0 e^{−t/τ}\cos
# (2πt/T + φ)

data = [data_analysis(f"runs/{file}") for file in os.listdir("runs") if file.endswith(".csv")]

deg = True

if deg:
    for i in data:
        i[0]["theta"] = i[0]["theta"] * np.pi / 180


periods = list()
# take the average of the period
for i in data:
    periods.extend(i[1])
    periods.extend(i[2])


fit = lambda t, theta_not, tau, phi, T: theta_not * np.exp(-t / tau) * np.cos(2 * np.pi * t / T + phi)
"""
Simply averaging the data would make phase difference accumulate
and no reasonable result can be obtained.

we must fit individual data to the function and then average the parameters, expect for phi, theta_0
"""

params = list()
for i in data:
    x = i[0]["time"]
    y = i[0]["theta"]
    theta_not = (y.max() - y.min())/2
    a = [theta_not]
    a.extend(curve_fit(lambda t, tau, phi, T: fit(t, theta_not, tau, phi, T), x, y)[0])
    params.append(a)

params = np.array(params)
tau = np.mean(params[:, 1])
T = np.mean(params[:, 3])

print(f"tau: {tau} +/- {np.std(params[:, 1])/np.sqrt(len(params))}")
print(f"T: {T} +/- {np.std(params[:, 3])/np.sqrt(len(params))}")


for n in range(len(data)):
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.4, 1]})
    i = data[n]
    time = np.linspace(0, i[0]["time"].max(), 1000)
    ax1.plot(i[0]["time"], i[0]["theta"], label="Data")
    ax1.plot(time, fit(time, *params[n]), label="Fit")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (rad)")
    ax1.set_title("Angle vs Time")
    ax1.legend()
    # plot the residuals as in period time
    ax2.plot(i[0]["time"], i[0]["theta"] - fit(i[0]["time"], *params[n]), label="Error", color="red")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals")
    ax2.legend()
    plt.subplots_adjust(hspace=0.5)
    plt.show()



