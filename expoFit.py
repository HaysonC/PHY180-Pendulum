import os

import matplotlib as plt
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit

from data_analysis import *
import numpy as np
import matplotlib.font_manager as fm
gs_font = fm.FontProperties(
                fname='/System/Library/Fonts/Supplemental/GillSans.ttc')
# fit an exponential function to the sine function, general form is \theta (t) = \theta_0 e^{−t/τ}\cos
# (2πt/T + φ)
dir = "runs"
data = [data_analysis(f"{dir}/{file}") for file in os.listdir(dir) if file.endswith(".csv")]

deg = False

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

Q1 = []
Q2 = []
for n in range(len(data)):
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.4, 1]})
    i = data[n]
    time = np.linspace(0, i[0]["time"].max(), 1000)
    # opac = 0.5
    ax1.plot(i[0]["time"], i[0]["theta"], label="Data", color="blue", alpha=0.5)
    ax1.plot(time, fit(time, *params[n]), label="Fit", color="orange")
    ax1.set_yticks(np.linspace(-np.pi / 2, np.pi / 2, 9))
    ax1.set_yticklabels(
        ["$-\pi/2$", "$-3\pi/8$", "$-\pi/4$", "$-\pi/8$", "0", "$\pi/8$", "$\pi/4$", "$3\pi/8$", "$\pi/2$"])
    ax1.axhline(y=0, color='grey', linestyle='--')
    # also plot the e^(-t/tau) and -e^(-t/tau) for reference
    tau = params[n][1]
    T = params[n][3]

    """
    method 1:
        Q = 2pi * tau/T
    """
    Q = np.pi * tau/T
    Q1.append(Q)
    print(f"Q-factor (method 1) for run {n + 1}: {Q}")

    """
    method 2:
        Q = |{period|theta > e^{-\pi}}|
    """
    Q = (data[n][0]['period'][data[n][0]['theta'] > data[n][0]['theta'].max()*np.exp(-np.pi/4)].max()+1) *4
    print(f"Q-factor (method 2) for run {n + 1}: {Q}")
    Q2.append(Q)

    ax1.plot(i[0]["time"], i[0]["theta"].max() * np.exp(-i[0]["time"] / tau), label=r"$\theta_{\mathrm{min, max}} e^{-t/\tau}$", color="black")
    ax1.plot(i[0]["time"], i[0]["theta"].min() * np.exp(-i[0]["time"] / tau), color="black")
    ax1.set_xlabel("Time (s)", fontproperties=gs_font)
    ax1.set_ylabel("Angle (rad)", fontproperties=gs_font)
    ax1.set_title(f"Angle vs Time for Run {n+1}", fontproperties=gs_font)
    ax1.legend()
    # with latex, display the function on the bottom right corner
    ax1.text(0.95, 0.01, f"$\\theta(t) = {params[n][0]:.2f}e^{{-t/{tau:.2f}}}\cos(2\pi t/{T:.2f} + {params[n][2]:.2f})$",
             verticalalignment='bottom', horizontalalignment='right',
                transform=ax1.transAxes,
                color='black', fontsize=13, fontproperties=gs_font)
    # plot the residuals as in period time
    ax2.plot(i[0]["time"], i[0]["theta"] - fit(i[0]["time"], *params[n]), label="Error", color="red")
    ax2.set_xlabel("Time (s)", fontproperties=gs_font)
    ax2.set_ylabel("Residuals", fontproperties=gs_font)
    ax2.set_title("Residuals",  fontproperties=gs_font)
    ax2.legend()

    ax2.axhline(y=0, color='grey', linestyle='--')
    plt.subplots_adjust(hspace=0.6)
    plt.savefig(f"runGraphs/run{n+1}.png")
    plt.show()

# print mean and uncertainty of Q-factor
Q1 = np.array(Q1)
print(f"Q-factor (method 1): {Q1.mean()} +/- {np.std(Q1)/np.sqrt(len(Q1))}")
Q2 = np.array(Q2)
print(f"Q-factor (method 2): {Q2.mean()} +/- {np.std(Q2)/np.sqrt(len(Q2))}")

