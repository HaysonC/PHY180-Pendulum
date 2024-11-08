import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import matplotlib.font_manager as fm
from scipy.stats import alpha

import dataAnalysis
def round_to_sigfigs(x, n):
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))

gs_font = fm.FontProperties(fname='/System/Library/Fonts/Supplemental/GillSans.ttc')
# Define the period function
def period(t, a, b, c, tau):
    return a* np.exp(-4*t / tau) + b * np.exp(-2*t / tau) + c


# Define the fit function
def fit(t, theta_not, tau, phi, a, b, c):
    return theta_not * np.exp(-t / tau) * np.cos(2 * np.pi * t / (period(t, a, b, c, tau)) + phi)


def exponential_fit(data, lengths=None):
    """
    Fit an exponential function to the data and return the fit parameters, tau, and T.
    :param data: tuple containing the data frame, the periods, and the anti-periods, comes from data_analysis
    :return: The fit parameters: theta_not, tau, phi, T
    """
    params = []
    covs = []
    for j,i in enumerate(data):
        x = i[0]["time"]
        y = i[0]["theta"]
        theta_not = (y.max() - y.min()) / 2
        a = [theta_not]
        try:
            ppt, pcov = curve_fit(lambda t, tau, phi, a ,b, c: fit(t, theta_not, tau, phi, a, b, c), x, y, maxfev=10000)
        except RuntimeError:
            print(f"Run {j+1} failed to converge")
            continue
        a.extend(ppt)
        covs.append(pcov)
        params.append(a)
    return params, covs


def plot_expo(data, params)-> None:
    """
    Plot the data and the fit for each run and calculate the Q-factor using two different methods.
    :param data: dataframes of time and theta values, comes from data_analysis
    :param params: fit parameters, comes from exponential_fit
    :param tau: decay constant, comes from exponential_fit
    :param T: period, comes from exponential_fit
    :return: None

    Q value calculation is based on the following formula:
    Q = pi * tau / T
            or
    Q = (period[theta > theta_max * exp(-pi/4)].max() + 1) * 4


    example usage:
    data = [data_analysis(f"{dir}/{file}") for file in os.listdir(dir) if file.endswith(".csv")]
    expo = exponential_fit(data)[0]
    plot_expo(data, *expo)
    """
    Q1 = []
    Q2 = []
    for n in range(len(data)):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.4, 1]})
        i = data[n]
        time = np.linspace(0, i[0]["time"].max(), 1000)
        ax1.scatter(i[0]["time"], i[0]["theta"], label="Data", color="blue", alpha=1, s=0.1)
        ax1.plot(time, fit(time, *params[n]), label="Fit", color="orange", alpha=0.6)
        ax1.set_yticks(np.linspace(-np.pi / 2, np.pi / 2, 9))
        ax1.set_yticklabels(["$-\pi/2$", "$-3\pi/8$", "$-\pi/4$", "$-\pi/8$", "0", "$\pi/8$", "$\pi/4$", "$3\pi/8$", "$\pi/2$"])
        ax1.axhline(y=0, color='grey', linestyle='--')
        # T would be the average of the period provided by the period function so \int T(t) dt/t
        tau = params[n][1]
        T = np.trapz(period(i[0]["time"], params[n][3], params[n][4], params[n][5], tau), i[0]["time"]) / i[0]["time"].max()
        Q = np.pi * tau / T
        Q1.append(Q)

        Q = (data[n][0]['period'][data[n][0]['theta'] > data[n][0]['theta'].max() * np.exp(-np.pi / 4)].max() + 1) * 4
        Q2.append(Q)
        ax1.plot(i[0]["time"], i[0]["theta"].max() * np.exp(-i[0]["time"] / tau), label=r"$\theta_{\mathrm{min, max}} e^{-t/\tau}$", color="black")
        ax1.plot(i[0]["time"], i[0]["theta"].min() * np.exp(-i[0]["time"] / tau), color="black")
        ax1.set_xlabel("Time (s)", fontproperties=gs_font)
        ax1.set_ylabel("Angle (rad)", fontproperties=gs_font)
        ax1.set_title(f"Angle vs Time for Run {n+1}", fontproperties=gs_font)
        ax1.legend(markerscale=20)
        # use latex in the text
        ax1.text(0.95, 0.01, fr"$\theta(t)={params[n][0]:.2f}e^{{-t/{params[n][1]:.2f}}}\cos\left(\frac{{2\pi t}}{{T}}+{params[n][2]:.2f}\right)$", ha='right', va='bottom', transform=ax1.transAxes, fontproperties=gs_font)
        ax1.text(0.95, 0.12, fr"$T={round(params[n][4],4)}\theta^4 + {round(params[n][3],3)}\theta^2 + {round(params[n][2],3)}$", ha='right', va='bottom', transform=ax1.transAxes, fontproperties=gs_font)
        ax2.scatter(i[0]["time"], i[0]["theta"] - fit(i[0]["time"], *params[n]), label="Error", color="red", s=0.1)
        ax2.set_xlabel("Time (s)", fontproperties=gs_font)
        ax2.set_ylabel("Residuals", fontproperties=gs_font)
        ax2.set_title("Residuals", fontproperties=gs_font)
        ax2.legend()
        ax2.axhline(y=0, color='grey', linestyle='--')
        # calcuate r squared
        residuals = i[0]["theta"] - fit(i[0]["time"], *params[n])
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((i[0]["theta"] - i[0]["theta"].mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"Run {n+1} R^2: {r_squared}")
        plt.subplots_adjust(hspace=0.6)
        plt.savefig(f"runGraphs/run{n+1}.png")
        plt.show()
    Q1 = np.array(Q1)
    Q2 = np.array(Q2)
    print(f"Q-factor (method 1): {Q1.mean()} +/- {np.std(Q1) / np.sqrt(len(Q1))}")
    print(f"Q-factor (method 2): {Q2.mean()} +/- {np.std(Q2) / np.sqrt(len(Q2))}")

    return None
