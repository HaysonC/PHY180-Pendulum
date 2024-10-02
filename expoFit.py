import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.font_manager as fm
import dataAnalysis


gs_font = fm.FontProperties(fname='/System/Library/Fonts/Supplemental/GillSans.ttc')


def exponential_fit(data):
    """
    Fit an exponential function to the data and return the fit parameters, tau, and T.

    :param: data: tuple containing the data frame, the periods, and the anti-periods, comes from data_analysis

    :returntuple: A tuple containing the fit parameters, tau, and T.
    """
    fit = lambda t, theta_not, tau, phi, T: theta_not * np.exp(-t / tau) * np.cos(2 * np.pi * t / T + phi)
    params = []
    for i in data:
        x = i[0]["time"]
        y = i[0]["theta"]
        theta_not = (y.max() - y.min()) / 2
        a = [theta_not]
        a.extend(curve_fit(lambda t, tau, phi, T: fit(t, theta_not, tau, phi, T), x, y)[0])
        params.append(a)
    params = np.array(params)
    tau = np.mean(params[:, 1])
    T = np.mean(params[:, 3])
    return params, tau, T


def plot_expo(data, params, tau, T):
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
    expo = exponential_fit(data)
    plot_expo(data, *expo)
    """
    Q1 = []
    Q2 = []
    fit = lambda t, theta_not, tau, phi, T: theta_not * np.exp(-t / tau) * np.cos(2 * np.pi * t / T + phi)
    for n in range(len(data)):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.4, 1]})
        i = data[n]
        time = np.linspace(0, i[0]["time"].max(), 1000)
        ax1.plot(i[0]["time"], i[0]["theta"], label="Data", color="blue", alpha=0.5)
        ax1.plot(time, fit(time, *params[n]), label="Fit", color="orange")
        ax1.set_yticks(np.linspace(-np.pi / 2, np.pi / 2, 9))
        ax1.set_yticklabels(["$-\pi/2$", "$-3\pi/8$", "$-\pi/4$", "$-\pi/8$", "0", "$\pi/8$", "$\pi/4$", "$3\pi/8$", "$\pi/2$"])
        ax1.axhline(y=0, color='grey', linestyle='--')
        tau = params[n][1]
        T = params[n][3]
        Q = np.pi * tau / T
        Q1.append(Q)
        Q = (data[n][0]['period'][data[n][0]['theta'] > data[n][0]['theta'].max() * np.exp(-np.pi / 4)].max() + 1) * 4
        Q2.append(Q)
        ax1.plot(i[0]["time"], i[0]["theta"].max() * np.exp(-i[0]["time"] / tau), label=r"$\theta_{\mathrm{min, max}} e^{-t/\tau}$", color="black")
        ax1.plot(i[0]["time"], i[0]["theta"].min() * np.exp(-i[0]["time"] / tau), color="black")
        ax1.set_xlabel("Time (s)", fontproperties=gs_font)
        ax1.set_ylabel("Angle (rad)", fontproperties=gs_font)
        ax1.set_title(f"Angle vs Time for Run {n+1}", fontproperties=gs_font)
        ax1.legend()
        ax1.text(0.95, 0.01, f"$\\theta(t) = {params[n][0]:.2f}e^{{-t/{tau:.2f}}}\cos(2\pi t/{T:.2f} + {params[n][2]:.2f})$",
                 verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes, color='black', fontsize=13, fontproperties=gs_font)
        ax2.plot(i[0]["time"], i[0]["theta"] - fit(i[0]["time"], *params[n]), label="Error", color="red")
        ax2.set_xlabel("Time (s)", fontproperties=gs_font)
        ax2.set_ylabel("Residuals", fontproperties=gs_font)
        ax2.set_title("Residuals", fontproperties=gs_font)
        ax2.legend()
        ax2.axhline(y=0, color='grey', linestyle='--')
        plt.subplots_adjust(hspace=0.6)
        plt.savefig(f"runGraphs/run{n+1}.png")
        plt.show()
    Q1 = np.array(Q1)
    Q2 = np.array(Q2)
    print(f"Q-factor (method 1): {Q1.mean()} +/- {np.std(Q1) / np.sqrt(len(Q1))}")
    print(f"Q-factor (method 2): {Q2.mean()} +/- {np.std(Q2) / np.sqrt(len(Q2))}")
