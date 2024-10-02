import os
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np

gs_font = plt.matplotlib.font_manager.FontProperties(fname='/System/Library/Fonts/Supplemental/GillSans.ttc')



def period_Time(data: str | list, bin_size:int=0.1, lengthString=1) -> Tuple[pd.DataFrame, float, float, float, float, float]:
    """
    Analyze the data from the directory and plot the period vs time graph

    :param data: the directory containing the data files, or a list of dataframes, periods, and anti_periods
    :param bin_size: the bin size for the angle
    :param lengthString: the length of the string
    :return: a tuple containing the data frame containing the period vs time data, the fit parameters, the sigma
    

    example usage:
    Period_Time("runs")
    """
    Period_Time: pd.DataFrame
    a: float
    b: float
    c: float
    sigma: float
    xErr: float
    
    if isinstance(data, str):
        from dataAnalysis import data_analysis
        data = [data_analysis(f"{data}/{file}", lengthString) for file in os.listdir(data) if file.endswith(".csv")]
    elif isinstance(data, list):
        data = data
    else:
        raise ValueError("The data should be a string or a list")

    period_time_data = []
    for df, periods, anti_periods in data:
        max_angles = {i: df["theta"][df["period"] == i].max() for i in range(1, len(periods))}
        min_angles = {i: df["theta"][df["anti period"] == i].min() for i in range(1, len(anti_periods))}
        for i in max_angles.keys():
            if i not in [1, len(max_angles), 2, len(max_angles) - 1]:
                period_time_data.append({"Angle": max_angles[i], "Period Time": periods[i]})
        for i in min_angles.keys():
            if i not in [1, len(min_angles), 2, len(min_angles) - 1]:
                period_time_data.append({"Angle": min_angles[i], "Period Time": anti_periods[i]})
    print(f"Total period data: {len(period_time_data)}")
    Period_Time = pd.DataFrame(period_time_data)
    Period_Time["Angle"] = Period_Time["Angle"]


    # Convert back to float for calculations
    Period_Time["Angle"] = Period_Time["Angle"].astype(float)
    Period_Time["Angle"] = Period_Time["Angle"]
    min_value = Period_Time["Angle"].min()
    max_value = Period_Time["Angle"].max()
    bin_edges = np.arange(min_value, max_value + bin_size, bin_size)
    Period_Time["Angle"] = pd.cut(Period_Time["Angle"], bins=bin_edges)
    # for each angle series, just xErr would be the bin size/2, and yErr would be the standard deviation of the period time.
    # At the end but them into a one to one df with the angle being the mean of that angle series and
    # The period time being the mean the entries in that angle series
    Period_Time = Period_Time.groupby("Angle",observed=False).agg({"Period Time": ["mean", "std", "count"]}).reset_index()
    Period_Time.columns = ["Angle", "Period Time", "Period Time std", "Count"]
    Period_Time["Angle"] = Period_Time["Angle"].apply(lambda x: x.mid)
    Period_Time["Period Time"] = Period_Time["Period Time"]
    Period_Time["Period Time std"] = Period_Time["Period Time std"]
    yErr = Period_Time["Period Time std"]/np.sqrt(Period_Time["Count"])
    # add the yErr to the data frame
    Period_Time["yErr"] = yErr
    xErr = bin_size/2
    # drop Nan
    Period_Time = Period_Time.dropna()
    # put them in float
    Period_Time["Angle"] = Period_Time["Angle"].astype(float)
    Period_Time["Period Time"] = Period_Time["Period Time"].astype(float)


    # Fit a polynomial to the data
    model = np.poly1d(np.polyfit(Period_Time["Angle"], Period_Time["Period Time"], 2))
    a, b, c = model[2], model[1], model[0]

    def fit_curve(t):
        return a * t ** 2 + b * t + c

    # Calculate sigma
    sigma = np.sqrt(
        np.sum((Period_Time["Period Time"] - fit_curve(Period_Time["Angle"])) ** 2)
        / (len(Period_Time) - 3)
    )
    # Calculate r_value
    r_value = np.corrcoef(Period_Time["Period Time"], fit_curve(Period_Time["Angle"]))[0, 1]
    # Print results
    print("sigma:", sigma)
    print("r^2:", r_value ** 2)
    print("a:", a)
    print("b:", b)
    print("c:", c)
    # print equation in latex, when it is negative, it will be in the form of -x instead of + - x
    print(f"y = {a}x^2{f'-{-b}' if b < 0 else f'+{b}'}x{f'-{-c}' if c < 0 else f'+{c}'}")
    print("Average period time:", Period_Time["Period Time"].mean())
    print("Predicted period time at 0 rad:", 2 * np.pi * np.sqrt(lengthString / 9.81))
    return Period_Time, a, b, c, sigma, xErr


def plot_period_time(Period_Time: pd.DataFrame, a: float, b: float, c: float, sigma: float, xErr:float) -> None:
    """
    Plot the period vs time graph

    :param Period_Time: the data frame containing the period vs time data
    :param a: the x^2 coefficient of the fit
    :param b: the x coefficient of the fit
    :param c: the constant of the fit
    :param sigma: the standard deviation of the residuals
    :param xErr: the x error
    :return: None

    example usage:
    Period_Time = Period_Time("runs")
    plot_period_time(*Period_Time)
    """

    def fit_curve(t):
        return a * t ** 2 + b * t + c
    # Create plots
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
        yerr=Period_Time["yErr"],
        xerr=xErr,
        fmt="none",
        label="Error Bar",
        capsize=3,
        capthick=1,
        color="red",
    )

    ax1.set_xlabel("Angle (rad)")
    ax1.set_ylabel("Period Time (s)")
    ax1.legend()
    ax1.set_title("Period Time vs Angle")

    # Set the y-axis limits such that the max is on top with a little bit of room to spare, same with bottom
    yRangeSize = Period_Time["Period Time"].max() - Period_Time["Period Time"].min()
    yRangeSize *= 0.3
    ax1.set_ylim(Period_Time["Period Time"].min()-yRangeSize-min(Period_Time['yErr']), Period_Time["Period Time"].max()+yRangeSize+max(Period_Time['yErr']))
    plt.subplots_adjust(hspace=0.5)

    # Calculate ±2 standard deviation values for residuals


    # Plot residuals
    ax2.scatter(
        Period_Time["Angle"],
        Period_Time["Period Time"] - fit_curve(Period_Time["Angle"]),
        label="Error",
        s=5,
        color="red",
    )
    ax2.axhline(y=0, color="black", linestyle="-")  # Residual = 0 line

    # Calculate sigma for ±σ and ±2σ lines
    upper_sigma = sigma
    lower_sigma = -sigma
    upper_2sigma = 2 * sigma
    lower_2sigma = -2 * sigma

    # Plot ±σ and ±2σ lines without labels
    ax2.axhline(y=upper_sigma, color="grey", linestyle="--", alpha=0.5)
    ax2.axhline(y=lower_sigma, color="grey", linestyle="--", alpha=0.5)
    ax2.axhline(y=upper_2sigma, color="grey", linestyle="--", alpha=0.5)
    ax2.axhline(y=lower_2sigma, color="grey", linestyle="--", alpha=0.5)

    # Add text annotations for ±σ and ±2σ
    sigma_x = 1.77
    ax2.text(sigma_x, upper_sigma, r"$+\sigma$", ha="right", color="grey")
    ax2.text(sigma_x, lower_sigma, r"$-\sigma$", ha="right", color="grey")
    ax2.text(sigma_x, upper_2sigma, r"$+2\sigma$", ha="right", color="grey")
    ax2.text(sigma_x, lower_2sigma, r"$-2\sigma$", ha="right", color="grey")

    # Make more dense y-axis levels
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.errorbar(
        Period_Time["Angle"],
        Period_Time["Period Time"] - fit_curve(Period_Time["Angle"]),
        yerr=Period_Time["yErr"],
        xerr=xErr,
        fmt="none",
        color="red",
        capsize=3,
        capthick=1,
    )

    # Set labels
    ax2.set_xlabel("Angle (rad)")
    ax2.set_ylabel("Residual (s)")

    plt.savefig("graph1.png", dpi=300)
    plt.show()

