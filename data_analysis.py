import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
from scipy.signal import find_peaks, savgol_filter


def data_analysis(file):

    # Read the data from the CSV file
    df = pd.read_csv(file)
    # Print the first few rows of the data

    start_from = "left"

    # The shape of a sine graph, we need to take the origin of the pendulum, which is (miny, mean x)
    # The length of the pendulum is the distance between the origin and the bob. We need to convert pixel to real length
    origin = (min(df["y"]) if start_from == "left" else max(df["y"]), df["x"].mean())

    df["theta"] = np.arctan((df["x"] - origin[1]) / (df["y"] - origin[0])) * 180 / np.pi

    lengthString = 0.137

    ratio = lengthString / (max(df["y"]) - min(df["y"]))
    df["x"] = (df["x"] - origin[1]) * ratio
    df["y"] = (df["y"] - origin[0]) * ratio
    origin = (0, 0)

    # plot the first 10 seconds of the data
    plt.plot(df["time"], df["theta"])
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Angle vs Time")
    plt.show()

    theta_smooth = savgol_filter(df["theta"], window_length=51, polyorder=3)

    # Find peaks in the smoothed data
    peaks, _ = find_peaks(theta_smooth)
    antipeaks, _ = find_peaks(-theta_smooth)

    peak_times = df["time"][peaks]
    antipeak_times = df["time"][antipeaks]

    # Calculate the period
    periods = list(np.diff(peak_times))
    period = np.mean(periods)
    # delete abnormal period, tolerance is 50% of the average period, less or more than that is considered abnormal
    periods = [p for p in periods if abs(p - period) < period * 0.3]

    anti_periods = list(np.diff(antipeak_times))
    anti_period = np.mean(anti_periods)
    anti_periods = [p for p in anti_periods if abs(p - anti_period) < anti_period * 0.3]

    # get the average period

    print("Period:", period)
    # add a period column to the dataframe
    df["period"] = np.zeros(len(df))
    df["anti period"] = np.zeros(len(df))
    # assign period number to each section of the data
    for i, t in enumerate(peak_times[1:]):
        # find the index of the data that is less than the current peak time
        index = df[df["time"] < t].index[-1]
        # assign the period number to the data
        df.loc[index:, "period"] = i + 1

    for i, t in enumerate(antipeak_times[1:]):
        # find the index of the data that is less than the current peak time
        index = df[df["time"] < t].index[-1]
        # assign the period number to the data
        df.loc[index:, "anti period"] = i + 1

    return df, periods, anti_periods


if __name__ == "__main__":
    data_analysis("andyRun2.csv")
