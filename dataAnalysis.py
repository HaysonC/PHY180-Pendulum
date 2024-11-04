import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

import numpy as np
from scipy.optimize import least_squares


def find_pendulum_center(points, radius):
    """
    Finds the center of a pendulum's swing assuming symmetric motion and known radius.

    Parameters:
    points : list of tuples
        A list of (x, y) coordinates of tracked points on the pendulum's path.
    radius : float
        Known radius of the pendulum swing.

    Returns:
    tuple
        (cx, cy) coordinates of the pendulum swing center.
    """
    # Convert to numpy array for easy manipulation
    points = np.array(points)

    # Find the leftmost and rightmost points
    min_x_point = points[np.argmin(points[:, 0])]
    max_x_point = points[np.argmax(points[:, 0])]

    # Midpoint between the leftmost and rightmost points
    midpoint_x = (max_x_point[0]+min_x_point[0] ) / 2
    midpoint_y = (max_x_point[1]+min_x_point[1] ) / 2

    # Center of the swing is radius distance directly above the midpoint
    center_y = midpoint_y + np.sqrt(radius ** 2 - (max_x_point[0] - midpoint_x)**2)

    return (midpoint_x, center_y)



def data_analysis(file, lengthString, update=False, returnPeaks=False):
    """
    Analyze the data from the file. It returns the data frame, the periods, and the anti-periods


    This script is used to analyze the data from the file. It returns the data frame, the periods, and the anti-periods

    It returns the data frame, the periods, and the anti-periods (the distance between the peaks and the troughs respectively)
    The data frame contains the following columns:
    - time: the time of the data point
    - x: the x coordinate of the data point (real length)
    - y: the y coordinate of the data point (real length)
    - theta: the angle of the pendulum
    - period: the period number of the data point
    - anti period: the anti period number of the data point
    Analyze the data from the file

    :param file: the file to analyze, the file should be a csv file containing the data
    :param lengthString: the length of the string
    :param update: whether to update the file


    :return: A tuple containing the data frame, the periods, and the anti-periods (the distance between the peaks and the troughs respectively)
    """
    df = pd.read_csv(file)
    points = []
    for i in df.values:
        points.append([i[2], i[3]])
    origin = find_pendulum_center(points, lengthString)
    print(origin)
    df["theta"] = np.arctan((df["x"] - origin[0]) / (df["y"] - origin[1]))
    ratio = lengthString / (origin[1] - min(df["y"]))
    df["x"] = (df["x"] - origin[0]) * ratio
    df["y"] = (df["y"] - origin[1]) * ratio


    theta_smooth = savgol_filter(df["theta"], window_length=51, polyorder=3)
    peaks, _ = find_peaks(theta_smooth)
    antipeaks, _ = find_peaks(-theta_smooth)
    peak_times = df["time"][peaks]
    antipeak_times = df["time"][antipeaks]

    peak_values = df["theta"].iloc[peaks]
    antipeak_values = df["theta"].iloc[antipeaks]

    # plot

    periods = list(np.diff(peak_times))
    period = np.mean(periods)
    periods = [p for p in periods if abs(p - period) < period * 0.3]
    anti_periods = list(np.diff(antipeak_times))
    anti_period = np.mean(anti_periods)
    anti_periods = [p for p in anti_periods if abs(p - anti_period) < anti_period * 0.3]
    df["period"] = np.zeros(len(df))
    df["anti period"] = np.zeros(len(df))
    for i, t in enumerate(peak_times[1:]):
        index = df[df["time"] < t].index[-1]
        df.loc[index:, "period"] = i + 1
    for i, t in enumerate(antipeak_times[1:]):
        index = df[df["time"] < t].index[-1]
        df.loc[index:, "anti period"] = i + 1

    if update:
        df.to_csv(file, index=False)
    # plot theta vs time
    plt.plot(df["time"], df["theta"])
    plt.show()
    if returnPeaks:
        return df, periods, anti_periods, peak_times, antipeak_times, peak_values, antipeak_values
    else:
        return df, periods, anti_periods