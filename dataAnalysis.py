import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def data_analysis(file, lengthString, update=False):
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
    origin = (min(df["y"]), df["x"].mean())



    df["theta"] = np.arctan((df["x"] - origin[1]) / (df["y"] - origin[0]))
    ratio = lengthString / (max(df["y"]) - min(df["y"]))
    df["x"] = (df["x"] - origin[1]) * ratio
    df["y"] = (df["y"] - origin[0]) * ratio
    origin = (0, 0)
    theta_smooth = savgol_filter(df["theta"], window_length=51, polyorder=3)
    peaks, _ = find_peaks(theta_smooth)
    antipeaks, _ = find_peaks(-theta_smooth)
    peak_times = df["time"][peaks]
    antipeak_times = df["time"][antipeaks]
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


    return df, periods, anti_periods

