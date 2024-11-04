
from dataAnalysis import data_analysis
import os
from expoFit import exponential_fit, plot_expo
from periodTime import plot_period_time, period_Time
from vision import video_capture

from periodVslength import *


def main():
    """
    pt = period_Time("runsArchive", 0.04, 0.157, 120)
    plot_period_time(*pt)
    """
    """ 
    data = [data_analysis(f"runsArchive/{file}", 0.157) for file in os.listdir("runsArchive") if file.endswith(".csv")]
    expo = exponential_fit(data)
    params = expo[0]
    plot_expo(data, params)
    """


    """
    lengths = [36, 61, 83, 102, 125, 147, 166, 192]
    for i in lengths:
        video_capture(f"andyRun/{i}.mp4", 0.9, fps=240 ,plotResults=True).to_csv(f"andyRun/{i}.csv")
     """

    lengths = [113, 135, 165, 185, 213, 239]
    data = [data_analysis(f"runs/{file}.csv", file, returnPeaks=True) for file in lengths]
    periodVLength(data, lengths, lengthUnits="mm", fps=60)


if __name__ == '__main__':
    main()
