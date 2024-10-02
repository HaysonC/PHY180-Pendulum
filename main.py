from dataAnalysis import data_analysis
import os
from expoFit import exponential_fit, plot_expo
from periodTime import plot_period_time, period_Time
from vision import video_capture


def main():
    pass
    dir = "runs"
    data = [data_analysis(f"{dir}/{file}", 0.155) for file in os.listdir(dir) if file.endswith(".csv")]

    pt = period_Time(data, lengthString=0.155)
    plot_period_time(*pt)
    # Plot the period time data

    # Capture the video from the file


if __name__ == '__main__':
    main()
