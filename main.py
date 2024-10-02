from dataAnalysis import data_analysis
import os
from expoFit import exponential_fit, plot_expo
from periodTime import plot_period_time, period_Time
from vision import video_capture


def main():
    # expo plot
    """
    dir = "runs"
    data = [data_analysis(f"{dir}/{file}",0.155) for file in os.listdir(dir) if file.endswith(".csv")]
    expo = exponential_fit(data)
    plot_expo(data, expo)
    """
    df = video_capture("videos/run1.mp4", threshold=0.8, color="ff0000", plotResults=True)
    df.to_csv("test.csv")


if __name__ == '__main__':
    main()
