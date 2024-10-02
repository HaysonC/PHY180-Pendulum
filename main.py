from dataAnalysis import data_analysis
import os
from expoFit import exponential_fit, plot_expo
from periodTime import Period_Time
from vision import video_capture



def main():
    pass
    """
    
    EXAMPLE USAGE:
    
    
    dir = "runs"
    
    
    data = [data_analysis(f"{dir}/{file}") for file in os.listdir(dir) if file.endswith(".csv")]
    expo = exponential_fit(data)
    plot_expo(data, *expo)
    # Plot the exponential fit data
    
    
    Period_Time = Period_Time("runs")
    plot_period_time(*Period_Time)
    # Plot the period time data
    
    
    video_capture("videos/run1.mp4", threshold=0.8, fps=30, color="ff0000")
    # Capture the video from the file
    """


if __name__ == '__main__':
    main()
