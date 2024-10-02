# Pendulum Data Analysis

This project analyzes pendulum motion data from CSV files, fits an exponential function to the data, and plots the results. It also captures video data for further analysis.

**PLEASE CREDIT ME AND ERIC IN YOUR REPORTS THANKS :)** 

## Project Structure

- `main.py`: Main script to run the data analysis and plotting.
- `dataAnalysis.py`: Contains the `data_analysis` function to process the CSV data.
- `expoFit.py`: Contains functions for fitting exponential functions and plotting the results.
- `periodTime.py`: Contains the `Period_Time` class for analyzing period time data.
- `vision.py`: Contains the `video_capture` function for capturing video data.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/HaysonC/pendulum-data-analysis.git
    cd pendulum-data-analysis
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Data Analysis and Plotting

To analyze the data and plot the results, run the `main.py` script:

```sh
python main.py
```
## Preprocessing
The video is best to be free of similar color in the background, it should have minmum noise and shot 
in good white light. The pendulum should be a solid color and the color should be different from the background.

**It also requires the video file to be in mp4**

You can use iMovie to crop the video to the pendulum only, and export it as mp4
TIPS: While cropping the video, you could also adjust the white balance and exposure to make the pendulum color more distinct from the background.
## Parameters
There are a few parameters you might need to adjust in the pendulum_tracker.py file:
- 'fps': the number of frames per second of the video
- 'color': the color of the pendulum in the video
- 'lengthString': the length of the pendulum
- 'threshold': the threshold for the color detection
- 'dir': The directory of the video file and the csv file


## Methodology
- `video_capture` function in `vision.py` captures the video data and saves it as a CSV file.
  - The function uses OpenCV to capture the video data.
  - Computer the dot product of the pendulum color and the background color to detect the pendulum.
  - Uses contour detection to find the pendulum.
  - Saves the pendulum position data to a CSV file.
- `data_analysis` function in `dataAnalysis.py` processes the CSV data.
  - Reads the CSV file and processes the data.
  - Computes the period time of the pendulum via finding the peaks of the pendulum position data.
  - Returns the period time data.
  - Saves the period time data to a CSV file.
- `Period_Time` function in `periodTime.py` analyzes the period time data.
  - Reads the period time data and processes the data.
  - Fits an quadratic function to the period time data.
  - Plots the period time data and the fitted function.
- `expo_fit` function in `expoFit.py` fits an exponential function to the data.
  - Fits an exponential function to the data using `scipy.optimize.curve_fit`.
  - Plots the data and the fitted function.
## Contributors
- [Hayson Cheung](https://github.com/HaysonC)
- [Eric Xie](https://github.com/Epic-Eric)

## Acknowledgements
This projects was developed under the course of Phy180 in the University of Toronto. Under the Division of Engineering Science, Faculty of Applied Science and Engineering.
This project was developed with the assistance of [GitHub Copilot](https://github.com/features/copilot).
