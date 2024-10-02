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
# Parameters
There are a few parameters you might need to adjust in the pendulum_tracker.py file:
- 'fps': the number of frames per second of the video
- 'color': the color of the pendulum in the video
- 'length': the length of the pendulum
- 'threshold': the threshold for the color detection
- 'file_path': the path to the video file

## Contributors
- [Hayson Cheung](https://github.com/HaysonC)
- [Eric Xie](https://github.com/Epic-Eric)

## Acknowledgements
This projects was developed under the course of Phy180 in the University of Toronto. Under the Division of Engineering Science, Faculty of Applied Science and Engineering.
This project was developed with the assistance of [GitHub Copilot](https://github.com/features/copilot).
