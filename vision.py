import os
from random import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import ndarray


# Initialize video capture
file = "videos/Pendulum 3.mp4"
print("file exists?", os.path.exists(file))
cap = cv2.VideoCapture(file)
fps = 120
# The first frame have the user to draw a line and specify the distance and the centre of the pendulum, resize before operation
ret, frame = cap.read()
aspect = frame.shape[1] / frame.shape[0]
frame = cv2.resize(frame, (800, int(800 / aspect)))
""" 
# Draw a line
line = None
def draw_line(event, x, y, flags, param):
    nonlocal line
    if event == cv2.EVENT_LBUTTONDOWN:
        if line is None:
            line = [(x, y)]
        else:
            line.append((x, y))
            cv2.line(frame, line[0], line[1], (0, 255, 0), 2)
            cv2.imshow('Pendulum Tracking', frame)

"""

# color detection for the bob
color = "ff0000"
threshold = 0.9
color = np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)])

color = tuple(color - 128 for color in color[::-1])

norm_color = np.linalg.norm(color)
# color detection, also use green to circle the part where it detects
# dataframe
df = pd.DataFrame(columns=["time", "x", "y", "theta"])
time = 0

while cap.isOpened():
    time += 1
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (800, int(800 / aspect)))
    s = frame - np.array([128, 128, 128])

    norm_s = np.linalg.norm(s, axis=2)
    dot = np.sum(s * color, axis=2) / (norm_s * norm_color)

    # imshow dot product

    # costheta > threshold
    mask = np.zeros_like(dot, dtype=np.uint8)
    mask[dot > threshold] = 255

    x = []
    y = []

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0
    weighted_sum_x = 0
    weighted_sum_y = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        center_x = x + w / 2
        center_y = y + h / 2

        # Accumulate weighted sums
        total_area += area
        weighted_sum_x += center_x * area
        weighted_sum_y += center_y * area

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{x},{y}", (x + w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"{w * h}",
            (x + w, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        theta = np.arctan2(y - frame.shape[0] // 2, x - frame.shape[1] // 2)
        df = df._append({"time": 0, "x": x, "y": y, "theta": theta}, ignore_index=True)

    if total_area > 0:
        avg_center_x = weighted_sum_x / total_area
        avg_center_y = weighted_sum_y / total_area
        cv2.circle(frame, (int(avg_center_x), int(avg_center_y)), 5, (255, 0, 0), -1)
    try:
        if avg_center_x is not None and avg_center_y is not None:
            # record x and y on a dataframe
            df = df._append(
                {"time": time / fps, "x": avg_center_x, "y": avg_center_y, "theta": 0},
                ignore_index=True,
            )
    except NameError:
        pass
    # Show the frame
    cv2.imshow("Pendulum Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# delte time = 0 row
df = df[df.time != 0]

# save the df as a csv file
df.to_csv("pendulum.csv", index=False)
# save t
# plot the dataframe, time and x,scatter plot, smaller dots
plt.scatter(df.time, df.y, label="y", s=1)
plt.savefig("y.pdf")
plt.show()


cap.release()

cv2.destroyAllWindows()
