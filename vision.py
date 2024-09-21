import os
from random import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import ndarray

# Initialize video capture
file = 'videos/test9.mp4'
print("file exists?", os.path.exists(file))
cap = cv2.VideoCapture(file)


# color detection for the bob
color = "ff0000"
threshold = 0.70
color = np.array([int(color[i:i + 2], 16) for i in (0, 2, 4)])

color = tuple(color - 128 for color in color[::-1])
# circle detection not working

""" 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Circle detection
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRe
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Pendulum Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
# color detection, also use green to circle the part where it detects
# dataframe
df = pd.DataFrame(columns=['time', 'x', 'y', 'theta'])
time=0

while cap.isOpened():
    time += 1
    ret, frame = cap.read()
    if not ret:
        break
    # resise with the same aspect ratio
    aspect = frame.shape[1] / frame.shape[0]
    frame = cv2.resize(frame, (800, int(800 / aspect)))
    # preprocess
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    # for each element in the lab, if the sum of the three elements is greater than 255*2.8, set it to 0
    lab[lab.sum(axis=2) > 255 * 2.8] = 0
    # centre the color
    lab = lab - [128, 128, 128]
    # if the sum of the three elements more then 118*3 set it to 0
    lab[lab.sum(axis=2) > 100 * 3] = 0

    # calculate the dot product
    try:
        dot = -(np.dot(lab, color) / np.linalg.norm(color) / np.linalg.norm(lab, axis=2))
    except ValueError:
        continue

    # imshow dot product
    cv2.imshow('Pendulum Tracking', dot)


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
        cv2.putText(frame, f"{x},{y}", (x + w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"{w * h}", (x + w, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        theta = np.arctan2(y - frame.shape[0] // 2, x - frame.shape[1] // 2)
        df = df._append({'time': 0, 'x': x, 'y': y, 'theta': theta}, ignore_index=True)

    if total_area > 0:
        avg_center_x = weighted_sum_x / total_area
        avg_center_y = weighted_sum_y / total_area
        cv2.circle(frame, (int(avg_center_x), int(avg_center_y)), 5, (255, 0, 0), -1)
        print(f"Weighted Average Center: ({avg_center_x}, {avg_center_y})")
    try:
        if avg_center_x is not None and avg_center_y is not None:
            # record x and y on a dataframe
            df = df._append({'time': time, 'x': avg_center_x, 'y': avg_center_y, 'theta': 0}, ignore_index=True)
    except NameError:
        pass
    # 1/1000 chance to quit
    if random() < 0.001:
        break
    # Show the frame
    cv2.imshow('Pendulum Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# delte time = 0 row
df = df[df.time != 0]
# plot  the dataframe, time and x,scatter plot, smaller dots
plt.scatter(df.time, df.x, label='x', s=1)
plt.savefig('x.pdf')
plt.show()


cap.release()

cv2.destroyAllWindows()

