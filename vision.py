import os

import cv2
import numpy as np

# Initialize video capture
file = 'pendulum_videos/test.mp4'
print("file exists?", os.path.exists(file))
cap = cv2.VideoCapture(file)


# color detection for the bob
color = "92dfd8"
color = np.array([int(color[i:i + 2], 16) for i in (0, 2, 4)])

color = tuple(color - 128 for color in color)

target_color_norm = color / np.linalg.norm(color)
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
                               param1=50, param2=30, minRadius=5, maxRadius=30)

    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Pendulum Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
# color detection, also use green to circle the part where it detects

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # all -128 to center the color space
    lab = lab - 128

    # put them into unit vectors in the color space
    lab_norm = lab / np.linalg.norm(lab, axis=2)[:, :, np.newaxis]

    # calculate the dot product
    dot = abs(np.sum(lab_norm * target_color_norm, axis=2))

    # imshow dot product


    # Threshold the dot product
    mask = (dot > 0.997).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 3000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Pendulum Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

cv2.destroyAllWindows()

