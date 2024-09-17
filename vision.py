import os

import cv2
import numpy as np

# Initialize video capture
file = 'pendulum_videos/250mm1trimmed.mp4'
print("file exists?", os.path.exists(file))
cap = cv2.VideoCapture(file)


# color detection for the bob
lower_bound = (70, 89, 79)
upper_bound = (140, 145, 130)

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

    mask = cv2.inRange(frame, lower_bound, upper_bound)
    cnt = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0] if len(cnt) == 2 else cnt[1]

    for c in cnt:
        area = cv2.contourArea(c)
        if area > 10 and area < 70:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (36, 255, 12), 2)
    cv2.imshow('Pendulum Tracking', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

