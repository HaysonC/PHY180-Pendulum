import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def video_capture(file: str, threshold: float, fps: int = 60, color: str = "ff0000",
                  plotResults: bool = False) -> pd.DataFrame:
    """
    Capture the video from the file
    :param file: the file to capture
    :param threshold: the threshold of the dot product. 0 < threshold < 1( recommended 0.9)
    :param plotResults: whether to plot the results
    :param fps: the frame per second
    :param color: the color of the target
    :return: the panda dataframe containing the raw data: pixel x pixel y, time
    """
    print("file exists?", os.path.exists(file))
    cap = cv2.VideoCapture(file)
    # The first frame have the user to draw a line and specify the distance and the centre of the pendulum, resize before operation
    ret, frame = cap.read()
    aspect = frame.shape[1] / frame.shape[0]
    frame = cv2.resize(frame, (800, int(800 / aspect)))
    color = np.array([int(color[i: i + 2], 16) for i in (0, 2, 4)])
    color = tuple(color - 128 for color in color[::-1])

    norm_color = np.linalg.norm(color)
    # color detection, also use green to circle the part where it detects
    df = pd.DataFrame(columns=["time", "x", "y", "theta"])
    time = 0

    while cap.isOpened():
        time += 1
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (400, int(400 / aspect)))
        s = frame.copy()
        s[:, :, 2] = s[:, :, 2] * 1.6
        s[:, :, 1] = s[:, :, 1] * 0.8
        s[:, :, 0] = s[:, :, 0] * 1.8
        s = s - np.array([128, 128, 128])
        # adjust white balance to cool down the color

        norm_s = np.linalg.norm(s, axis=2)
        dot = np.sum(s * color, axis=2) / (norm_s * norm_color)

        # imshow dot product

        # costheta > threshold
        mask = np.zeros_like(dot, dtype=np.uint8)
        mask[dot > threshold] = 255

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

        if total_area > 10:
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
    df = df[df.time != 0]

    if plotResults:
        # plot the dataframe, time and x,scatter plot, smaller dots
        plt.scatter(df.time, df.x, label="x", s=1)
        plt.savefig("x.pdf")
        plt.show()

    cap.release()

    cv2.destroyAllWindows()
    return df
