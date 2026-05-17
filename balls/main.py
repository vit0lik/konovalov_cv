import cv2
import numpy as np
import time
from math import dist
import json
from pathlib import Path
from random import shuffle
import itertools


cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)


position = [0, 0]
clicked = False
hsv = None


def on_click(event, x, y, flags, param):
    global position, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        position = [x, y]
        clicked = True


cv2.setMouseCallback("Image", on_click)
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: could not open camera")
    exit()

lower = None
upper = None
positions = []

prev_time = time.time()
curr_time = time.time()
d = 6.36

print("Press 'm' to switch between 3 and 4 ball modes. Press 'c' to reset selected colors. Press 'q' to quit.")
num_balls = 3
target_colors = []
shuffled_colors = []

while True:
    ret, frame = capture.read()

    if not ret or frame is None:
        print("Failed to get frame from camera")
        continue

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('m'):
        num_balls = 4 if num_balls == 3 else 3
        target_colors = []
        shuffled_colors = []

    if key == ord('c'):
        target_colors = []
        shuffled_colors = []

    if len(target_colors) < num_balls:
        cv2.putText(frame, f"Mode: {num_balls} balls. Click on color {len(target_colors) + 1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (225, 255, 0))
        if clicked:
            clicked = False
            color = hsv[position[1], position[0]]
            lower = np.clip(color * 0.9, 0, 255).astype("uint8")
            upper = np.clip(color * 1.1, 0, 255).astype("uint8")
            upper[1] = 255
            upper[2] = 255
            target_colors.append([lower, upper])

    if len(target_colors) == num_balls and len(shuffled_colors) == 0:
        shuffled_colors = target_colors.copy()
        shuffle(shuffled_colors)

    if len(shuffled_colors) == num_balls:
        detected_balls = []

        for color in shuffled_colors:
            lower, upper = color
            inr = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                if radius > 10:
                    cv2.circle(frame, (int(cx), int(cy)), int(radius), (52, 252, 0), 2)
                    detected_balls.append((int(cx), int(cy), color))

        if num_balls == 3:
            detected_balls.sort(key=lambda b: b[0])
        elif num_balls == 4:
            detected_balls.sort(key=lambda b: b[1])
            top_row = sorted(detected_balls[:2], key=lambda b: b[0])
            bottom_row = sorted(detected_balls[2:], key=lambda b: b[0])
            detected_balls = top_row + bottom_row

        flag = 0
        for i, (cx, cy, color) in enumerate(detected_balls):
            if np.array_equal(color[0], shuffled_colors[i][0]) and np.array_equal(color[1], shuffled_colors[i][1]):
                flag += 1

        if flag == num_balls:
            cv2.putText(frame, "guessed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0))
        else:
            cv2.putText(frame, f"{flag}/{num_balls} balls in correct position", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (225, 255, 0))

    cv2.imshow("Image", frame)

capture.release()
cv2.destroyAllWindows()
