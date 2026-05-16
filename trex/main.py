import pyautogui
import mss
import time
import numpy as np
from collections import deque

time.sleep(3)

frame = 0
start_time = time.time()
c = 10
last_pos = None
pixels_per_frame = 0
speed_buffer = deque(maxlen=10)
dino_pos = None

base_width = 80

scan_left = 150
scan_right = 400
scan_height = 40

jump_coefficient = 2

def calibrate(dino_pos) -> map:
    print("Open https://chromedino.com/")
    print("Move cursor to DINO EYE and press Enter")
    input()
    dino_pos = pyautogui.position()
    print(f"Dino position: {dino_pos}")
    return dino_pos

x, y = calibrate(dino_pos)

with mss.mss() as sct:
    while True:
        frame += 1

        combined_top = y - c
        combined_height = (y + scan_height) - combined_top
        
        combined_zone = {
            "top": combined_top,
            "left": x + scan_left,
            "width": scan_right - scan_left,
            "height": combined_height
        }
        
        img = np.array(sct.grab(combined_zone))
        gray = img[:, :, :3].mean(axis=2)

        upper_gray = gray[:scan_height // 2 + 15, :]
        upper_dark_cols = np.where(upper_gray < 128)[1]

        lower_gray = gray[scan_height // 2 + 16:, :]
        lower_dark_cols = np.where(lower_gray < 128)[1]

        if len(lower_dark_cols) > 0:
    
            lower_left = lower_dark_cols.min()
            lower_right = lower_dark_cols.max()
            lower_width = lower_right - lower_left + 1

            if last_pos is not None:
                delta = last_pos - lower_left
                if delta > 0:
                    speed_buffer.append(delta)
                    pixels_per_frame = max(1, int(np.mean(speed_buffer)))

            last_pos = lower_left

            if pixels_per_frame > 0:
                jump_distance = base_width - lower_width + pixels_per_frame * jump_coefficient
                if lower_left <= jump_distance:
                    pyautogui.hotkey("up")
                    time.sleep(lower_width / pixels_per_frame * 0.035)
                    pyautogui.hotkey("down")
                    
        elif len(upper_dark_cols) > 0:
            
            upper_left = upper_dark_cols.min()
            upper_right = upper_dark_cols.max()
            upper_width = upper_right - upper_left + 1

            if last_pos is not None:
                delta = last_pos - upper_left
                if delta > 0:
                    speed_buffer.append(delta)
                    pixels_per_frame = max(1, int(np.mean(speed_buffer)))

            last_pos = upper_left

            if pixels_per_frame > 0:
                duck_distance = base_width - upper_width + pixels_per_frame * 2
                if upper_left <= duck_distance:
                    pyautogui.keyDown("down")
                    time.sleep(upper_width / pixels_per_frame * 0.06)
                    pyautogui.keyUp("down")

        if frame % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame / elapsed
            print(f"Frame {frame}, FPS: {fps:.1f}, speed={pixels_per_frame}")
