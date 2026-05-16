import pyautogui
import mss
import time
import numpy as np

time.sleep(3)

frame = 0
start_time = time.time()
c = 10
last_pos = None
skip_frames = 0
obstacles = []
pixels_per_frame = 0
speed = np.array([])

def calibrate():
    global DINO_POS
    print("Open https://chromedino.com/")
    print("Move cursor to DINO EYE and press Enter")
    input()
    DINO_POS = pyautogui.position()
    print(f"Dino position: {DINO_POS}")
    
def find_obstacle_width(zone):
    img = np.array(sct.grab(scan_zone))
    gray = img[:, :, :3].mean(axis=2)
    left_col = np.where(gray < 128)[1].min()
    right_col = np.where(gray < 128)[1].max()
    obstacle_width = right_col - left_col + 1
    return obstacle_width
calibrate()

with mss.mss() as sct:
    while True:
        frame += 1
        x, y = DINO_POS
            
        scan_zone = {"top": y - c, "left": x + 350, "width": 80, "height": 6 * c}
        img = np.array(sct.grab(scan_zone))
        gray = img[:, :, :3].mean(axis=2)
        if skip_frames > 0:
            skip_frames -= 1
            
        if (gray < 128).any():
            left_col = np.where(gray < 128)[1].min()
            if last_pos and ( last_pos - left_col) > 0:
                speed = np.append(speed, last_pos - left_col)
                pixels_per_frame = int(speed.mean())
                if len(speed) > 10:
                    speed = np.delete(speed, 0)
            last_pos = left_col
        
        scan_zone = {"top": y - c, "left": x + 300, "width": 30, "height": 6 * c}
        img = np.array(sct.grab(scan_zone))
        gray = img[:, :, :3].mean(axis=2)
        if (gray < 128).any() and skip_frames == 0 and pixels_per_frame:
            obstacle_zone = {"top": y - c, "left": x + 300, "width": 160, "height": 6 * c}
            if pixels_per_frame > 0:
                img1 = np.array(sct.grab(obstacle_zone))
                gray = img1[:, :, :3].mean(axis=2)
                cols = np.where(gray < 128)[1]
                if len(cols) > 0:
                    obstacle_width = cols.max() - cols.min() + 1
                    obstacles.append(obstacle_width)
                    print(frame, obstacle_width, pixels_per_frame)
                    skip_frames = obstacle_width // pixels_per_frame
                
        if obstacles and pixels_per_frame is not None:        
            monitor = {"top": y-c, "left": x + 200 + pixels_per_frame - obstacles[0], "width": 2*c, "height": 4*c}
            img = np.array(sct.grab(monitor))
            
            if (img[:, :, :3].mean(axis=2) < 128).any():
                pyautogui.hotkey("up")
                obstacles.pop(0)
        
        if frame % 500 == 0:
            elapsed = time.time() - start_time
            fps = frame / elapsed
            
            print(f"Frame {frame}, FPS: {fps:.1f}")
