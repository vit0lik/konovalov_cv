import matplotlib.pyplot as plt
import numpy as np 
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.color import rgb2hsv

def get_key(value, keys, delta) -> float:
    for key in keys:
        if abs(key - value) < delta:
            return key
    return value
        
if __name__ == "__main__":
    
    image = imread("data/balls_and_rects.png")
    hsv = rgb2hsv(image)
    h = hsv[:, :, 0]

    binary = np.mean(image, axis=2) > 0
    labeled = label(binary)
    props = regionprops(labeled, intensity_image=h)

    delta = 0.05

    rect_colors = {}
    balls_colors = {}

    for prop in props:
        is_rect = prop.image.sum() / prop.image.size == 1
        color = prop.intensity_mean
        
        if is_rect:
            key = get_key(color, rect_colors.keys(), delta)
            rect_colors[key] = rect_colors.get(key, 0) + 1
        
        else:
            key = get_key(color, balls_colors.keys(), delta)
            balls_colors[key] = balls_colors.get(key, 0) + 1

    print(f"Всего фигур: {sum(rect_colors.values()) + sum(balls_colors.values())}")
    print(f"Всего прямоугольков: {sum(rect_colors.values())}")
    print("Цвета:")
    for color, count in rect_colors.items():
        print(color, count)
    print(f"Всего шаров: {sum(balls_colors.values())}")
    print("Цвета: ")
    for color, count in balls_colors.items():
        print(color, count)

    plt.subplot(121)
    plt.imshow(image[:, :, 0], cmap="gray")
    plt.subplot(122)
    plt.plot(np.unique(h), "o")
    plt.show()