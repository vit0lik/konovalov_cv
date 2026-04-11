import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path\

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    return np.max(label(new_image)) - 1

def count_lines(region):
    shape = region.image.shape
    image = region.image
    v_lines = (np.sum(image, 0) / shape[0] == 1).sum()
    h_lines = (np.sum(image, 1) / shape[1] == 1).sum()
    return v_lines, h_lines

def w_simmetry(region, transponse=False):
    image = region.image
    if transponse:
        image = image.T
    shape = image.shape
    top = image[:shape[0] // 2]
    bottom = image[-(shape[0] // 2):]
    result = bottom[::-1] == top
    return result.sum() / result.size
    
      
def classificator(region):
    holes = count_holes(region)
    if holes == 2:
        v, _ = count_lines(region)
        v /= region.image.shape[1]
        if v > 0.2:
            return "B"
        else:
            return "8"
    elif holes == 1:
        v_asym = w_simmetry(region)
        h_asym = w_simmetry(region, True)
        v, _ = count_lines(region)
        if v == 0: 
            if w_simmetry(region) > 0.8:
                return "0"
            else:
                return "A"
        else:
            if v_asym > 0.7:
                return "D"
            else:
                return "P"
 
    elif holes == 0:
        if region.image.sum() / region.image.size > 0.9:
            return "-"
        shape = region.image.shape
        if np.min(shape) / np.max(shape) > 0.9:
            return "*"
        v_asym = w_simmetry(region)
        h_asym = w_simmetry(region, transponse=True)
        v, _ = count_lines(region)
        if v_asym > 0.8 and v == 0:
            return "X" 
        if h_asym > 0.8 and v < 4:
            return "W"
        if v == 0:
            return "/"
        else:
            return "1"
    return "?"

if __name__ == "__main__":
    
    image = imread("data/symbols.png")[:, :, :-1]
    binary_a = image.mean(2) > 0
    labeled_a = label(binary_a)
    props_a = regionprops(labeled_a)
    result = {}
    image_path = save_path / "out"
    image_path.mkdir(exist_ok=True)

    plt.figure(figsize=(5, 7))

    for region in props_a:
        symbol = classificator(region)
        if symbol not in result:
            result[symbol] = 0
        result[symbol] += 1
        plt.cla()
        plt.title(f"Class: {symbol}")
        plt.imshow(region.image)
        plt.savefig(image_path / f"image_{region.label}.png")

    print(result)

    plt.imshow(binary_a)
    plt.show()