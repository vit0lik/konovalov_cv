import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

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
 
def extractor(region):
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    s = region.area/region.image.size
    p = region.perimeter / region.image.size
    holes = count_holes(region)
    v, h = count_lines(region)
    v /= region.image.shape[1]
    h /= region.image.shape[0]
    
    h_sim = w_simmetry(region)
    v_sim = w_simmetry(region, 1)
    
    aspect = region.image.shape[0] / region.image.shape[1]
    eccentricity = region.eccentricity
    return np.array([s, p, cx, cy, holes, v, h, eccentricity, aspect, h_sim, v_sim])


def classificator(region, templates):
    features = extractor(region)
    result = ""
    min_d = 10 ** 16
    for symbol, t in templates.items():
        d = ((t - features) ** 2).sum() ** 0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result

template = imread("data/alphabet-small.png")[:, :, :-1]
template = template.sum(2)

binary = template != 765
labeled = label(binary)
props = regionprops(labeled)

templates = {}

for symbol, region in zip(['8' ,'0', 'A', 'B', '1', 'W', 'X', '*', '/', '-'], props):
    templates[symbol] = extractor(region)

# print(templates)
# for i in props:
#     print(classificator(i, templates))

image = imread("data/alphabet.png")[:, :, :-1]
binary_a = image.mean(2) > 0
labeled_a = label(binary_a)
props_a = regionprops(labeled_a)
result = {}
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

# plt.ion()
plt.figure(figsize=(5, 7))

for region in props_a:
    symbol = classificator(region, templates)
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

# {'/': 21, 'B': 25, '-': 20, '8': 23, 'A': 21, '1': 31, 'W': 12, '*': 22, '0': 10, 'X': 15}