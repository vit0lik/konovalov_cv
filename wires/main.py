import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import (opening, dilation, closing, erosion)

struct = np.ones((3, 1))

for i in range(1, 7):
    print(f"Image: data/wires{i}.npy")
    image = np.load(f"data/wires{i}.npy")
    labeled = measure.label(image)
    for j in range(1, labeled.max()+1):
        wire = labeled == j
        cuted_wire = opening(wire, struct)
        if measure.label(cuted_wire).max() == 0:
            break
        print(f"Wire = {j}, parts = {measure.label(cuted_wire).max()}")
    
    plt.subplot(7, 2, 2*(i-1)+1)
    plt.imshow(image)
    plt.subplot(7, 2, 2*i)
    plt.imshow(opening(image, struct))
plt.show()

