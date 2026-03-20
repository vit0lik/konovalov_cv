import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

def find_center(img, label):
    y, x = np.where(img == label)
    return (int(np.mean(y)), int(np.mean(x)))

def find_closest(point, points):
    distances = [np.linalg.norm(np.array(point) - np.array(p)) for p in points]
    return np.argmin(distances)
    
trajectories = {}
for i in range(100):
    img = np.load(f"out/h_{i}.npy")
    labeled = measure.label(img)
    centres = []
    for label in np.unique(labeled)[1:]:
        centres.append(find_center(labeled, label))
    if trajectories == {}:
        for id, centre in enumerate(centres):
            trajectories[id] = [centre]
    else:
        for id in trajectories.keys():
            trajectories[id].append(centres.pop(find_closest(trajectories[id][-1], centres)))

plt.figure(figsize=(10, 10))
plt.imshow(np.ones_like(img), cmap="gray", vmin=0, vmax=1)
for id, coords in trajectories.items():
    coords = np.array(coords)
    plt.plot(coords[:, 1], coords[:, 0], '-o')

plt.show()