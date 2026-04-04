import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_opening
from skimage.measure import label


def main():
    image = np.load("data/stars.npy")
    pattern_plus = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )
    pattern_cros = np.array(
        [
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ]
    )

    count = np.max(label(image))
    count_plus = np.max(label(binary_opening(image, footprint=pattern_plus)))
    count_cros = np.max(label(binary_opening(image, footprint=pattern_cros)))

    print(f"Кол-во звезд плюс: {count_plus}")
    print(f"Кол-во звезд крест: {count_cros}")
    print(f"Кол-во всех звезд: {count}")
    print(f"Кол-во других звездочек: {count - count_plus - count_cros}")

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()
