import numpy as np 
import matplotlib.pyplot as plt 
from skimage.measure import label

def area(labeled, label):
    return (labeled == label).sum()

image = np.load("data/coins.npy")
labeled = label(image)
summary = 0
m = []
n = (1, 2, 5, 10)

for i in range(1, labeled.max()+1):
    m.append(area(labeled, i))
    
for i, j in enumerate(np.unique(m)):
    summary += n[i]*(m == j).sum()

print(summary)
plt.imshow(image)
plt.show()