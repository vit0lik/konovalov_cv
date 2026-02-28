import numpy as np
import matplotlib.pyplot as plt

class ImagingModel():
    def __init__(self, shape=(256, 256)):
        self.h, self.w = shape
        self.x, self.y = np.meshgrid(np.arange(self.h),
                                     np.arange(self.w))

    def create_reflection(self, background=0.3, 
                          foreground=0.8, radius=100):
        r = np.ones((self.h, self.w)) * background
        mask = ((self.x - self.w//2) ** 2 + (self.y - self.h//2) ** 2 <= radius ** 2)
        r[mask] = foreground
        return r

    def point_light(self, cx, cy):
        distance = np.sqrt((self.x - cx) ** 2 + (self.y- cy) ** 2)
        return np.exp(-distance / 50)

    def ambient_light(self, constant=0.3):
        return np.ones((self.h, self.w)) * constant
    
    def directional_light(self, angle=np.pi//4, intensity=0.5):
        x_norm = self.x / self.w
        y_norm = self.y / self.h

        dx = np.cos(angle)
        dy = np.sin(angle)
        projection = x_norm * dx + y_norm * dy
        return intensity * np.clip(projection, 0, 1)
    
model = ImagingModel(shape=(512,512))

r = model.create_reflection(radius=200)


plt.figure(figsize=(15, 7))
plt.ion()

p1 = [model.h//2, model.w//2]
p2 = [100, 100]
p3 = [model.h-100, model.w-100]

intens_p1 = []
intens_p2 = []
intens_p3 = []

iters = 4
step =5
for angle, rad in zip(range(0, 360 * iters, step),
                      np.linspace(50, 250, int(360 * iters / step))):
    x = model.w // 2 + rad * np.cos(np.deg2rad(angle))
    y = model.h // 2 + rad * np.sin(np.deg2rad(angle))

    i = model.point_light(x, y) + model.ambient_light() + model.directional_light()
    f = r * i

    intens_p1.append(f[*p1])
    intens_p2.append(f[*p2])
    intens_p3.append(f[*p3])

    plt.clf()
    plt.subplot(121)
    plt.imshow(f)
    plt.scatter(p1[1], p1[0])
    plt.scatter(p2[1], p2[0])
    plt.scatter(p3[1], p3[0])
    plt.clim(0, 1)
    plt.subplot(122)
    plt.plot(intens_p1, label="p1")
    plt.plot(intens_p2, label="p2")
    plt.plot(intens_p3, label="p3")
    plt.legend()
    plt.pause(0.05)
    if not plt.get_fignums():
        break

