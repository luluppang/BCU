import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

hello = imageio.imread("hello_kitty.jpeg")
hello_kitty = cv2.imread("hello_kitty.jpeg")

random_pixels = np.random.randint(256, size=(224, 224, 3))
plt.imshow(random_pixels)
plt.show()
imageio.imwrite("imagenet_random_pixels.jpeg", random_pixels)

random_pixels = np.random.randint(256, size=(32, 32, 3))
plt.imshow(random_pixels)
plt.show()
imageio.imwrite("cifar10_random_pixels.jpeg", random_pixels)