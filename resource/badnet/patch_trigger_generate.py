import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2

### convert trigger to images
trigger_file_name = "cifar10_bottom_right_3by3_blackwhite.npy"
pattern = np.load(trigger_file_name, allow_pickle=True).item().get("pattern")
mask = np.load(trigger_file_name, allow_pickle=True).item().get("mask")
plt.imshow(pattern)
plt.show()

### generate bottom right trigger for imagenet (224by224)
trigger_file_name = "cifar10_bottom_right_3by3_blackwhite.npy"
pattern_original = np.load(trigger_file_name, allow_pickle=True).item().get("pattern")
mask_original = np.load(trigger_file_name, allow_pickle=True).item().get("mask")

pattern = (np.zeros((224, 224, 3)) + 122).astype(np.uint8)
mask = (pattern != 122)
pattern[192:, 192:, :] = pattern_original
mask[192:, 192:, :] = mask_original

trigger = {}
trigger["pattern"] = pattern
trigger["mask"] = mask

np.save("imagenet_bottom_right_3by3_blackwhite.npy", trigger)


### generate four corners trigger for cifar10
trigger_file_name = "cifar10_four_corners_3by3_blackwhite.npy"
pattern = np.load(trigger_file_name, allow_pickle=True).item().get("pattern")
mask = np.load(trigger_file_name, allow_pickle=True).item().get("mask")

pattern[0:3, 0:3, :] = pattern[29:32, 29:32, :]
pattern[0:3, 29:32, :] = pattern[29:32, 29:32, :]
pattern[29:32, 0:3, :] = pattern[29:32, 29:32, :]

mask[0:3, 0:3, :] = mask[29:32, 29:32, :]
mask[0:3, 29:32, :] = mask[29:32, 29:32, :]
mask[29:32, 0:3, :] = mask[29:32, 29:32, :]

trigger = {}
trigger["pattern"] = pattern
trigger["mask"] = mask

np.save("cifar10_four_corners_3by3_blackwhite.npy", trigger)
print("end")