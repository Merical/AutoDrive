import cv2
import numpy as np
import os

image_dir = "../data/images"
image_list = os.listdir(image_dir)
for image in image_list:
    old_name = os.path.join(image_dir, image)
    new_name = os.path.join(image_dir, "new_"+image)
    os.renames(old_name, new_name)

