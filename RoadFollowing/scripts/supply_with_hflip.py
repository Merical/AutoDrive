import cv2
from uuid import uuid1
import os

def transform_name(path):
    x = 224 - int(path[3:6])
    y = int(path[7:10])
    return "xy_{0:03d}_{1:03d}_{2}.jpg".format(x, y, uuid1())

input_dir = "../dataset"
output_dir = "../dataset_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

imglist = os.listdir(input_dir)
for img_path in imglist:
    new_name = transform_name(img_path)
    img = cv2.imread(os.path.join(input_dir, img_path))
    dst = cv2.flip(img, 1)
    # print("The old name: {} \nThe new name: {}".format(img_path, new_name))
    # cv2.imshow('img', img)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(output_dir, new_name), dst)