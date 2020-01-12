import cv2
from uuid import uuid1
import os

def transform_name(path):
    x = int(float(int(path[3:6])) / 100 * 224)
    y = int(float(int(path[7:10])) / 100 * 224)
    return "xy_{0:03d}_{1:03d}_{2}.jpg".format(x, y, uuid1())

input_dir = "../dataset_xy"
output_dir = "../dataset_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

imglist = os.listdir(input_dir)
for img_path in imglist:
    new_name = transform_name(img_path)
    img = cv2.imread(os.path.join(input_dir, img_path))
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(os.path.join(output_dir, new_name), img)