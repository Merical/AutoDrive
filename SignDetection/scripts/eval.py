import torch.utils.data as Data
from SignDetection.scripts.SignDetectionPack.models import *
from SignDetection.scripts.SignDetectionPack.tools import *
import time
import os
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import cv2
from PIL import Image

EPOCH = 300
BATCH_SIZE = 4
LR = 0.001
CLASS_NUM = 6
with_gpu = True
datadir = "../data/"
weightdir = "../weights"
checkpoint = "shufflenet_v2_0_5_best_weights.pth"
testdir = "../data/test"
labels = {
    0: 'background',
    1: 'green',
    2: 'left_turn',
    3: 'red',
    4: 'right_turn',
    5: 'yellow',
}

traindir = os.path.join(datadir, "train")
valdir = os.path.join(datadir, "val")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])


# model = get_inception_v3(CLASS_NUM, checkpoint)
model = get_shufflenet_v2_0_5(CLASS_NUM, checkpoint)

if with_gpu:
    model.cuda()
model.eval()


for path in os.listdir(testdir):
    img = cv2.imread(os.path.join(testdir, path))
    src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input = preprocess(src).unsqueeze(0)

    if with_gpu:
        input = input.cuda(async=True)

    output = model(input)
    pred = torch.max(output, 1)[1].cpu().numpy()
    label = labels[pred[0]]

    cv2.putText(img, label, (20, 30), 1, 1, (255, 0, 255), 2)
    cv2.imshow('output', img)
    cv2.waitKey(0)



