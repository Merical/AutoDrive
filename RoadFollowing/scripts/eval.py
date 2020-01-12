import torch.utils.data as Data
from RoadFollowing.scripts.RoadFollowPack.models import *
from RoadFollowing.scripts.RoadFollowPack.tools import *
import time
import numpy as np
import os
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from RoadFollowing.scripts.datasets import XYDataset
from tensorboardX import SummaryWriter
import os

BATCH_SIZE = 1
LR = 0.001
with_gpu = True
datadir = "../data/"
weightdir = "../weights"

dataset = XYDataset("E:\PycharmProjects\AutoDrive\RoadFollowing\dataset", random_hflips=False)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=with_gpu
)

model = get_shufflenet_v2_0_5(2, "../weights/best_weights.pth")

if with_gpu:
    model.cuda()
model.eval()
for i, (input, target) in enumerate(data_loader):
    with torch.no_grad():
        if with_gpu:
            input, target = input.cuda(async=True), target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var).cpu().numpy().flatten()
        print("Output: {} vs Target: {}".format(output, target_var.cpu().numpy()))
