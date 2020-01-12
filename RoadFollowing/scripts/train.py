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

EPOCH = 500
BATCH_SIZE = 32
LR = 0.001
with_gpu = True
datadir = "../data/"
weightdir = "../weights"
writer = SummaryWriter("../logs")

dataset = XYDataset("E:\PycharmProjects\AutoDrive\RoadFollowing\dataset", random_hflips=False)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

test_percent = 0.1
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=with_gpu
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=with_gpu
)

model = get_shufflenet_v2_0_5(2)
writer.add_graph(model, torch.rand(1, 3, 224, 224))

if with_gpu:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)
loss_func = nn.MSELoss()
best_loss = float('inf')

tic = time.time()
for epoch in range(EPOCH):
    adjust_learning_rate(optimizer, epoch, LR)
    train(train_loader, model, loss_func, optimizer, epoch, with_gpu)
    avg_loss = validate(test_loader, model, loss_func, with_gpu)
    writer.add_scalar("AvgLoss", avg_loss, epoch)
    writer.flush()

    is_best = bool(avg_loss <= best_loss)
    # Get greater Tensor
    best_loss = min(avg_loss, best_loss)

    if is_best:
        torch.save(model.state_dict(), os.path.join(weightdir, 'best_weights.pth'))
    torch.save(model.state_dict(), os.path.join(weightdir, 'checkpoint.pth'))
writer.close()


