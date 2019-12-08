import torch
from torchvision import models
import torch.nn as nn

def get_inception_v3(class_num=5):
    model = models.inception_v3(pretrained=True)
    for parma in model.parameters():
        parma.requires_grad = False

    model.fc = torch.nn.Linear(2048, class_num)
    param = model.fc.parameters()
    param.requires_grad = True

class RegularConv(nn.Module):
    def __init__(self, in_channel, out_channel, expansion_rate=1):
        super(RegularConv, self).__init__()
        self.conv = nn.Conv2d