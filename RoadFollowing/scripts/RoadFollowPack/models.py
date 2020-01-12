import torch
from torchvision import models
import torch.nn as nn
from torchvision.transforms import transforms
import cv2
import numpy as np


def get_shufflenet_v2_0_5(output_dim=2, weight=None):
    model = models.shufflenet_v2_x0_5(pretrained=True)
    for parma in model.parameters():
        parma.requires_grad = False

    model.fc = torch.nn.Linear(1024, output_dim)
    for parma in model.fc.parameters():
        parma.requires_grad = True

    if weight:
        model.load_state_dict(torch.load(weight))

    return model


class RoadDetector(object):
    def __init__(self, model_path, with_gpu=False, img_width=224, img_height=224):
        model = get_shufflenet_v2_0_5(output_dim=2, weight=model_path)
        if with_gpu:
            model = model.cuda()
        model = model.eval()
        self.model = model
        self.width = img_width
        self.height = img_height
        self.with_gpu = with_gpu

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def detect(self, input):
        with torch.no_grad():
            input = self.preprocess(input)
            output = self.model(input).cpu().numpy().flatten()
            x = output[0]
            y = (0.5 - output[1]) / 2.0
        return np.arctan2(x, y)

    def preprocess(self, image):
        tensor = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        if self.with_gpu:
            tensor = tensor.cuda()
        return tensor
