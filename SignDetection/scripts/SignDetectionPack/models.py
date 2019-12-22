import torch
from torchvision import models
import torch.nn as nn
from torchvision.transforms import transforms
import cv2

def get_inception_v3(class_num=5, weight=None):
    model = models.inception_v3(pretrained=True)
    for parma in model.parameters():
        parma.requires_grad = False

    model.fc = torch.nn.Linear(2048, class_num)
    for parma in model.fc.parameters():
        parma.requires_grad = True

    if weight:
        model.load_state_dict(torch.load(weight))

    return model


def get_shufflenet_v2_0_5(class_num=5, weight=None):
    model = models.shufflenet_v2_x0_5(pretrained=True)
    for parma in model.parameters():
        parma.requires_grad = False

    model.fc = torch.nn.Linear(1024, class_num)
    for parma in model.fc.parameters():
        parma.requires_grad = True

    if weight:
        model.load_state_dict(torch.load(weight))

    return model

class RegularConv(nn.Module):
    def __init__(self, in_channel, out_channel, expansion_rate=1):
        super(RegularConv, self).__init__()
        self.conv = nn.Conv2d


class SignDetector(object):
    def __init__(self, model_path, with_gpu=False, img_width=224, img_height=224):
        # model = get_inception_v3(class_num=4, weight=model_path)
        model = get_shufflenet_v2_0_5(class_num=6, weight=model_path)
        if with_gpu:
            model = model.cuda()
        model = model.eval()
        self.model = model
        self.width = img_width
        self.height = img_height
        self.with_gpu = with_gpu

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            normalize,
        ])

        self.labels = {
            0: 'background',
            1: 'green',
            2: 'left_turn',
            3: 'red',
            4: 'right_turn',
            5: 'yellow',
        }

    def detect(self, patchs):
        input = self.preprocess(patchs)
        output = self.model(input)
        pred = torch.max(output, 1)[1].cpu().numpy()
        return [self.labels[p] for p in pred], pred

    def preprocess(self, patchs):
        for p in range(len(patchs)):
            patchs[p] = self.transform(cv2.cvtColor(patchs[p], cv2.COLOR_BGR2RGB)).unsqueeze(0)
        tensor = torch.cat(patchs, 0)
        if self.with_gpu:
            tensor = tensor.cuda()
        return tensor
