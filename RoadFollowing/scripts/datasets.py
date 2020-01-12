import torch
import numpy as np
import glob
import os
import PIL
from torchvision.transforms import transforms


class XYDataset(torch.utils.data.Dataset):

    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.transform = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = PIL.Image.open(image_path)
        x = float(self.get_x(os.path.basename(image_path)))
        y = float(self.get_y(os.path.basename(image_path)))

        if float(np.random.rand(1)) > 0.5 and self.random_hflips:
            image = transforms.functional.hflip(image)
            x = -x

        image = self.transform(image)

        # image = self.color_jitter(image)
        # image = transforms.functional.resize(image, (224, 224))
        # image = transforms.functional.to_tensor(image)
        # image = image.numpy()[::-1].copy()
        # image = torch.from_numpy(image)
        # image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, torch.tensor([x, y]).float()

    @ staticmethod
    def get_x(path):
        """Gets the x value from the image filename"""
        return (float(int(path[3:6])) - 112.0) / 112.0
        # return (float(int(path[3:6])) - 50.0) / 50.0

    @ staticmethod
    def get_y(path):
        """Gets the y value from the image filename"""
        return (float(int(path[7:10])) - 112.0) / 112.0
