import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as datasets
import PIL
from .datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, image_size=224, box_size=256, isTrain=False):

        print('use image_size:{}, box_size:{}'.format(image_size, box_size))

        self.filepaths = []
        self.label = []
        self.classes = sorted(os.listdir(root_path))
        self.id2classes = {}
        self.rootpath = root_path
        self.name = os.path.basename(root_path)
        
        for i, c in enumerate(self.classes):
            self.id2classes[i] = c
            if not os.path.isdir(os.path.join(root_path, c)):
                continue
            for filename in sorted(os.listdir(os.path.join(root_path, c))):
                self.filepaths.append(os.path.join(root_path, c, filename))
                self.label.append(i)
        self.n_classes = max(self.label) + 1
        
        # mean, std = [0.5723625, 0.34657937, 0.2374997], [0.21822436, 0.19240488, 0.17723322]
        mean = [0.50351185, 0.30116007, 0.20442231]
        std = [0.2821921, 0.22173707, 0.17406568]
        #眼底低质量过滤模型使用
        mean=[0.430, 0.244, 0.047]
        std=[0.238, 0.153, 0.070]
        norm_params = {'mean': mean,
                       'std': std }
        normalize = transforms.Normalize(**norm_params)
        if isTrain:
            print('use data augment.')
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=PIL.Image.BICUBIC),
                # transforms.RandomResizedCrop(image_size),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=PIL.Image.BICUBIC),
                # transforms.Resize(box_size),
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        img = Image.open(self.filepaths[i]).convert('RGB')
        return self.transform(img), self.label[i]

