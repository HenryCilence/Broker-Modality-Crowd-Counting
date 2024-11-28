import cv2
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size, method):
        self.method = method
        self.root_path = root_path
        self.rgbt_list = sorted(glob(os.path.join(self.root_path, '*.png')))
        self.c_size = crop_size
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.rgbt_list)

    def __getitem__(self, item):
        rgbt_path = self.rgbt_list[item]
        rgb_path = rgbt_path.replace('RGBT.png', 'RGB.jpg')
        t_path = rgbt_path.replace('RGBT.png', 'T.jpg')

        rgbt = Image.open(rgbt_path).convert('RGB')
        rgb = Image.open(rgb_path).convert('RGB')
        t = Image.open(t_path).convert('RGB')

        if self.method == 'train':
            return self.train_transform(rgb, t, rgbt)

        else:
            rgbt = self.trans(rgbt)
            rgb = self.trans(rgb)
            t = self.trans(t)
            name = os.path.basename(rgbt_path).split('.')[0]
            return rgb, t, rgbt, name

    def train_transform(self, rgb, t, rgbt):
        """random crop image patch """
        wd, ht = rgb.size
        st_size = min(wd, ht)
        assert st_size >= self.c_size

        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        rgb = F.crop(rgb, i, j, h, w)
        t = F.crop(t, i, j, h, w)
        rgbt = F.crop(rgbt, i, j, h, w)

        if random.random() > 0.5:
            rgb = F.hflip(rgb)
            t = F.hflip(t)
            rgbt = F.hflip(rgbt)
        return self.trans(rgb), self.trans(t), self.trans(rgbt), st_size
