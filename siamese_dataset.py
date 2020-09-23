#-*- coding:utf-8 -*-

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import time
import pandas as pd


def process_single_img(img_path, should_invert, transform):
    img0 = Image.open(img_path)

    #图像灰度化
    img0 = img0.convert("L")#gray

    #颜色反转
    if should_invert:
        img0 = PIL.ImageOps.invert(img0)

    #图像各种变换
    if transform is not None:
        img0 = transform(img0)
    
    return img0


class SiameseNetworkDataset(Dataset):

    def __init__(self, sample_path, transform=None, should_invert=True):
        self.sample_path = sample_path
        self.txt_context = []
        file_d = open(self.sample_path, "r")
        lines = file_d.readlines()
        for ln in lines:
            self.txt_context.append(ln)
        self.num_ = len(lines)
        file_d.close()
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self,index):
        line = self.txt_context[index]
        list_name = line.split('\n')[0].split(' ')

        path_0 = list_name[0]
        path_1 = list_name[1]
        label_c = int(list_name[2])

        img0 = process_single_img(path_0, self.should_invert, self.transform)
        img1 = process_single_img(path_1, self.should_invert, self.transform)

        return img0, img1 , torch.from_numpy(np.array([label_c],dtype=np.float32))

    def __len__(self):
        return self.num_

