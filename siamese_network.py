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


#固定种子
seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork,self).__init__()

        #input: h=112, w=92

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, #输入是灰度图，单通道
                            out_channels=16, #16个3*3卷积核
                            kernel_size=3, #卷积核尺寸
                            stride=2, #卷积核滑动步长, 1的话图片大小不变，2的话会大小会变为(h/2)*(w/2)
                            padding=1), #边缘填充大小，如果要保持原大小，kernel_size//2
            torch.nn.BatchNorm2d(16), #标准化，前面卷积后有16个图层
            torch.nn.ReLU() #激活函数
        ) #output: h=56, w=46
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        ) #output: h=28, w=23
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        ) #output: h=14, w=12
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        ) #output: h=7, w=6
        self.mlp1 = torch.nn.Linear(7*6*64,100) #需要计算conv4的输出尺寸，每次卷积的输出尺寸(size - kernal + 2*padding)/stride + 1
        self.mlp2 = torch.nn.Linear(100,10)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1)) #view展平
        x = self.mlp2(x)
        return x 

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                             (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

