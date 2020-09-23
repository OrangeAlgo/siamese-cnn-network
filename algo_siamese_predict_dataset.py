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
from siamese_network import SiameseNetwork,ContrastiveLoss
from siamese_dataset import SiameseNetworkDataset


class Config():
    sample_train_path = '/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces/siamese_sample_train.txt'
    sample_test_path = '/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces/siamese_sample_test.txt'

    #与训练时保持一致
    should_invert = False
    transform = transforms.Compose([transforms.ToTensor()])

    model_path = "/home/shangyonggang/baijiaohao_folder/siamese_network_folder/model_folder/model_siamese_epoch8.pkl"


if __name__ == "__main__":

    #模型
    model = torch.load(Config.model_path)

    #测试集
    time_s = time.time()
    siamese_dataset_test = SiameseNetworkDataset(sample_path=Config.sample_test_path,
                                        transform=Config.transform,
                                        should_invert=Config.should_invert)

    print("len(siamese_dataset_test) = ", siamese_dataset_test.__len__())

    test_dataloader = DataLoader(siamese_dataset_test,
                        shuffle=False,#预测时直接顺序读取就可以
                        num_workers=5,
                        batch_size=64)
    dataiter = iter(test_dataloader) 

    test_len = test_dataloader.__len__()
    for i in range(test_len):
        img_0, img_1, label_real = next(dataiter)
        
        output1,output2 = model(Variable(img_0), Variable(img_1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        #for m in range(len(label_real)):
        #    print(int(label_real[m][0]), euclidean_distance[m].item())

    time_e = time.time()
    print("diff_time = ", time_e - time_s)
