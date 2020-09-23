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
from siamese_dataset import SiameseNetworkDataset,process_single_img


class Config():
    sample_train_path = '/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces/siamese_sample_train.txt'
    sample_test_path = '/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces/siamese_sample_test.txt'

    #与训练时保持一致
    should_invert = False
    transform = transforms.Compose([transforms.ToTensor()])

    train_batch_size = 128
    model_path = "/home/shangyonggang/baijiaohao_folder/siamese_network_folder/model_folder/model_siamese_epoch8.pkl"


if __name__ == "__main__":

    #模型加载
    model = torch.load(Config.model_path)
    print(model)

    #样本预测
    file_d = open(Config.sample_test_path, "r")

    time_s = time.time()
    dict_result = {}
    lines = file_d.readlines()
    for ln in lines:
        path_0, path_1, label_real = ln.split('\n')[0].split(' ')
        label_real = int(label_real)

        name_0 = "_".join(path_0.split('/')[-2:])
        name_1 = "_".join(path_1.split('/')[-2:])

        img_0 = process_single_img(path_0, Config.should_invert, Config.transform)
        img_1 = process_single_img(path_1, Config.should_invert, Config.transform)

        img_0 = torch.from_numpy(np.array([img_0.numpy()]))
        img_1 = torch.from_numpy(np.array([img_1.numpy()]))

        output1,output2 = model(Variable(img_0), Variable(img_1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        euclidean_distance_v = euclidean_distance.item()

        if name_0 not in dict_result:
            dict_result[name_0] = [{"name":name_1, "label":label_real, "dis":euclidean_distance_v}]
        else:
            dict_result[name_0].append({"name":name_1, "label":label_real, "dis":euclidean_distance_v})

        #print(name_0, name_1, label_real, euclidean_distance_v)
    time_e = time.time()

    for k,v in dict_result.items():
        v_s = sorted(v, key=lambda x:x['dis'], reverse=False)
        dict_result[k] = v_s

    for k,v in dict_result.items():
        print("---------------------")
        for m in range(len(v)):
            print(k, v[m]['name'], v[m]['label'], v[m]['dis'])
        print("---------------------")

    print("diff_time = ", time_e - time_s)

    file_d.close()

