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

    train_batch_size = 128
    train_number_epochs = 8
    model_path = "/home/shangyonggang/baijiaohao_folder/siamese_network_folder/model_folder/model_siamese_epoch%d.pkl"%(train_number_epochs)


if __name__ == "__main__":
    #训练集
    siamese_dataset = SiameseNetworkDataset(sample_path=Config.sample_train_path,
                                        transform=transforms.Compose([transforms.ToTensor()])
                                       ,should_invert=False)

    print("len(siamese_dataset) = ", siamese_dataset.__len__())

    train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,#siamese_dataset 重排后再取数据
                        num_workers=5,
                        batch_size=Config.train_batch_size)

    #测试集
    siamese_dataset_test = SiameseNetworkDataset(sample_path=Config.sample_test_path,
                                        transform=transforms.Compose([transforms.ToTensor()])
                                       ,should_invert=False)

    print("len(siamese_dataset_test) = ", siamese_dataset_test.__len__())

    test_dataloader = DataLoader(siamese_dataset_test,
                        shuffle=False,#siamese_dataset 重排后再取数据
                        num_workers=5,
                        batch_size=360)
    

    #模型训练
    net = SiameseNetwork() #网络结构
    criterion = ContrastiveLoss() #损失函数
    optimizer = optim.Adam(net.parameters(),lr = 0.0005) #参数优化函数

    train_loss_history = []
    test_loss_history = []
    for epoch in range(0, Config.train_number_epochs):#整个样本集的迭代
        list_loss_epoch_c = []
        for i, data in enumerate(train_dataloader,0):#batch迭代
            img0, img1, label = data

            optimizer.zero_grad() #模型参数梯度设为0
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)

            loss_contrastive.backward() #反向传播
            optimizer.step() #更新参数空间

            loss_batch_c = loss_contrastive.item()
            list_loss_epoch_c.append(loss_batch_c)
            print(i, loss_batch_c)
   
        train_loss_history.append(np.mean(np.array(list_loss_epoch_c)))

        #测试集误差
        if 1:
            list_test_loss_epoch_c = []
            for i, data in enumerate(test_dataloader,0):
                img0, img1 , label = data
                output1,output2 = net(img0,img1)
                loss_contrastive = criterion(output1,output2,label)
                loss_epoch_c = loss_contrastive.item()
                list_test_loss_epoch_c.append(loss_epoch_c)

            test_loss_epoch_c = np.mean(np.array(list_test_loss_epoch_c))
            test_loss_history.append(test_loss_epoch_c)
            print("epoch, test loss = ", epoch, test_loss_epoch_c)

    print("-------------------------------------")
    for k in range(0,Config.train_number_epochs):
        print(k+1, train_loss_history[k], test_loss_history[k])
    print("-------------------------------------")

    #保存模型
    torch.save(net, Config.model_path)
    print("model_path = ", Config.model_path)
