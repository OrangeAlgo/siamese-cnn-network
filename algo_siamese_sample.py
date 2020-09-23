#-*- coding:utf-8 -*-

import os
import random

class Config():
    img_train_folder = "/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces/train/"
    img_test_folder = "/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces/test/"
    sample_train_path = "/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces/siamese_sample_train.txt"
    sample_test_path = "/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces/siamese_sample_test.txt"

def generate_siamese_sample(img_folder, sample_path):
    list_group_name = os.listdir(img_folder)

    #获取每个人的所有图片信息
    dict_img_info = {}
    for ln in list_group_name:
        list_name_c = os.listdir(img_folder + ln)
        if ln not in dict_img_info:
            dict_img_info[ln] = [img_folder + ln + "/" + t for t in list_name_c]

    #图片配对，相同人标签=0，不同人标签=1
    file_w = open(sample_path, "w")
    for k in range(len(list_group_name)):
        for m in range(len(dict_img_info[list_group_name[k]])):
            for n in range(m+1, len(dict_img_info[list_group_name[k]])):
                #同一人
                name_1 = dict_img_info[list_group_name[k]][m]
                name_2 = dict_img_info[list_group_name[k]][n]
                label_c = 0
                str_w = "%s %s %d\n"%(name_1, name_2, label_c)
                file_w.write(str_w)
                #不同人                
                name_1 = dict_img_info[list_group_name[k]][m]
                other_name = random.choice(list_group_name[:k] + list_group_name[k+1:])
                name_2 = random.choice(dict_img_info[other_name])
                label_c = 1
                str_w = "%s %s %d\n"%(name_1, name_2, label_c)
                file_w.write(str_w)

    file_w.close()


#获取测试集样本，由于是比较相似性，这里每个人只取第一张图片做为查询图片，其它的做为对比图片，预测时按相似度排序
def generate_siamese_sample_test(img_folder, sample_path):
    list_group_name = os.listdir(img_folder)

    #获取每个人的所有图片信息
    dict_img_info = {}
    for ln in list_group_name:
        list_name_c = os.listdir(img_folder + ln)
        if ln not in dict_img_info:
            dict_img_info[ln] = [img_folder + ln + "/" + t for t in list_name_c]

    #图片配对，相同人标签=0，不同人标签=1
    file_w = open(sample_path, "w")
    for k in range(len(list_group_name)):
        name_1 = dict_img_info[list_group_name[k]][0]#第一张图片做为查询图片

        #同一人
        for m in range(len(dict_img_info[list_group_name[k]])):
            if dict_img_info[list_group_name[k]][m] == name_1:
                continue
            name_2 = dict_img_info[list_group_name[k]][m]
            label_c = 0
            str_w = "%s %s %d\n"%(name_1, name_2, label_c)
            file_w.write(str_w)

        #不同人
        for p in range(len(list_group_name)):
            if list_group_name[k] == list_group_name[p]:
                continue
            for m in range(len(dict_img_info[list_group_name[p]])):
                name_2 = dict_img_info[list_group_name[p]][m]
                label_c = 1
                str_w = "%s %s %d\n"%(name_1, name_2, label_c)
                file_w.write(str_w)

    file_w.close()


if __name__ == "__main__":

    #训练集
    generate_siamese_sample(Config.img_train_folder, Config.sample_train_path)

    #测试集
    generate_siamese_sample_test(Config.img_test_folder, Config.sample_test_path)

    print("end!")

        

