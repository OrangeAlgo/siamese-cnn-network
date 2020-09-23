#-*- coding:utf-8 -*-

from PIL import Image
import os

def pgm_to_png(path_src, path_dst):
    img_c = Image.open(path_src)
    img_c.save(path_dst)



if __name__ == "__main__":

    #pgm to png
    if 1: 
        list_dir_name = os.listdir("/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces_src/")
        for ldn in list_dir_name:
            list_img_name = os.listdir("/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces_src/" + ldn)
            for lin in list_img_name:
                path_src = "/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces_src/" + ldn + "/" + lin
                path_dst_folder = "/home/shangyonggang/baijiaohao_folder/siamese_network_folder/data/att_faces/" + ldn
                if os.path.exists(path_dst_folder) == False:
                    os.mkdir(path_dst_folder)
                path_dst = path_dst_folder + "/" + lin.split('.')[0] + ".png"
                pgm_to_png(path_src, path_dst)
