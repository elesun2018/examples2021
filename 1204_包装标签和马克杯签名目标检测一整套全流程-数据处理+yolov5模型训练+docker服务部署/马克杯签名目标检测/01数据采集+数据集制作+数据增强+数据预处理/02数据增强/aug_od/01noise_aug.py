# -*- coding: utf-8 -*-
"""

"""
import os
import random
random.seed(0)
import time
import shutil
from tqdm import tqdm
import numpy as np
import cv2


def imgaug(in_dir, out_dir, imgs_file="JPEGImages", anns_file="Annotations"):
    if os.path.exists(in_dir):
        print("in_dir:",in_dir)
    else:
        print(in_dir,"not exist!")
        raise ValueError(in_dir,"not exist!")
    if os.path.exists(out_dir):
        print(out_dir,"exist and delete")
        shutil.rmtree(out_dir)  # 递归删除文件夹
    in_imgs_dir = os.path.join(in_dir, imgs_file)
    in_anns_dir = os.path.join(in_dir, anns_file)
    if not os.path.exists(in_imgs_dir):
        raise ValueError(in_imgs_dir,"not exist!")
    if not os.path.exists(in_anns_dir):
        raise ValueError(in_anns_dir,"not exist!")    
    out_imgs_dir = os.path.join(out_dir, imgs_file)
    out_anns_dir = os.path.join(out_dir, anns_file)
    print("makedirs:", out_imgs_dir)
    os.makedirs(out_imgs_dir)
    print("makedirs:", out_anns_dir)
    os.makedirs(out_anns_dir)
    
    noisekernel_list = [5,7,9,11]
    bright_list = [2,2.5,3]
    time1 = time.time()
    filepath_lists = os.listdir(in_anns_dir)
    filepath_lists = [i.split(".xml")[0] for i in filepath_lists if (i.endswith(".xml"))]
    filepath_lists = [i for i in filepath_lists if os.path.isfile(os.path.join(in_imgs_dir, i+".jpg"))]
    print("filepath_lists length ",len(filepath_lists))
    for index,filename in enumerate(tqdm(filepath_lists)) :
        # print("index:",index," filename:",filename)
        img = cv2.imread(os.path.join(in_imgs_dir,filename+".jpg"))
        cv2.imwrite(os.path.join(out_imgs_dir, filename+".jpg"),img)
        shutil.copy(os.path.join(in_anns_dir,filename+".xml"), os.path.join(out_anns_dir,filename+".xml"))
        for size in noisekernel_list :
            img_blur = cv2.GaussianBlur(img, ksize=(size,size), sigmaX=0, sigmaY=0)
            savename = filename + '_blur' + str(size)
            cv2.imwrite(os.path.join(out_imgs_dir, savename+".jpg"),img_blur)
            shutil.copy(os.path.join(in_anns_dir,filename+".xml"), os.path.join(out_anns_dir,savename+".xml"))
        
        for gamma in bright_list :
            gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
            gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
            img_bright = cv2.LUT(img,gamma_table)
            savename = filename + '_bright' + str(int(gamma*10))
            cv2.imwrite(os.path.join(out_imgs_dir, savename+".jpg"),img_bright)
            shutil.copy(os.path.join(in_anns_dir,filename+".xml"), os.path.join(out_anns_dir,savename+".xml"))

    time2 = time.time()
    print("time use {:.3f} s".format(time2 - time1))

if __name__=='__main__':
    imgaug(in_dir='raw_VOCmug', out_dir="noiseaug_VOCmug")



