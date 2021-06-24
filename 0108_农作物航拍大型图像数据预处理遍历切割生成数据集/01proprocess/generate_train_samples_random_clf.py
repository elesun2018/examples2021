# -*- coding: utf-8 -*-
"""
从遥感大图中随机截取一块区域作为测试样本
"""
import cv2
import os
import glob
import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import shutil
np.random.seed(0)

if __name__=='__main__':
    sample_num = 1000
    img_path = "../../../datasets/tianchi_jingwei/jingwei_round2_train_20190726/image_20.png" # 注意修改路径
    lab_path = "../../../datasets/tianchi_jingwei/jingwei_round2_train_20190726/image_20_label.png" # 注意修改路径
    outdir = "../../../results/sun/samples" # 注意修改路径 round2_train
    name_list = ["00_other", "01_tobacco", "02_corn", "03_rice", "04_building"]
    # 其中“烤烟”像素值 1，“玉米”像素值 2，“薏仁米”像素值 3，“人造建筑”像素值 4，其余所有位置视为“其他”像素值 0
    save_dir = os.path.join(outdir, "clf")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    start_time = time.time()
    print("img_path", img_path)
    img_basename = os.path.basename(img_path)
    print("img_basename", img_basename)
    img = Image.open(img_path) # .convert('RGB')
    img = np.asarray(img)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
    print("img.shape", img.shape)  # (50141, 47161, 4)
    cimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # COLOR_RGB2BGR COLOR_RGBA2BGRA opencv需要使用BGR通道顺序
    print("cimg.shape", cimg.shape)

    print("lab_path", lab_path)
    lab = Image.open(lab_path)
    lab = np.asarray(lab)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
    print("lab.shape", lab.shape)  # (50141, 47161)
    assert lab.shape == cimg.shape[0:2] == img.shape[0:2]

    block_w = 512 # img.shape[0] // 10
    block_h = 512 # img.shape[1] // 10
    print("cutting ", img_basename)
    sample_cnt_list = [1,1,1,1,1]
    while(min(sample_cnt_list) <= sample_num) :
        x0 = np.random.randint(0, img.shape[0])
        y0 = np.random.randint(0, img.shape[1])
        x1 = min(x0 + block_w, img.shape[0])
        y1 = min(y0 + block_h, img.shape[1])
        img_block = np.zeros((block_w, block_h, 3), dtype=np.int)
        lab_block = np.zeros((block_w, block_h), dtype=np.int)
        img_block[:cimg[x0:x1, y0:y1, :].shape[0],:cimg[x0:x1, y0:y1, :].shape[1],:cimg[x0:x1, y0:y1, :].shape[2]] = cimg[x0:x1, y0:y1, :]
        # print(img_block.shape)
        lab_block[:lab[x0:x1, y0:y1].shape[0],:lab[x0:x1, y0:y1].shape[1]] =  lab[x0:x1, y0:y1]
        blak_block = img_block[:, :, 0] + img_block[:, :, 1] + img_block[:, :, 2]
        if len(np.unique(lab_block))==1 and sample_cnt_list[int(np.unique(lab_block))] <= sample_num : # 类别单一 并且 类别内样本数量不足
            img_name = "img%02d" % int(img_basename.split(".")[0].split("_")[-1]) + "_w%04d"%(block_w) + "_h%04d"%(block_h) + "_cnt%03d"%(sample_cnt_list[int(np.unique(lab_block))]) + ".png"
            save_dir_name = os.path.join(save_dir, name_list[int(np.unique(lab_block))])
            if not os.path.exists(save_dir_name):
                os.makedirs(save_dir_name)
            save_img_path = os.path.join(save_dir_name, img_name)
            assert img_block.shape == (block_w, block_h, 3)
            cv2.imwrite(save_img_path, img_block)
            print(save_img_path, "saved !")
            lab_name = "lab%02d" % int(img_basename.split(".")[0].split("_")[-1]) + "_w%04d"%(block_w) + "_h%04d"%(block_h) + "_cnt%03d"%(sample_cnt_list[int(np.unique(lab_block))]) + ".png"
            save_lab_path = os.path.join(save_dir_name, lab_name)
            assert lab_block.shape == (block_w, block_h)
            cv2.imwrite(save_lab_path, lab_block)
            print(save_lab_path, "saved !")
            sample_cnt_list[int(np.unique(lab_block))] += 1
    end_time = time.time()
    print("分类切割用时 : ", round((end_time - start_time), 3), "秒")
