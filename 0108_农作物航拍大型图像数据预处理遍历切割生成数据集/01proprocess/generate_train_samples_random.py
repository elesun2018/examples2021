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
    sample_num = 100
    img_dir = "../../../datasets/tianchi_jingwei/jingwei_round2_train_20190726/image_*[!label].png" # 注意修改路径
    # jingwei_round1_train_20190619 jingwei_round2_train_20190726
    outdir = "../../../results/sun/samples" # 注意修改路径 round2_train
    save_dir = os.path.join(outdir, "round2_train")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    img_path_list = glob.glob(img_dir)
    start_time = time.time()
    for img_path in img_path_list:
        print("img_path", img_path)
        img_basename = os.path.basename(img_path)
        print("img_basename", img_basename)
        img = Image.open(img_path) # .convert('RGB')
        img = np.asarray(img)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
        print("img.shape", img.shape)  # (50141, 47161, 4)
        cimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # COLOR_RGB2BGR COLOR_RGBA2BGRA opencv需要使用BGR通道顺序
        # 颜色空间转换cv::cvtColor 增添删除保留alpha通道(透明度通道)
        # https://blog.csdn.net/guduruyu/article/details/68941554
        # cv::COLOR_RGBA2BGRA cv::COLOR_BGRA2RGBA cv::COLOR_RGB2RGBA cv::COLOR_RGBA2RGB cv::COLOR_RGBA2GRAY
        print("cimg.shape", cimg.shape)

        lab_path = os.path.join(os.path.dirname(img_path), img_basename.split(".")[0] + "_label.png")
        print("lab_path", lab_path)
        lab = Image.open(lab_path)
        lab = np.asarray(lab)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
        print("lab.shape", lab.shape)  # (50141, 47161)
        assert lab.shape == cimg.shape[0:2] == img.shape[0:2]

        block_w = img.shape[0] // 10
        block_h = img.shape[1] // 10

        print("cutting ", img_basename)
        sample_cnt = 1
        while(sample_cnt <= sample_num) :
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
            # print(len(blak_block[blak_block==0])) # 黑色像素计数统计 (block_size*block_size)<0.3
            if img_block.max() > 0 and len(blak_block[blak_block == 0])/(block_w * block_h) < 0.3:  # 不是全0全黑 + 黑色背景比例不能超过0.3
                img_name = "img%02d" % int(img_basename.split(".")[0].split("_")[-1]) + "_w%04d"%(block_w) + "_h%04d"%(block_h) + "_cnt%03d"%(sample_cnt) + ".png"
                save_img_path = os.path.join(save_dir, img_name)
                assert img_block.shape == (block_w, block_h, 3)
                cv2.imwrite(save_img_path, img_block)
                print(save_img_path, "saved !")
                lab_name = "lab%02d" % int(img_basename.split(".")[0].split("_")[-1]) + "_w%04d"%(block_w) + "_h%04d"%(block_h) + "_cnt%03d"%(sample_cnt) + ".png"
                save_lab_path = os.path.join(save_dir, lab_name)
                assert lab_block.shape == (block_w, block_h)
                cv2.imwrite(save_lab_path, lab_block)
                print(save_lab_path, "saved !")
                sample_cnt += 1

    end_time = time.time()
    print("切割用时 : ", round((end_time - start_time), 3), "秒")
