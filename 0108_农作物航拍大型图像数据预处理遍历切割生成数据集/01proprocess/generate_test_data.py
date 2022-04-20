# -*- coding: utf-8 -*-
"""
从遥感大图中遍历切割区域生成测试数据集
"""
import cv2
import os
import glob
import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import shutil

if __name__=='__main__':
    block_size = 1024
    stride = int(block_size/2)
    img_dir = "../../../datasets/tianchi_jingwei/jingwei_round2_test_b_20190830/image_*[!label].png" # 注意修改路径
    # jingwei_round1_test_a_20190619 jingwei_round2_test_a_20190726 jingwei_round2_test_b_20190830
    outdir = "../../../results/sun/round2_test_b" # 注意修改路径 # round1_test_a round2_test_a round2_test_b
    # train_round1 train_round2
    save_img_dir = os.path.join(outdir, "images")
    if os.path.exists(save_img_dir):
        shutil.rmtree(save_img_dir)
    os.makedirs(save_img_dir)

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
        # 常量法，常数值图像四周填充
        # top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT
        cimg = cv2.copyMakeBorder(cimg, stride, stride, stride, stride, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        print("cimg.shape", cimg.shape)

        print("cutting ", img_basename)
        for row in range(img.shape[0] // stride): # 50141 row
            for col in range(img.shape[1] // stride): # 47161 col
                img_block = np.zeros((block_size,block_size,3),dtype=np.int)
                x0 = row * stride
                x1 = min(row * stride + block_size, img.shape[0])
                y0 = col * stride
                y1 = min(col * stride + block_size, img.shape[1])
                img_block[:cimg[x0:x1, y0:y1, :].shape[0], :cimg[x0:x1, y0:y1, :].shape[1],:cimg[x0:x1, y0:y1, :].shape[2]] = cimg[x0:x1, y0:y1, :]
                # img_block = cimg[row*stride:min(row*stride+block_size,img.shape[0]), col*stride:min(col*stride+block_size,img.shape[1]), :]
                blak_block = img_block[:,:,0] + img_block[:,:,1] + img_block[:,:,2]
                # print(len(blak_block[blak_block==0])) # 黑色像素计数统计 (block_size*block_size)<0.3
                if img_block.max() > 0 and len(blak_block[blak_block==0])/(block_size*block_size)<0.3: # 不是全0全黑 + 黑色背景比例不能超过0.3
                    img_name = "img%02d"%int(img_basename.split(".")[0].split("_")[-1])+ "_block%04d"%(block_size) + "_row%03d"%(row) + "_col%03d"%(col) + ".png"
                    save_img_path = os.path.join(save_img_dir, img_name)
                    assert img_block.shape == (block_size, block_size, 3)
                    cv2.imwrite(save_img_path, img_block)
                    print(save_img_path, "saved !")
    end_time = time.time()
    print("切割用时 : ", round((end_time - start_time), 3), "秒")
