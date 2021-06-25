# -*- coding: utf-8 -*-
'''
小图批量可视化
label标签RGB可视化
https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.18.25ea4054aaYJei&postId=61355

类别	  像素值	颜色        B     G     R    | 颜色        R     G     B
烤烟	  1	        棕色br     165， 42， 42     | 蓝色        0     0     255
玉米	  2	        黄色ye     255，255， 0      | 绿色        0     255   0
薏仁米	  3	        白色wh     255，255，255     | 红色        255   0     0
建筑      4         灰色gr     190，190，190     | 黄色        255   255   0
其他	  0	        黑色bl     0  ，  0， 0      | 黑色        0     0     0
'''
import os
import shutil
import time
import glob
import cv2
from PIL import Image
import numpy as np
from collections import Counter
Image.MAX_IMAGE_PIXELS = None


def visual(lab_dir, img_dir, outdir):
    save_data_dir = os.path.join(outdir, "RGB_small")
    if os.path.exists(save_data_dir):
        shutil.rmtree(save_data_dir)
    os.makedirs(save_data_dir)
    lab_lists = os.listdir(lab_dir)
    # 排序
    # lab_lists = sorted(mask_path, key=lambda row: int(row.split('/')[-1].split('.')[0].split('_')[1]))
    # print("lab_lists",lab_lists)
    start_time = time.time()
    for lab_name in lab_lists:
        lab_path = os.path.join(lab_dir, lab_name)
        print("lab_path", lab_path)
        lab = Image.open(lab_path)
        lab = np.asarray(lab)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
        print("lab.shape", lab.shape)

        # 简单方法
        # print("all num of pix2 : ",np.sum(lab == 2, axis=None))  # axis=None 所有;axis=0, 按列相加;axis=1, 按行相加
        # Counter（计数器）：用于追踪值的出现次数
        dict = Counter(lab.ravel())
        # print("Counter dict :",dict)
        # 遍历键和值
        for key, value in dict.items():
            # print(key,":",value)
            dict[key] = [name_list[key],color_list[key],value]
        for key, value in dict.items():
            print(key,":",value)
            # print(dict[key][0],":",dict[key][2])
        img_name = lab_name # "img" + lab_name.split("lab")[-1] # img02_block1024_row088_col139.png
        img_path = os.path.join(img_dir, img_name)
        print("img_path", img_path)
        img = Image.open(img_path)
        img = np.asarray(img)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
        print("img.shape", img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # COLOR_RGB2BGR COLOR_RGBA2BGRA opencv需要使用BGR通道顺序
        print("img.shape", img.shape)
        assert img.shape[0:2] == lab.shape
        # 其他	  0	        黑色bl     0  ，  0，  0
        R = np.zeros(lab.shape, np.int)
        G = np.zeros(lab.shape, np.int)
        B = np.zeros(lab.shape, np.int)
        # 烤烟	    1	      棕色br     165， 42， 42
        R[lab == 1] =   0; G[lab == 1] =   0; B[lab == 1] = 255
        # 玉米      2         黄色ye     255，255， 0
        R[lab == 2] =   0; G[lab == 2] = 255; B[lab == 2] =   0
        # 薏仁米	3	      白色wh     255，255，255
        R[lab == 3] = 255; G[lab == 3] =   0; B[lab == 3] =   0
        # 建筑      4         灰色gr     190，190，190  深灰色 130, 130, 130
        R[lab == 4] = 255; G[lab == 4] = 255; B[lab == 4] =   0

        lab_BGR = np.dstack((B, G, R))  # opencv需要使用BGR通道顺序
        print("lab_BGR.shape", lab_BGR.shape)
        # print("cimg\n",cimg)
        # print("lab_BGR\n", lab_BGR)
        alpha = 0.5 # 权重透明度
        overlap = cv2.addWeighted(img, alpha, lab_BGR, 1 - alpha, 0, dtype = cv2.CV_32F)

        cv2.imwrite(os.path.join(save_data_dir, img_name), overlap,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 注意修改 可视化img 的路径
        print("source image and lab resized maped saved in", save_data_dir)
        del lab,img
    end_time = time.time()
    print("图像标签颜色保存用时 : ", round((end_time - start_time), 3), "秒")


if __name__ == "__main__":
    name_list = ["other","tobacco","corn","rice","building"]
    color_list = ["black","blue","green","red","yellow"] #其中“烤烟”像素值 1，“玉米”像素值 2，“薏仁米”像素值 3，“人造建筑”像素值 4，其余所有位置视为“其他”像素值 0
    lab_dir = "../../../results/lin_tcny/Crop1024/label"  # 注意修改路径
    img_dir = "../../../results/lin_tcny/Crop1024/image"  # 注意修改路径
    outdir = "../../../results/lin_tcny/visual"  # 注意修改路径
    visual(lab_dir, img_dir, outdir)


