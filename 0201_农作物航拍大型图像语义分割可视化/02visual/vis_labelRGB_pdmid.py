# -*- coding: utf-8 -*-
'''
大图可视化
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
import re
import glob
import cv2
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None


def visual(lab_dir, img_dir, outdir):
    save_data_dir = os.path.join(outdir, "RGB_linpdmid")
    if os.path.exists(save_data_dir):
        shutil.rmtree(save_data_dir)
    os.makedirs(save_data_dir)
    lab_path_list = glob.glob(lab_dir)

    start_time = time.time()
    for lab_path in lab_path_list:
        print("lab_path", lab_path)
        print("lab_name", os.path.basename(lab_path))
        lab = Image.open(lab_path)
        lab = np.asarray(lab)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
        print("lab.shape", lab.shape)
        clab = cv2.resize(lab, None, fx=0.1, fy=0.1)  # 缩小 0.1 * 0.1
        print("clab.shape", clab.shape)
        del lab
        # 根据不同的文件名称做修改匹配img_name
        # os.path.basename(lab_path).split("_label")[0] + ".png"
        # test_image10_pipeline_predict.png -> image_10.png
        str_temp = os.path.basename(lab_path).split("_")[1] # image10
        img_name = "image_" + re.findall("\d+", str_temp)[0] + ".png"
        img_path = os.path.join(img_dir, img_name)
        print("img_path", img_path)
        img = Image.open(img_path)
        img = np.asarray(img)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
        print("img.shape", img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # COLOR_RGB2BGR COLOR_RGBA2BGRA opencv需要使用BGR通道顺序
        cimg = cv2.resize(img, None, fx=0.1, fy=0.1)  # 缩小 0.1 * 0.1
        print("cimg.shape", cimg.shape)
        del img
        assert cimg.shape[0:2] == clab.shape
        # 其他	  0	        黑色bl     0  ，  0，  0
        R = np.zeros(clab.shape, np.int)
        G = np.zeros(clab.shape, np.int)
        B = np.zeros(clab.shape, np.int)
        # 烤烟	    1	      棕色br     165， 42， 42
        R[clab == 1] =   0; G[clab == 1] =   0; B[clab == 1] = 255
        # 玉米      2         黄色ye     255，255， 0
        R[clab == 2] =   0; G[clab == 2] = 255; B[clab == 2] =   0
        # 薏仁米	3	      白色wh     255，255，255
        R[clab == 3] = 255; G[clab == 3] =   0; B[clab == 3] =   0
        # 建筑      4         灰色gr     190，190，190  深灰色 130, 130, 130
        R[clab == 4] = 255; G[clab == 4] = 255; B[clab == 4] =   0

        lab_BGR = np.dstack((B, G, R))  # opencv需要使用BGR通道顺序
        print("lab_BGR.shape", lab_BGR.shape)
        # print("cimg\n",cimg)
        # print("lab_BGR\n", lab_BGR)
        alpha = 0.5 # 权重透明度
        overlap = cv2.addWeighted(cimg, alpha, lab_BGR, 1 - alpha, 0, dtype = cv2.CV_32F)

        cv2.imwrite(os.path.join(save_data_dir, img_name), overlap,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 注意修改 可视化img 的路径
        print("source image and lab resized maped saved in", save_data_dir)
        del clab,cimg
    end_time = time.time()
    print("图像标签压缩颜色保存用时 : ", round((end_time - start_time), 3), "秒")


if __name__ == "__main__":
    lab_dir = "../../../results/lin_tcny/lin_pd/20210105_deeplabv3plus_resnet101_StepLR_Adam_temp/test*predict.png"  # 注意修改路径 jingwei_round2_train_20190726  jingwei_round1_train_20190619
    img_dir = "../../../datasets/tianchi_jingwei/all"  # 注意修改路径
    outdir = "../../../results/sun/visual"  # 注意修改路径
    visual(lab_dir, img_dir, outdir)


