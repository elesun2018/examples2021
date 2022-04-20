# -*- coding: utf-8 -*-
'''
原图压缩保存+label标签压缩保存
用pillow + opencv操作
https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.18.25ea4054aaYJei&postId=61355
'''
import os
import shutil
import glob
import cv2
import time
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None

img_dir = "../../../datasets/tianchi_jingwei/jingwei_*/image_*[!label].png" # 注意修改路径
lab_dir = "../../../datasets/tianchi_jingwei/jingwei_*/image_*label.png" # 注意修改路径
outdir = "../../../results/visual" # 注意修改路径
save_data_dir = os.path.join(outdir, "small")
if os.path.exists(save_data_dir):
    shutil.rmtree(save_data_dir)
os.makedirs(save_data_dir)

img_path_list = glob.glob(img_dir)
start_time = time.time()
for path in img_path_list :
    print("path",path)
    print("basename",os.path.basename(path))
    img = Image.open(path)
    img = np.asarray(img) # array仍会copy出一个副本，占用新的内存，但asarray不会。
    print("img.shape", img.shape) # (50141, 47161, 4)
    cimg = cv2.resize(img, None, fx=0.1, fy=0.1) # 缩小 0.1 * 0.1
    print("cimg.shape", cimg.shape) # (50141, 47161, 4)
    cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR) # COLOR_RGB2BGR COLOR_RGBA2BGRA opencv需要使用BGR通道顺序
    print("cimg.shape", cimg.shape)
    print("source image resized and saved in", save_data_dir)
    del img,cimg
end_time = time.time()
print("图像压缩保存用时 : ",round((end_time-start_time),3),"秒")

lab_path_list = glob.glob(lab_dir)
start_time = time.time()
for path in lab_path_list :
    print("path", path)
    print("basename", os.path.basename(path))
    img = Image.open(path)
    img = np.asarray(img)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
    print("img.shape", img.shape)
    cimg = cv2.resize(img, None, fx=0.1, fy=0.1) # 缩小 0.1 * 0.1
    print("cimg.shape", cimg.shape)
    cv2.imwrite(os.path.join(save_data_dir, os.path.basename(path)), cimg,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 注意修改 可视化img 的路径
    print("source image resized and saved in", save_data_dir)
end_time = time.time()
print("标签压缩保存用时 : ",round((end_time-start_time),3),"秒")



