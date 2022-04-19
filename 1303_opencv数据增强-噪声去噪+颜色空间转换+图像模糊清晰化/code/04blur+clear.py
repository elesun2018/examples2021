"""
    opencv图片
"""
import os
import cv2
from cv2 import dnn_superres
import numpy as np
import random
import shutil

# 读取并保存
def filter_noise(input_dir, output_dir1, output_dir2):
    if os.path.exists(output_dir1):
        print(output_dir1,"exist and delete")
        shutil.rmtree(output_dir1)  # 递归删除文件夹
    print("makedirs:", output_dir1)
    os.makedirs(output_dir1)
    if os.path.exists(output_dir2):
        print(output_dir2,"exist and delete")
        shutil.rmtree(output_dir2)  # 递归删除文件夹
    print("makedirs:", output_dir2)
    os.makedirs(output_dir2)
    
    kernel = 5
    
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "models/EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 4)
    
    for filename in os.listdir(input_dir):
        path = os.path.join(input_dir, filename) # 获取文件路径
        print("doing... ", path)
        img = cv2.imread(path)#读取图片
        # 图像模糊化
        img_blur = cv2.blur(img, (kernel, kernel))  # 均值滤波
        # img_Gblur = cv2.GaussianBlur(noise_img, (kernel, kernel), 0)  # 高斯滤波
        cv2.imwrite(os.path.join(output_dir1, filename), img_blur)
        # 图像清晰化
        img_clear = sr.upsample(img_blur)
        cv2.imwrite(os.path.join(output_dir2, filename), img_clear)
        
# 程序入口
if __name__ == '__main__':
    input_dir = "images"    # 输入数据文件夹 images
    output_dir1 = "results-blur" # 输出数据文件夹
    output_dir2 = "results-clear" # 输出数据文件夹
    filter_noise(input_dir, output_dir1, output_dir2)

