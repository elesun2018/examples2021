"""
    opencv图片去噪滤波
https://blog.csdn.net/codedoctor/article/details/72500440
对于椒盐噪声，选择中值滤波器（Median Filter），在去掉噪声的同时，不会模糊图像
对于高斯噪声，选择均值滤波器（Mean Filter），能够去掉噪声，但会对图像造成一定的模糊。
"""
import os
import cv2
import numpy as np
import random
import shutil

# 读取并保存
def filter_noise(input_dir, output_dir, kernel):
    if os.path.exists(output_dir):
        print(output_dir,"exist and delete")
        shutil.rmtree(output_dir)  # 递归删除文件夹
    print("makedirs:", output_dir)
    os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        path = input_dir + "/" + filename # 获取文件路径
        print("doing... ", path)
        noise_img = cv2.imread(path)#读取图片
        
        # img_noise = gaussian_noise(noise_img, 0, 0.12) # 高斯噪声
        # filter_img = cv2.blur(noise_img, (kernel, kernel))  # 均值滤波
        # filter_img = cv2.GaussianBlur(noise_img, (kernel, kernel), 0)  # 高斯滤波 xxx???
        # img_noise = sp_noise(noise_img,0.025)# 椒盐噪声
        filter_img = cv2.medianBlur(noise_img, kernel)  # 中值滤波
        # img_noise  = random_noise(noise_img,500)# 随机噪声
        # ++++++ ???
        
        cv2.imwrite(output_dir+'/'+filename,filter_img)
# 程序入口
if __name__ == '__main__':
    input_dir = "results-sp_noise"    # 输入数据文件夹 gaussian_noise sp_noise
    output_dir = "results-filtersp" # 输出数据文件夹 filtergaussian filtersp
    kernel = 5
    filter_noise(input_dir, output_dir, kernel)

