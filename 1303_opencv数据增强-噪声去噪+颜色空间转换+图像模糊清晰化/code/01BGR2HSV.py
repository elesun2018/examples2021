"""
    opencv图片轮廓边缘

https://blog.csdn.net/weixin_44414948/article/details/106077973
"""
import os
import cv2
import shutil
import numpy as np

img_path = 'imgs/Lena.png'
out_dir = "results-hsv"
h_step = 10 #色调 步长
s_step = 20 # 饱和度 步长
v_step = 20 # 亮度 步长

if os.path.exists(out_dir):
    print(out_dir,"exist and delete")
    shutil.rmtree(out_dir)  # 递归删除文件夹
print("makedirs:", out_dir)
os.makedirs(out_dir)

img=cv2.imread(img_path, cv2.IMREAD_COLOR)
# 通过cv2.cvtColor把图像从BGR转换到HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 色调变化范围 [0， 179]
img_h = img_hsv.copy()
num = 0
for count in range(-30, 91, h_step): # -180, 180  -90, 90
    img_h[:, :, 0] = img_h[:, :, 0] + h_step*count
    img_h[:, :, 0] = np.clip(img_h[:, :, 0], 0, 179)
    img_save = cv2.cvtColor(img_h, cv2.COLOR_HSV2BGR)
    num = num + 1
    save_path = os.path.join(out_dir, 'H_%02d_%02d.jpg'%(num, count))
    cv2.imwrite(save_path, img_save)
    print("successfully save ",save_path)

# 饱和度变化范围 [0， 255]
img_s = img_hsv.copy()
num = 0
for count in range(-80, 120, s_step): # -256, 256  -128, 128
    img_s[:, :, 1] = img_s[:, :, 1] + h_step*count
    img_s[:, :, 0] = np.clip(img_s[:, :, 0], 0, 255)
    img_save = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)
    num = num + 1
    save_path = os.path.join(out_dir, 'S_%02d_%02d.jpg'%(num, count))
    cv2.imwrite(save_path, img_save)
    print("successfully save ",save_path)

# 亮度变化范围 [0， 255]
img_v = img_hsv.copy()
num = 0
for count in range(-80, 80, v_step): # -256, 256  -128, 128
    img_v[:, :, 2] = img_v[:, :, 2] + h_step*count
    img_v[:, :, 0] = np.clip(img_v[:, :, 0], 0, 255)
    img_save = cv2.cvtColor(img_v, cv2.COLOR_HSV2BGR)
    num = num + 1
    save_path = os.path.join(out_dir, 'V_%02d_%02d.jpg'%(num, count))
    cv2.imwrite(save_path, img_save)
    print("successfully save ",save_path)
