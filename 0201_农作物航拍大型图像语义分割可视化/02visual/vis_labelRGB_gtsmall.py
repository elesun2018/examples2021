# -*- coding: utf-8 -*-
'''
小图可视化
label标签RGB可视化
https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.18.25ea4054aaYJei&postId=61355
'''
import os
import shutil
import time
import glob
import cv2
from PIL import Image
import numpy as np

lab_dir = "small/image_*label.png"

outdir = "BGR_small"
if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.makedirs(outdir)

lab_path_list = glob.glob(lab_dir)
start_time = time.time()
for path in lab_path_list :
    print("path", path)
    print("basename", os.path.basename(path))
    lab = Image.open(path)
    lab = np.asarray(lab)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
    print("lab.shape", lab.shape)

    B = lab.copy()   # 蓝色通道
    B[B == 1] = 255
    B[B == 2] = 0
    B[B == 3] = 0
    B[B == 0] = 0
    G = lab.copy()   # 绿色通道
    G[G == 1] = 0
    G[G == 2] = 255
    G[G == 3] = 0
    G[G == 0] = 0
    R = lab.copy()   # 红色通道
    R[R == 1] = 0
    R[R == 2] = 0
    R[R == 3] = 255
    R[R == 0] = 0

    lab_BGR = np.dstack((B,G,R)) # opencv需要使用BGR通道顺序
    # lab_BGR = cv2.resize(lab_BGR, None, fx= 0.1, fy=0.1)
    cv2.imwrite(os.path.join(outdir, os.path.basename(path)), lab_BGR)
    print("BGR label maped and saved in",outdir)

end_time = time.time()
print("标签BGR转换保存用时 : ",round((end_time-start_time),3),"秒")