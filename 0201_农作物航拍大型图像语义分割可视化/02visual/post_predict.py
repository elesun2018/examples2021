# -*- coding: utf-8 -*-
'''
读取图像分割地块统计类别面积产量并可视化
图像比例尺： 1pix:35mm 0.035m
1亩地=666.67平方米=544220pix
类别	  像素值	颜色        R     G     B     亩产/kg   产量系数kg/10000pix
烤烟	  1	        蓝色        0     0     255   200        3.67
玉米	  2	        绿色        0     255   0     500        9.18
薏仁米	  3	        红色        255   0     0     200        3.67
建筑      4         黄色        255   255   0     1          1
其他	  0	        黑色        0     0     0     1          1
'''
import os
import shutil
import time
import cv2
from PIL import Image
import numpy as np
from collections import Counter
Image.MAX_IMAGE_PIXELS = None

def postprocess(img_path, lab_path, outdir):
    if not os.path.exists(img_path):
        raise ValueError(img_path,"not exist !")
    if not os.path.exists(lab_path):
        raise ValueError(lab_path,"not exist !")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    start_time = time.time()
    # print("img_path", img_path)
    print("图像路径：", img_path)
    img = Image.open(img_path)
    img = np.asarray(img)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
    # print("img.shape", img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # COLOR_RGB2BGR COLOR_RGBA2BGRA opencv需要使用BGR通道顺序
    # print("img.shape", img.shape)

    # print("lab_path", lab_path)
    lab = Image.open(lab_path)
    lab = np.asarray(lab)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
    # print("lab.shape", lab.shape)

    assert img.shape[0:2] == lab.shape
    print("图像区域长：{0:.2f}m；宽：{1:.2f}m；合计面积：{2:.2f}㎡".format(lab.shape[1]*rate,lab.shape[0]*rate,lab.shape[0]*rate*lab.shape[1]*rate))
    # 像素统计
    dict = Counter(lab.ravel())
    # print("Counter dict :",dict)
    # 遍历键和值
    for key, value in dict.items():
        # print(key,":",value)
        dict[key] = [name_list[key],color_list[key],value]
    for key, value in dict.items():
        print(key,":",value)
        # print(dict[key][0],":",dict[key][2])
        if key==0 or key==4 : continue # 其他背景和建筑不同计算
        print("颜色：{0}；类别：{1}；像素量：{2}pix；面积：{3:.2f}㎡；预估产量：{4:.2f}Kg".format(
            dict[key][1],dict[key][0],dict[key][2],dict[key][2]*rate*rate,dict[key][2]/10000*coe_list[key]))

    # 可视化颜色映射
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
    # 可视化
    lab_BGR = np.dstack((B, G, R))  # opencv需要使用BGR通道顺序
    # print("lab_BGR.shape", lab_BGR.shape)
    # print("img\n",img)
    # print("lab_BGR\n", lab_BGR)
    alpha = 0.5 # 权重透明度
    overlap = cv2.addWeighted(img, alpha, lab_BGR, 1 - alpha, 0, dtype = cv2.CV_32F)
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(outdir, img_name), overlap,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 注意修改 可视化img 的路径
    print("完成图像处理可视化并将输出文件保存在", outdir)
    del lab,img
    end_time = time.time()
    print("预测用时:", round((end_time - start_time), 3), "秒")

if __name__ == "__main__":
    rate = 0.035
    print("图像比例尺为：1pix:{}m".format(rate))
    name_list = ["其他","烤烟","玉米","水稻","建筑"] # ["other","tobacco","corn","rice","building"]
    color_list = ["黑色","蓝色","绿色","红色","黄色"] # ["black","blue","green","red","yellow"] #其中“烤烟”像素值 1，“玉米”像素值 2，“薏仁米”像素值 3，“人造建筑”像素值 4，其余所有位置视为“其他”像素值 0
    coe_list = [1 , 3.67, 9.18, 3.67, 1 ]
    img_path = "../../../results/sun/samples/round2_train/img10_w3360_h4434_cnt023.png" # "small/image_10.png"  # 注意修改路径
    lab_path = "../../../results/sun/samples/round2_train/lab10_w3360_h4434_cnt023.png" # "small/image_10_label.png"  # 注意修改路径
    outdir  =  "result_post"  # 注意修改路径
    postprocess(img_path, lab_path, outdir)


