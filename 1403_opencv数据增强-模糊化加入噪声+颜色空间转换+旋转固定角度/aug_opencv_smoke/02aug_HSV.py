# -*- coding: utf-8 -*-
"""
CV — 数据增强：yolov5 HSV色调变换
https://blog.csdn.net/pentiumCM/article/details/119145896
"""
import cv2
import numpy as np

def augment_hsv(image, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    HSV color-space augmentation
    :param image:       待增强的图片
    :param hgain:       HSV 中的 h 扰动系数，yolov5：0.015
    :param sgain:       HSV 中的 s 扰动系数，yolov5：0.7
    :param vgain:       HSV 中的 v 扰动系数，yolov5：0.4
    :return:
    """
    if hgain or sgain or vgain:
        # 随机取-1到1三个实数，乘以 hsv 三通道扰动系数
        # r:【1-gain ~ 1+gain】
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # cv2.split：通道拆分
        # h:[0~180], s:[0~255], v:[0~255]
        hue, sat, val = cv2.split(image_hsv)
        dtype = image.dtype  # uint8

        # 构建查找表
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # cv2.LUT：dst(I) = lut(src(I) + d)，d为常数0 / 128
        hue = cv2.LUT(hue, lut_hue)
        sat = cv2.LUT(sat, lut_sat)
        val = cv2.LUT(val, lut_val)

        # 通道合并
        image_hsv = cv2.merge((hue, sat, val)).astype(dtype)

        # 将 hsv 格式转为 BGR 格式
        image_dst = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        return image_dst
    else:
        return image
    
if __name__ == '__main__':
    image = cv2.imread('smoke/JPEGImages/1.jpg') # 1 759
    cv2.imshow('org_img', image)
    for idx in range(0, 10):
        img_hsv = augment_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4)
        cv2.imshow('img_hsv: %d' % idx, img_hsv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()