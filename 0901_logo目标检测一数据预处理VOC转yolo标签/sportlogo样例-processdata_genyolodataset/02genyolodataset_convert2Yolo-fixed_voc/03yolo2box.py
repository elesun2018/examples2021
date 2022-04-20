#!/usr/bin/env python3
# coding=utf-8
import os
import shutil
from tqdm import tqdm
import cv2
import time

name_classes = ['puma', 'adidas', 'kappa', '361', 'xtep', 'anta', 'lining', 'erke', 'nb', 'nike']
def detection_result_visual(img_dir, label_dir, out_visual):
    """
    可视化数据集
    :param img_path: 原始图片文件
    :param label_path: 标签
    :param out_data_visual: 将yolo的标签可视化到图片文件上
    :return:
    """
    if os.path.exists(out_visual):
        shutil.rmtree(out_visual)
    os.mkdir(out_visual)
    time1 = time.time()
    label_lists = os.listdir(label_dir)
    for label_name in tqdm(label_lists) :
        # 获取图片路径，用于获取图像大小以及通道数
        label_path = os.path.join(label_dir, label_name)
        image_name = label_name.replace('.txt', '.jpg')
        image_path = os.path.join(img_dir, image_name)
        if not os.path.isfile(image_path) : continue
        # print("processing ",image_name)
        img = cv2.imread(image_path)
        img_h,img_w = img.shape[:2]
        fp = open(label_path,"r")
        for row in fp.read().splitlines():
            i,x,y,w,h = int(row.split()[0]),float(row.split()[1]),float(row.split()[2]),float(row.split()[3]),float(row.split()[4])
            label_name = name_classes[int(i)]
            xmin = int((x-w/2)*img_w)
            ymin = int((y-h/2)*img_h)
            xmax = int((x+w/2)*img_w)
            ymax = int((y+h/2)*img_h)
            cv2.rectangle(img, (xmin,ymin),(xmax,ymax),color=(0, 255, 255), thickness=2)
            # # 文字坐标
            word_x = xmin + 0
            word_y = ymin - 15
            cv2.putText(img, label_name, (word_x, word_y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        fp.close()
        save_img_path = os.path.join(out_dir, image_name)
        cv2.imwrite(save_img_path, img)
    time2 = time.time()
    print("time use {:.3f} s".format(time2 - time1))

if __name__ == '__main__':
    img_path = "VOCsport2yolo/images/trainval"
    label_path = "VOCsport2yolo/labels/trainval"
    out_dir = "yolo_vis_output"
    # 可视化数据集
    detection_result_visual(img_path, label_path, out_dir)
