# -*- coding: utf-8 -*-
'''
小图批量可视化
maskel标签RGB可视化
https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.18.25ea4054aaYJei&postId=61355
farmland forest grass road urban_area countryside industrial_land construction water bareland
#                                          R  G  B          颜色        像素值   类别
R[mask == x],G[mask == x],B[mask == x] = 127,255,  0,   # Chartreuse       1: "farmland"
R[mask == x],G[mask == x],B[mask == x] =  34,139, 34,   # ForestGreen      2: "forest"
R[mask == x],G[mask == x],B[mask == x] =   0,255,  0,   # Green        绿色 3: "grass"
R[mask == x],G[mask == x],B[mask == x] = 255,255,255,   # White            4: "road"
R[mask == x],G[mask == x],B[mask == x] = 255,  0,  0,   # Red              5: "urban_area"
R[mask == x],G[mask == x],B[mask == x] = 255,165,  0,   # Orange           6: "countryside"
R[mask == x],G[mask == x],B[mask == x] = 165, 42, 42,   # Brown            7: "industrial_land"
R[mask == x],G[mask == x],B[mask == x] = 190,190,190,   # Gray     灰色     8: "construction"
R[mask == x],G[mask == x],B[mask == x] = 255,255,  0,   # Yellow           9: "water"
R[mask == x],G[mask == x],B[mask == x] = 255,255,224,   # LightYellow     10: "bareland"
'''
import os
import shutil
import time
import glob
import cv2
from PIL import Image
import numpy as np
from collections import Counter
import datetime
curr_time = datetime.datetime.now()
time_str = datetime.datetime.strftime(curr_time,'%m%d%H%M')

def process_visual(data_dir, out_dir):
    save_vis_out_dir = os.path.join(out_dir, "visual_"+time_str)
    if os.path.exists(save_vis_out_dir):
        shutil.rmtree(save_vis_out_dir)
    os.makedirs(save_vis_out_dir)
    save_dat_out_dir = os.path.join(out_dir, "data_" + time_str)
    if os.path.exists(save_dat_out_dir):
        shutil.rmtree(save_dat_out_dir)
    os.makedirs(save_dat_out_dir)
    ann_dict = {
        0 : ["none", "balance"],
        1 : ["Chartreuse", "farmland"],
        2 : ["ForestGreen", "forest"],
        3 : ["Green", "grass"],
        4 : ["White", "road"],
        5 : ["Red", "urban_area"],
        6 : ["Orange", "countryside"],
        7 : ["Brown", "industrial_land"],
        8 : ["Gray", "construction"],
        9 : ["Yellow", "water"],
        10: ["LightYellow", "bareland"]
    }
    img_lists = os.listdir(data_dir)
    img_lists = [i for i in img_lists if i.endswith(".tif")]
    print("length img_lists : ",len(img_lists))
    # 排序
    # img_lists = sorted(mask_path, key=lambda row: int(row.split('/')[-1].split('.')[0].split('_')[1]))
    # print("img_lists",img_lists)
    start_time = time.time()
    for img_name in img_lists:
        img_path = os.path.join(data_dir, img_name)
        print("img_path", img_path)
        img = Image.open(img_path)
        img = np.asarray(img)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
        print("img.shape", img.shape) # (256, 256, 4)
        # print("values in img : ", np.unique(img)) # 0-255

        mask_name = img_name.split('.')[0] + ".png"
        mask_path = os.path.join(data_dir, mask_name)
        # print("mask_path", mask_path)
        mask = Image.open(mask_path)
        mask = np.asarray(mask)  # array仍会copy出一个副本，占用新的内存，但asarray不会。
        # print("mask.shape", mask.shape)
        # print("values in mask : ",np.unique(mask)) # [1 2 3 4 6 9]
        if 0 in np.unique(mask) : print(mask_name,"have 0 value pix") # 没有0值的像素存在
        if len(np.unique(mask)) <=2 : print(mask_name,"have less class ",len(np.unique(mask)))
        pix_dict = Counter(mask.ravel())
        max_pix_nums = max(pix_dict.values())  # 以列表返回字典中的所有值0
        text_str_list = [] # 写入图片的文字列表
        for key, value in pix_dict.items(): # 以列表返回可遍历的(键, 值) 元组数组
            print("key:{0},cls:{1},clr:{2},value:{3}".format(key,ann_dict[key][1],ann_dict[key][0],value))
            text_str_list.append("key:{0},cls:{1},clr:{2},value:{3}".format(key,ann_dict[key][1],ann_dict[key][0],value))
            if value == max_pix_nums :
                main_key = key
        if pix_dict[main_key] > mask.shape[0]*mask.shape[0]*0.5: # 如果主要的像素数量占据了整张图的x%
            save_vis_cls_dir = os.path.join(save_vis_out_dir,ann_dict[main_key][1])
            save_dat_cls_dir = os.path.join(save_dat_out_dir,ann_dict[main_key][1])
        else :
            save_vis_cls_dir = os.path.join(save_vis_out_dir,ann_dict[0][1])
            save_dat_cls_dir = os.path.join(save_dat_out_dir,ann_dict[0][1])
        if not os.path.exists(save_vis_cls_dir):
            os.makedirs(save_vis_cls_dir)
        if not os.path.exists(save_dat_cls_dir):
            os.makedirs(save_dat_cls_dir)
        shutil.copy(img_path, save_dat_cls_dir)
        shutil.copy(mask_path, save_dat_cls_dir)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # COLOR_RGB2BGR COLOR_RGBA2BGRA opencv需要使用BGR通道顺序
        img = img[:,:,[2,1,3]] # RGBNir - BGNir
        print("img.shape", img.shape)  # (256, 256, 4)
        assert img.shape[0:2] == mask.shape

        # 其他	  0	        黑色bl     0  ，  0，  0
        R = np.zeros(mask.shape, np.int)
        G = np.zeros(mask.shape, np.int)
        B = np.zeros(mask.shape, np.int)
        #                                          R  G  B          颜色        像素值   类别
        R[mask == 1], G[mask == 1], B[mask == 1] = 127, 255, 0,  # Chartreuse       1: "耕地"
        R[mask == 2], G[mask == 2], B[mask == 2] = 34, 139, 34,  # ForestGreen      2: "林地"
        R[mask == 3], G[mask == 3], B[mask == 3] = 0, 255, 0,  # Green        绿色 3: "草地"
        R[mask == 4], G[mask == 4], B[mask == 4] = 255, 255, 255,  # White            4: "道路"
        R[mask == 5], G[mask == 5], B[mask == 5] = 255, 0, 0,  # Red              5: "城镇建设用地"
        R[mask == 6], G[mask == 6], B[mask == 6] = 255, 165, 0,  # Orange           6: "农村建设用地"
        R[mask == 7], G[mask == 7], B[mask == 7] = 165, 42, 42,  # Brown            7: "工业用地"
        R[mask == 8], G[mask == 8], B[mask == 8] = 190, 190, 190,  # Gray     灰色     8: "构筑物"
        R[mask == 9], G[mask == 9], B[mask == 9] = 255, 255, 0,  # Yellow           9: "水域"
        R[mask ==10], G[mask ==10], B[mask ==10] = 255, 255, 224,  # LightYellow     10: "裸地"

        mask_BGR = np.dstack((B, G, R))  # opencv需要使用BGR通道顺序
        print("mask_BGR.shape", mask_BGR.shape)
        # print("cimg\n",cimg)
        # print("mask_BGR\n", mask_BGR)
        alpha = 0.5 # 权重透明度
        overlap = cv2.addWeighted(img, alpha, mask_BGR, 1 - alpha, 0, dtype = cv2.CV_32F)
        save_img_path = os.path.join(save_vis_cls_dir, mask_name)
        for i,text_str in enumerate(text_str_list) :
            overlap = cv2.putText(overlap, text_str, (10, 10*(i+1)), cv2.FONT_ITALIC, 0.3, (255, 255, 255), 1)
            # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度 cv2.FONT_ITALIC cv2.FONT_HERSHEY_SIMPLEX
        cv2.imwrite(save_img_path, overlap,[int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 注意修改 可视化img 的路径
        print("source image and mask resized maped saved in", save_img_path)
        del mask,img
    end_time = time.time()
    print("图像标签颜色保存用时 : ", round((end_time - start_time), 3), "秒")


if __name__ == "__main__":
    data_dir = "../../../datasets/seg/lishuitianchi/suichang_round1_train_210120"  # 注意修改路径
    out_dir = "../../../results/lishui/sun"  # 注意修改路径
    process_visual(data_dir, out_dir)


