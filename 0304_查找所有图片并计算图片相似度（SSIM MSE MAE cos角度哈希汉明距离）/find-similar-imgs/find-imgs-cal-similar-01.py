# -*- coding: utf-8 -*-
'''
    查找所有图片并进行图片相似度计算
fun:
    SSIM（结构相似性度量）计算图片相似度
    基于直方图 计算图片相似度
    基于互信息 计算图片相似度
ref:
    https://blog.csdn.net/u010977034/article/details/82733137/
    https://blog.csdn.net/weixin_42769131/article/details/85230295
    numpy中的argsort()排序
        https://blog.csdn.net/m0_37712157/article/details/81433910
        https://blog.csdn.net/Python798/article/details/81138040
'''
import os
import shutil
import time
import numpy as np
from PIL import Image
from skimage.measure import compare_ssim
from sklearn.metrics import mutual_info_score

def get_all_imgs_path(imgs_dir):
    img_list = []
    for fpathe, dirs, fs in os.walk(imgs_dir):
        for f in fs:
            if (f.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
                # print("img : ",f)
                img_path = os.path.join(fpathe, f)
                img_list.append(img_path)
    return img_list

def cal_similar_img(input_img_path, imgs_dir, outdir, count=3, thred=0.0) :
    if not os.path.isfile(input_img_path) : raise ValueError(input_img_path)
    if not (input_img_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))): raise ValueError("not img file")
    img_in = Image.open(input_img_path) # pil读入RGB通道
    img_in_rs = img_in.resize((img_in.width//10, img_in.height//10), Image.ANTIALIAS) # 压缩忽略细节
    img_in_rs_np = np.asarray(img_in_rs) # val 0-255
    print("img_in_rs_np.shape", img_in_rs_np.shape)
    if not os.path.isdir(imgs_dir) : raise ValueError(imgs_dir)
    img_list = get_all_imgs_path(imgs_dir)
    print("len img_list", len(img_list))
    # print("img_list", img_list)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    similar_val_list = []
    similar_img_list = []
    for img_path in img_list:
        print("img_path : ", img_path)
        img = Image.open(img_path) # pil读入RGB通道
        img_rs = img.resize((img_in.width//10, img_in.height//10),Image.ANTIALIAS)  # img.resize((width, height),Image.ANTIALIAS)
        img_rs_np = np.asarray(img_rs) # val 0-255
        print("img_rs_np.shape", img_rs_np.shape)
        # img_rs_np = np.resize(img_np, img_in_rs_np.shape)
        print("img_rs_np.shape", img_rs_np.shape)
        # SSIM（结构相似性度量）
        # assert img_rs_np.shape == img_in_rs_np.shape
        # similar = compare_ssim(img_in_rs_np, img_rs_np, multichannel=True)
        # 基于直方图
        # assert len(img_rs.histogram()) == len(img_in_rs.histogram())
        # similar = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(img_rs.histogram(), img_in_rs.histogram())) / len(img_in_rs.histogram())
        # 基于互信息（Mutual Information）
        similar = mutual_info_score(np.reshape(img_in_rs_np, -1), np.reshape(img_rs_np, -1))

        if similar >= thred :
            similar_val_list.append(similar)
            similar_img_list.append(img_path)
    for i in np.argsort(-np.asarray(similar_val_list))[:min(count,len(similar_val_list))] :
        save_path = os.path.join(outdir, "sim%.3f"%(similar_val_list[i]) + "_nam" + os.path.basename(similar_img_list[i]))
        shutil.copyfile(similar_img_list[i], save_path)

if __name__ == "__main__":
    input_img_path = "hat.png" # car hat
    imgs_dir = "images" # images cars
    outdir = "results_hat"
    start_time = time.time()
    cal_similar_img(input_img_path, imgs_dir, outdir, count=5, thred=0.4)
    end_time = time.time()
    print("time use : ", round((end_time - start_time), 3), "s")
