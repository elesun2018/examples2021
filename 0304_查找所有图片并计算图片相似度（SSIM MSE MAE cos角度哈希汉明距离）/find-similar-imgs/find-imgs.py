# -*- coding: utf-8 -*-
'''
    查找所有图片
fun:
    遍历文件下所有图片并拷贝
ref:
    https://www.cnblogs.com/bigtreei/p/10683537.html
    https://blog.csdn.net/z772330927/article/details/103683461
'''
import os
import shutil
import time

def get_all_imgs_path(imgs_dir):
    img_list = []
    for fpathe, dirs, fs in os.walk(imgs_dir):
        for f in fs:
            if (f.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
                # print("img : ",f)
                img_path = os.path.join(fpathe, f)
                img_list.append(img_path)
    return img_list

def find_imgs_cp(imgs_dir, outdir) :
    if not os.path.isdir(imgs_dir) : raise ValueError(imgs_dir)
    img_list = get_all_imgs_path(imgs_dir)
    print("len img_list", len(img_list))
    # print("img_list", img_list)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    for img_path in img_list:
        print("img_path : ", img_path)
        save_path = os.path.join(outdir,os.path.basename(img_path))
        shutil.copyfile(img_path, save_path)

if __name__ == "__main__":
    imgs_dir = "images"
    outdir = "results"
    start_time = time.time()
    find_imgs_cp(imgs_dir, outdir)
    end_time = time.time()
    print("time use : ", round((end_time - start_time), 3), "s")
