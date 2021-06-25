# -*- coding: utf-8 -*-
'''
    查找所有图片并进行图片相似度计算
fun:

ref:
    https://blog.csdn.net/u010977034/article/details/82733137/
    https://blog.csdn.net/weixin_42769131/article/details/85230295

'''
import os
import shutil
import time
import numpy as np
from PIL import Image
from skimage.measure import compare_ssim
from sklearn.metrics import mean_absolute_error,mean_squared_error
from numpy import average, linalg, dot
from functools import reduce

def get_all_imgs_path(imgs_dir):
    img_list = []
    for fpathe, dirs, fs in os.walk(imgs_dir):
        for f in fs:
            if (f.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
                # print("img : ",f)
                img_path = os.path.join(fpathe, f)
                img_list.append(img_path)
    return img_list

# cosin相似度（余弦相似度）
def cos_similarity_imgs(image1, image2):
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res

# 汉明距离表示两个图片相似度
def hanming_similarity_imgs(img_in, img):
    img_in = img_in.resize((8, 8), Image.ANTIALIAS)
    avg = reduce(lambda x, y: x + y, img_in.getdata()) / 64.
    hash_value_in = reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img_in.getdata())),0)

    img = img.resize((8, 8), Image.ANTIALIAS)
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    hash_value = reduce(lambda x, y: x | (y[1] << y[0]),
                        enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)
    # 计算汉明距离
    distance = bin(hash_value_in ^ hash_value).count('1')
    print("hanming distance",distance)
    similary = 1 - distance / max(len(bin(hash_value_in)), len(bin(hash_value)))
    return similary

def cal_similar_img(input_img_path, imgs_dir, outdir, count=3, thred=0.0) :
    if not os.path.isfile(input_img_path) : raise ValueError(input_img_path)
    if not (input_img_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))): raise ValueError("not img file")
    img_in = Image.open(input_img_path).convert('L') # pil读入gray通道
    img_in_rs = img_in.resize((img_in.width//2, img_in.height//2), Image.ANTIALIAS) # 压缩忽略细节
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
        img = Image.open(img_path).convert('L') # pil读入gray通道
        img_rs = img.resize((img_in.width//2, img_in.height//2),Image.ANTIALIAS)  # img.resize((width, height),Image.ANTIALIAS)
        img_rs_np = np.asarray(img_rs) # val 0-255
        # print("img_rs_np.shape", img_rs_np.shape)
        # img_rs_np = np.resize(img_np, img_in_rs_np.shape)
        # print("img_rs_np.shape", img_rs_np.shape)

        assert img_rs_np.shape == img_in_rs_np.shape
        # similar = compare_ssim(img_in_rs_np, img_rs_np)
        # similar = mean_squared_error(img_in_rs_np, img_rs_np)
        # similar = mean_absolute_error(img_in_rs_np, img_rs_np)
        # similar = cos_similarity_imgs(img_in_rs, img_rs)
        similar = hanming_similarity_imgs(img_in, img)
        print("similar",similar)
        if similar >= thred :
            similar_val_list.append(similar)
            similar_img_list.append(img_path)
    for i in np.argsort(-np.asarray(similar_val_list))[:min(count,len(similar_val_list))] :
        save_path = os.path.join(outdir, "sim%.3f"%(similar_val_list[i]) + "_nam" + os.path.basename(similar_img_list[i]))
        shutil.copyfile(similar_img_list[i], save_path)

if __name__ == "__main__":
    input_img_path = "hat.png" # car hat /home/wk/datasets/od/VOC2yolohat/images/test/00000001.jpg
    imgs_dir = "images" # images cars
    outdir = "results_hat2" #
    start_time = time.time()
    cal_similar_img(input_img_path, imgs_dir, outdir, count=5, thred=0.3)
    end_time = time.time()
    print("time use : ", round((end_time - start_time), 3), "s")
