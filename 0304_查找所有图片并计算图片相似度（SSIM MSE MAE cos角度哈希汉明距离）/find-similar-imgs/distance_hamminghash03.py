# -*- coding: utf-8 -*-
'''
    图片相似度计算
fun:
    汉明距离表示两个（相同长度）字对应位不同的数量，向量相似度越高，对应的汉明距离越小。
ref:
    https://zhuanlan.zhihu.com/p/88869743

'''
# 比较两张图片的相似度
from PIL import Image
from functools import reduce
import time

# 计算Hash
def phash(img):
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    return reduce(
        lambda x, y: x | (y[1] << y[0]),
        enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())),
        0)
# 计算汉明距离
def hamming_distance(a, b):
    print("hamming_distance",bin(a ^ b).count('1'))
    return bin(a ^ b).count('1')
# 计算图片相似度
def is_imgs_similar(img1, img2):
    return True if hamming_distance(phash(img1), phash(img2)) <= 10 else False # elesun 5

if __name__ == '__main__':
    img1_path = "hat.png"
    img2_path = "images/hat/red/hathat.png" # images/coco/000000000025.jpg images/hat/red/00000001.jpg  images/hat/yellow/00000476.jpg hat.png  images/hat/red/hathat.png
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    start_time = time.time()
    similar = is_imgs_similar(img1, img2)
    end_time = time.time()
    print("similar ? : ",similar)
    print("time use : ", round((end_time - start_time), 3), "s")