# -*- coding: utf-8 -*-
'''
    图片相似度计算
fun:
    汉明距离表示两个（相同长度）字对应位不同的数量，向量相似度越高，对应的汉明距离越小。
ref:
    http://www.ruanyifeng.com/blog/2011/07/imgHash.txt
'''
import glob
import os
import sys
from functools import reduce
from PIL import Image

EXTS = 'jpg', 'jpeg', 'JPG', 'JPEG', 'gif', 'GIF', 'png', 'PNG'

def avhash(img_path):
    if not isinstance(img_path, Image.Image):
        im = Image.open(img_path)
    im = im.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, im.getdata()) / 64.
    return reduce(lambda x, y : x | (y[1] << y[0]),enumerate(map(lambda i: 0 if i < avg else 1, im.getdata())),0) # (y, z): x | (z << y) (y[0], y[1])

def hamming(h1, h2):
    h, d = 0, h1 ^ h2
    while d:
        h += 1
        d &= d - 1
    return h

if __name__ == '__main__':
    img_path = 'hat.png'
    h = avhash(img_path)
    imgs_dir = "images/hat/red"
    os.chdir(imgs_dir)
    images = []
    for ext in EXTS:
        images.extend(glob.glob('*.%s' % ext))
    seq = [] # 存储图片名称和汉明距离列表
    for f in images:
        seq.append((f, hamming(avhash(f), h)))
    for f, ham in sorted(seq, key=lambda i: i[1]):
        print("ham val :%d\t,img name :%s" % (ham, f))