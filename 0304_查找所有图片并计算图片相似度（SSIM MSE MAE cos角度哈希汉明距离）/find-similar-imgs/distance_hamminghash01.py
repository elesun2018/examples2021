# -*- coding: utf-8 -*-
'''
    图片相似度计算
fun:
    汉明距离表示两个（相同长度）字对应位不同的数量，向量相似度越高，对应的汉明距离越小。
ref:
    https://blog.csdn.net/sinat_26917383/article/details/70287521
'''
import cv2
import numpy as np

def pHash(imgfile):
    """get image pHash value"""
    #加载并调整图片为32x32灰度图片
    img=cv2.imread(imgfile, 0)
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
    #创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h,:w] = img       #填充数据
    #二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    #cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(32,32)
    #把二维list变成一维list
    img_list= [b for a in vis1.tolist() for b in a]
    #计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i<avg else '1' for i in img_list]
    #得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,32*32,4)])

'''
cv2.imreadflags>0时表示以彩色方式读入图片 flags=0时表示以灰度图方式读入图片 flags<0时表示以图片的本来的格式读入图片
interpolation - 插值方法。共有5种：１）INTER_NEAREST - 最近邻插值法２）INTER_LINEAR - 双线性插值法（默认）３）INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
４）INTER_CUBIC - 基于4x4像素邻域的3次插值法５）INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值http://blog.csdn.net/u012005313/article/details/51943442
'''
def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

HASH1=pHash('hat.png')
HASH2=pHash('images/hat/red/hathat.png') # images/hat/red/hathat.png images/hat/red/00000001.jpg
print("HASH1",HASH1)
print("HASH2",HASH2)
print("hamming distance : ",hammingDist(HASH1,HASH2))
similar = 1 - hammingDist(HASH1,HASH2)*1. / (32*32/4)
print("similar",similar)
