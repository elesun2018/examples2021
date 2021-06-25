# -*- coding: utf-8 -*-
'''
    图片相似度计算
fun:
    汉明距离表示两个（相同长度）字对应位不同的数量，向量相似度越高，对应的汉明距离越小。
ref:
    https://blog.csdn.net/u013421629/article/details/85007793
'''
from functools import reduce
from PIL import Image
# 这种算法的优点是简单快速，不受图片大小缩放的影响，
# 缺点是图片的内容不能变更。如果在图片上加几个文字，它就认不出来了。
# 所以，它的最佳用途是根据缩略图，找出原图。

# 计算图片的局部哈希值--pHash
def phash(img):
    """
    :param img: 图片
    :return: 返回图片的局部hash值
    """
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    hash_value=reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)
    return hash_value

# 自定义计算两个图片相似度函数局部敏感哈希算法
def hanming_img_similarity(img1_path,img2_path):
    """
    :param img1_path: 图片1路径
    :param img2_path: 图片2路径
    :return: 图片相似度
    """
    # 读取图片
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    # 计算两个图片的局部哈希值
    img1_phash = str(phash(img1))
    img2_phash = str(phash(img2))
    # 打印局部敏感哈希值
    print("img1_phash",img1_phash)
    print("img2_phash",img2_phash)
    # 计算汉明距离
    distance = bin(phash(img1) ^ phash(img2)).count('1')
    print("distance",distance)
    print("max len",max(len(bin(phash(img1))), len(bin(phash(img1)))))
    similary = 1 - distance / max(len(bin(phash(img1))), len(bin(phash(img1))))
    print("两张图片相似度为:%s" % similary)

if __name__ == '__main__':
    img1_path = 'hat.png'
    img2_path = 'images/hat/red/hathat.png' # images/hat/red/hathat.png images/hat/red/00000001.jpg
    similary = hanming_img_similarity(img1_path, img2_path)
