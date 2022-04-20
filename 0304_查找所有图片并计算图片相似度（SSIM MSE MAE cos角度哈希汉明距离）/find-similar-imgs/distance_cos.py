# -*- coding: utf-8 -*-
'''
    图片相似度计算
fun:
    余弦相似度用向量空间中两个向量夹角的余弦值作为衡量两个个体间差异的大小。
    两个向量越相似夹角越小，越小越相似，最小0
ref:
    https://zhuanlan.zhihu.com/p/88869743

'''
import numpy as np
from scipy.spatial.distance import pdist
from PIL import Image
# 生成两个不同的一维向量
# x=np.random.random(10) # 浮点数范围 : (0,1)
# y=np.random.random(10) #
# 生成两个相同的向量
# x=np.ones(10) # 1
# y=np.ones(10) #
# 生成两张伪图像
# img1=np.random.random((64,64)) #
# img2=np.random.random((64,64)) #
# x= np.reshape(img1, -1)
# y= np.reshape(img2, -1)
# 打开两张图，保证两个向量长度一致
img_in = Image.open("hat.png").convert('L') # pil读入gray通道
img_in_rs = img_in.resize((img_in.width//10, img_in.height//10),Image.ANTIALIAS)
img_in_rs_np = np.asarray(img_in_rs)/256.0
img = Image.open("images/hat/red/hathat.png").convert('L') # car.png images/hat/red/hathat.png
img_rs = img.resize((img_in.width//10, img_in.height//10), Image.ANTIALIAS)
img_rs_np = np.asarray(img_rs)/256.0
x= np.reshape(img_in_rs_np, -1)
y= np.reshape(img_rs_np, -1)

# 代码实现方法一
dist1 = 1 - np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
# 代码实现方法二
dist2 = pdist(np.vstack([x,y]),'cosine')

print('x.shape',x.shape)
print('y.shape',y.shape)
print('dist1',dist1)
print('dist2',dist2)
