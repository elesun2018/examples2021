# -*- coding: utf-8 -*-
'''
https://blog.csdn.net/qq_41375318/article/details/108380694
图像分割的各项一般指标的计算一般分两步，一是计算混淆矩阵，二是计算各项指标
'''
import numpy as np
np.random.seed(0)
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

numclass = 4 # 保证提供的示例内的值分布范围 0-numclass 正整数 <numclass
# 简单示例
# gt_image = np.array([ # numclass = 3
#     [0, 1, 2, 4],
#     [0, 0, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 0]
# ])
# pre_image = np.array([ # numclass = 3
#     [0, 1, 2, 4],
#     [0, 1, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0]
# ])
# 随机数示例
# pre_image = np.random.randint(0, numclass, (512, 512, 3))
# gt_image = np.random.randint(0, numclass, (512, 512, 3))
# 图片示例
pre_image = cv2.imread("img1_0_85_pd.png",cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE：读入灰度图片  numclass = 4
gt_image = cv2.imread("img1_0_85_gt.png",cv2.IMREAD_GRAYSCALE) # 这个图的值的分布为 0 1 2 3
# 图片示例3 numclass = 4
# pre_image = Image.open("lin_pd/20210105_deeplabv3plus_resnet101_StepLR_Adam_temp/test_image10_pipeline_predict.png") # lm_pd/image_1_predict.png
# pre_image = np.array(pre_image)
# gt_image = Image.open("tc_gt/image_10_label.png") # image_2_label
# gt_image = np.array(gt_image)
print("all values gt_image : ",np.unique(gt_image))
print("all values pre_image : ", np.unique(pre_image))
print("gt_image.shape",gt_image.shape)
print("pre_image.shape",pre_image.shape)
'''
计算混淆矩阵
'''
def generate_matrix(gt_image, pre_image, num_class=8):
    mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)  # 21 * 21(for pascal)
    return confusion_matrix
matrix = generate_matrix(gt_image, pre_image, num_class = numclass)
print("confusion_matrix\n",matrix)
'''
正确的像素占总像素的比例
'''
def Pixel_Accuracy(confusion_matrix):
    Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return Acc
PA = Pixel_Accuracy(matrix)
print("PA = ",round(PA,3))
'''
分别计算每个类分类正确的概率
'''
def Pixel_Accuracy_Class(confusion_matrix):
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    Acc = np.nanmean(Acc)
    return Acc
MPA = Pixel_Accuracy_Class(matrix)
print("MPA = ",round(MPA,3))
'''
    Mean Intersection over Union(MIoU，均交并比)：为语义分割的标准度量。其计算两个集合的交集和并集之比.
    在语义分割的问题中，这两个集合为真实值（ground truth）和预测值（predicted segmentation）。
    这个比例可以变形为正真数（intersection）比上真正、假负、假正（并集）之和。在每个类上计算IoU，之后平均。
'''
def Mean_Intersection_over_Union(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)  # 跳过0值求mean,shape:[21]
    return MIoU
MIoU = Mean_Intersection_over_Union(matrix)
print("MIoU = ",round(MIoU,3))

def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU
FWIoU = Frequency_Weighted_Intersection_over_Union(matrix)
print("FWIoU = ",round(FWIoU,3))