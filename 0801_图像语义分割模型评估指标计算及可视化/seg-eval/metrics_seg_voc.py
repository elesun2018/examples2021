# -*- coding: utf-8 -*-
# 计算VOC语义分割模型评估指标
# ref: 0302_语义分割mIOU评估指标分类计算
import numpy as np
import os,time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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
'''
正确的像素占总像素的比例
'''
def Pixel_Accuracy(confusion_matrix):
    Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return Acc
'''
分别计算每个类分类正确的概率
'''
def Pixel_Accuracy_Class(confusion_matrix):
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    Acc = np.nanmean(Acc)
    return Acc
'''
分别计算每个类分类的像素量
'''
def Class_Pix_Count(confusion_matrix, name_classes):
    count = confusion_matrix.sum(axis=1)
    return dict(zip(name_classes, count))
'''
在每个类上计算IoU，之后平均得到mIOU
Mean Intersection over Union(MIoU，均交并比)：为语义分割的标准度量。其计算两个集合的交集和并集之比.
在语义分割的问题中，这两个集合为真实值（ground truth）和预测值（predicted segmentation）。
这个比例可以变形为正真数（intersection）比上真正、假负、假正（并集）之和。
'''
def Class_Intersection_over_Union(confusion_matrix, name_classes):
    ClassIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    return dict(zip(name_classes, ClassIoU))

def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def calculation(mask_gt_dir, mask_pd_dir, name_classes):#计算mIoU的函数
    num_classes = len(name_classes)
    matrix = np.zeros((num_classes, num_classes))
    mask_gt_lists = os.listdir(mask_gt_dir)
    mask_gt_lists = [i for i in mask_gt_lists if i.endswith(".png")]
    mask_pd_lists = os.listdir(mask_pd_dir)
    mask_pd_lists = [i for i in mask_pd_lists if i.endswith(".png")]
    start_time = time.time()
    for mask_name in tqdm(mask_pd_lists):
        if mask_name not in mask_gt_lists: raise ValueError(mask_gt_dir, "find no", mask_name)
        mask_pd_path = os.path.join(mask_pd_dir, mask_name)
        mask_gt_path = os.path.join(mask_gt_dir, mask_name)
        mask_gt = Image.open(mask_gt_path)
        mask_gt = np.array(mask_gt)
        mask_pd = Image.open(mask_pd_path)
        mask_pd = np.array(mask_pd)
        # print("all values mask_gt : ", np.unique(mask_gt))
        # print("all values mask_pd : ", np.unique(mask_pd))
        assert mask_gt.shape == mask_pd.shape
        matrix += generate_matrix(mask_gt, mask_pd, num_class=num_classes)
    matrix = (matrix/len(mask_pd_lists)).astype(int)
    print("confusion_matrix\n", matrix)
    plt.figure()
    sns.heatmap(matrix, fmt='g', cmap='Blues', annot=True, cbar=False, xticklabels=name_classes,
                yticklabels=name_classes)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
    plt.title('confusion_matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('img_ConfusionMatrix.png')
    # plt.show()
    class_pix_count = Class_Pix_Count(matrix, name_classes)
    print("class_pix_count\n",class_pix_count)
    plt.figure()
    plt.barh(list(class_pix_count.keys()), list(class_pix_count.values()))
    plt.title('Pix Count')
    plt.xlabel('Pix Count')
    plt.ylabel('Class Name')
    plt.savefig('img_classpixcount.png')
    # plt.show()
    PA = Pixel_Accuracy(matrix)
    print("PA = ", round(PA, 3))
    mPA = Pixel_Accuracy_Class(matrix)
    print("mPA = ", round(mPA, 3))
    ClassIoU = Class_Intersection_over_Union(matrix, name_classes)
    print("ClassIoU\n",ClassIoU)
    plt.figure()
    plt.barh(list(ClassIoU.keys()), list(ClassIoU.values()))
    plt.title('Class IoU')
    plt.xlabel('IOU')
    plt.ylabel('Class')
    plt.savefig('img_ClassIoU.png')
    # plt.show()
    mIoU = np.nanmean(list(ClassIoU.values()))  # 跳过0值求mean
    print("mIoU = ", round(mIoU, 3))
    FWIoU = Frequency_Weighted_Intersection_over_Union(matrix)
    print("FWIoU = ", round(FWIoU, 3))

name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motobike", "person", "pottedplant", "sheep", "sofa", "train",
                    "tvmonitor"]

calculation('gt_VOC2007','pd_VOC2007', name_classes)#执行主函数 分别为 ‘ground truth’,'自己的实验分割结果'