# -*- coding: utf-8 -*-
"""
评价指标：PA、CPA、MPA、IoU、MIoU详细总结和代码实现
【语义分割】评价指标实现代码
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
https://blog.csdn.net/pangxing6491/article/details/108773785?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-2&spm=1001.2101.3001.4242
"""
import numpy as np
np.random.seed(0)
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc) # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU) # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel): # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        print("confusionMatrix\n",confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    numclass = 5 # 保证提供的示例内的值分布范围 0-numclass 正整数 <numclass
    # 简单示例
    # imgPredict = np.array([0, 1, 1, 1, 2, 2]) # 可直接换成pd预测图片  numclass = 3
    # imgLabel =   np.array([0, 1, 1, 1, 2, 2])  # 可直接换成gt标注图片
    # 随机数示例
    # imgPredict = np.random.randint(0, numclass, (512, 512, 3))
    # imgLabel = np.random.randint(0, numclass, (512, 512, 3))
    # 图片示例1 numclass = 4
    # imgPredict = cv2.imread("img1_0_85_pd.png",cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE：读入灰度图片
    # imgLabel = cv2.imread("img1_0_85_gt.png",cv2.IMREAD_GRAYSCALE) # 这个图的值的分布为 0 1 2 3
    # 图片示例2 numclass = 4
    # imgPredict = Image.open("image_1_label_pd.png")
    # imgPredict = np.array(imgPredict)
    # imgLabel = Image.open("image_1_label_gt.png")
    # imgLabel = np.array(imgPredict)
    # 图片示例3 numclass = 5
    imgPredict = Image.open("tc_lin_pd/20210105_deeplabv3plus_resnet101_StepLR_Adam_temp/test_image20_pipeline_predict.png") # "tc_lm_pd/image_1_predict.png"
    imgPredict = np.array(imgPredict)
    imgLabel = Image.open("tc_gt/image_20_label.png") # image_2_label
    imgLabel = np.array(imgLabel)
    print("all values imgPredict : ",np.unique(imgPredict))
    print("all values imgLabel : ", np.unique(imgLabel))

    print("imgPredict.shape", imgPredict.shape)
    print("imgLabel.shape", imgLabel.shape)

    metric = SegmentationMetric(numclass) # numclass表示有numclass个分类，有几个分类就填几
    metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    print('pa is : %f' % pa)
    print('cpa is :',cpa) # 列表
    print('mpa is : %f' % mpa)
    print('mIoU is : %f' % mIoU)