# -*- coding: utf-8 -*-
# 计算统计VOC语义分割miou指标
# https://blog.csdn.net/weixin_42188270/article/details/86632714?utm_medium=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.wap_baidujs&dist_request_id=&depth_1-utm_source=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.wap_baidujs
# https://blog.csdn.net/u013249853/article/details/95458872
import numpy as np
import os,time
from PIL import Image
from tqdm import tqdm

def fast_hist(a, b, n):#a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    k = (a >= 0) & (a < n) #k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)#np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)

def per_class_iu(hist):#分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))#矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,) #hist.sum(0)=按列相加  hist.sum(1)按行相加

def compute_mIoU(mask_gt_dir, mask_pd_dir):#计算mIoU的函数
    num_classes = 21
    print('Num classes', num_classes)
    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motobike", "person", "pottedplant", "sheep", "sofa", "train",
                    "tvmonitor"]
    hist = np.zeros((num_classes, num_classes))

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

        hist += fast_hist(mask_gt.flatten(), mask_pd.flatten(), num_classes)#对一张图片计算19×19的hist矩阵，并累加
    mIoUs = per_class_iu(hist)#计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):#逐类别输出一下mIoU值
        print('=>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))#在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs

# gt_VOC2007 pd_VOC2007
compute_mIoU('gt_VOC2007','pd_VOC2007')#执行主函数