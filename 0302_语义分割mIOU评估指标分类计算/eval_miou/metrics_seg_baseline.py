"""
Pytorch Unet模型
初赛：利用算法对遥感影像进行10大类地物要素分类，主要考察算法地物分类的准确性
ref:
https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.6cc26423onxSOX&postId=169396
https://tianchi.aliyun.com/competition/entrance/531860/information
"""
import os
import numpy as np
import time
from tqdm import tqdm
from PIL import Image

#计算mIoU的函数
def compute_mIoU(mask_gt_dir, mask_pd_dir, num_class):
    iou_list = [] # 列为每个class iou，行为每个图片的值
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
        mask_gt = np.array(mask_gt) - 1
        mask_pd = Image.open(mask_pd_path)
        mask_pd = np.array(mask_pd) - 1
        # print("all values mask_gt : ", np.unique(mask_gt))
        # print("all values mask_pd : ", np.unique(mask_pd))
        assert mask_gt.shape == mask_pd.shape
        iou_class_list = per_class_iou(mask_pd, mask_gt, num_class)
        iou_list.append(iou_class_list)
    return np.array(iou_list)

# 一张预测图，一张真值图，计算每个类别的iou
def per_class_iou(mask_pd, mask_gt, num_class):
    iou_class_list = []
    for idx in range(num_class):
        p = (mask_gt == idx).reshape(-1)
        t = (mask_pd == idx).reshape(-1)
        uion = p.sum() + t.sum() + 0.001
        overlap = (p * t).sum()
        # print(idx, uion, overlap)
        iou = 2 * overlap / uion
        iou_class_list.append(iou) # 沿着新轴连接数组的序列。
    return iou_class_list

# 类别名称与标签值从小到大依次排列 farmland-1 forest-2
class_name_list = ["farmland", "forest",  "grass", "road", "urban_area", "countryside","industrial_land", "construction",
                    "water", "bareland"]
print("class_name_list : ",class_name_list)
# gt_VOC2007 pd_VOC2007 gt_lishui_train pd_lishui_train
iou_np = compute_mIoU('gt_lishui_train','pd_lishui_train',len(class_name_list))
print("iou_np.shape",iou_np.shape) # 行数为比较图片的数量，列数为类别的数量
# print("iou_np\n", iou_np)
# 逐个样本图片输出一下mIoU值 对行求平均
print(np.mean(iou_np,axis=1))
for idx in range(iou_np.shape[0]) :
    print('mask : ',idx,"  miou : ",round(np.mean(iou_np[idx,:]), 3))
# 逐类别输出一下mIoU值 对列求平均
print(np.mean(iou_np,axis=0))
for idx,class_name in enumerate(class_name_list) :
    print('class : ',class_name,'  miou : ',round(np.mean(iou_np[:,idx]), 3))
# 在所有图像上求所有类别平均的mIoU值，计算时忽略NaN值
print(np.mean(iou_np))
print('mIoU: ',round(np.nanmean(iou_np), 2))


