# -*- coding: utf-8 -*-
"""
    VOC目标检测模型评估指标计算及可视化
fun:
    不同类别的图像框个数统计:图像类别分布,数据集中不同类别的图像框个数统计。
    一个类别的P-R曲线:根据每种分类的置信度对样例进行排序，逐个把样例加入正例进行预测，算出此时的精准率和召回率。使用这一系列的精准率和召回率绘制的曲线，即是一个类别的P-R曲线。
    不同目标框交并比阈值下的mAP: 计算不同目标框交并比阈值下的mAP值，并绘制曲线，反馈mAP值最高的阈值。其中交并比阈值是用于NMS时过滤可能预测为同一物体的重叠框的阈值。
ref:
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py 修改得到的mAP计算脚本
"""
import xml.etree.ElementTree as ET
import os
import numpy as np
import time
import codecs
import pandas as pd
from itertools import cycle
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

name_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def parse_gt_xml(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(gt_xml_path_lists, classname, pd_classname_dict, ovthresh=0.5, use_07_metric=False):
    # first load gt
    gt_class_dict, npos = gt_xmls2class_dict(gt_xml_path_lists, classname)
    # second load pd
    # if len(pd_classname_dict) == 0 : return 0, 0, 0
    splitlines = [x.strip().split(' ') for x in pd_classname_dict]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = gt_class_dict[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh: # IOU 0.5
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) # recall = tp/(tp+fn)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) # precision = tp/(tp+fp)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def pd_xmls2class_dict(pd_xml_path_lists, name_classes):
    """
    将 “按照图片名逐个文件保存预测结果的方式” 转换成 “按照类别名逐个文件保存预测结果的方式”
    """
    results = {}
    for class_name in name_classes:
        results[class_name] = []
    for pd_xml_path in pd_xml_path_lists:
        tree = ET.parse(pd_xml_path)
        for obj in tree.findall('object'):
            class_name = obj.find('name').text
            if class_name not in name_classes :
                raise ValueError("class_name value error!")
            score = float(obj.find('score').text)
            if score < 0 or score > 1 :
                raise ValueError("score value error!")
            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            if int(xmin) < 0 or int(ymin) < 0 or int(xmax) < 0 or int(ymax) < 0 \
                or int(xmax) < int(xmin) or int(ymax) < int(ymin) :
                raise ValueError("bbox value error!")
            line = (os.path.basename(pd_xml_path).split(".")[0] + ' ' + '%.4f' %score + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax)
            results[class_name].append(line)
    return results
def gt_xmls2class_dict(gt_xml_path_lists, classname):
    # first load gt
    recs = {}
    for gt_xml_path in gt_xml_path_lists:
        recs[os.path.basename(gt_xml_path).split(".")[0]] = parse_gt_xml(gt_xml_path)
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for gt_img_path in gt_xml_path_lists:
        R = [obj for obj in recs[os.path.basename(gt_img_path).split(".")[0]] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[os.path.basename(gt_img_path).split(".")[0]] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    return class_recs,npos

def calculation(gt_dir, pd_dir, result_dir, name_classes=name_classes):
    start_time = time.time()
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    num_classes = len(name_classes)
    gt_lists = os.listdir(gt_dir)
    gt_xml_lists = [i for i in gt_lists if i.endswith(".xml")]
    gt_xml_path_lists = [os.path.join(gt_dir, i) for i in gt_xml_lists]
    pd_lists = os.listdir(pd_dir)
    pd_xml_path_lists = [os.path.join(pd_dir, i) for i in gt_xml_lists if os.path.exists(os.path.join(pd_dir, i))]

    pd_class_dict = pd_xmls2class_dict(pd_xml_path_lists, name_classes)
    # print("pd_class_dict\n", pd_class_dict)
    plt.figure()
    plt.barh(list(pd_class_dict.keys()), [len(pd_class_dict[i]) for i in pd_class_dict.keys()])
    plt.title('Class Box Count')
    plt.xlabel('Box Count')
    plt.ylabel('Class Name')
    img_classboxcount_path = os.path.join(result_dir, "img_classboxcount.png")
    plt.savefig(img_classboxcount_path)
    # plt.show

    aps_list = []  # 保存各类ap
    plt.figure()
    plt.title('PRcurve@Class')
    plt.ylabel('precise')
    plt.xlabel('recall')
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for classname, color in zip(name_classes, colors) :
        rec, prec, ap = voc_eval(gt_xml_path_lists, classname, pd_class_dict[classname], ovthresh=0.5, use_07_metric=False)
        plt.plot(rec, prec, color=color, label='class:{0},AP:{1:0.3f})'.format(classname, ap))
        aps_list.append([round(rec[-1], 4), round(prec[-1], 4), round(ap, 4)])
    img_PRcurvexclass_path = os.path.join(result_dir, "img_PRcurvexclass.png")
    plt.legend(loc='best')
    plt.savefig(img_PRcurvexclass_path)
    # plt.show()
    # AP50table
    aps_np = np.array(aps_list)
    # print("aps_np\n", aps_np)
    plt.figure()
    sns.heatmap(aps_np, fmt='g', cmap='Blues', annot=True, cbar=False, xticklabels=["recall", "precise", "AP"],
                yticklabels=name_classes)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
    plt.title('AP50table')
    plt.ylabel('Class Label')
    plt.xlabel('Metrics AP')
    img_AP50table_path = os.path.join(result_dir, "img_AP50table.png")
    plt.savefig(img_AP50table_path)
    # plt.show()
    mRecalls50 = np.mean(aps_np[:, 0])
    mPrecise50 = np.mean(aps_np[:, 1])
    mAP50 = np.mean(aps_np[:, 2])

    # mAP @ ovthres
    mAP_list = []
    ovthresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for ovthresh_val in ovthresh_list :
        aps_list = []  # 保存各类ap
        for classname in name_classes :
            rec, prec, ap = voc_eval(gt_xml_path_lists, classname, pd_class_dict[classname], ovthresh=ovthresh_val, use_07_metric=False)
            aps_list.append([round(rec[-1],4), round(prec[-1],4), round(ap,4)])
        aps_np = np.array(aps_list)
        # print("aps_np\n", aps_np)
        mAP_list.append(np.mean(aps_np[:,2]))
    plt.figure()
    plt.plot(ovthresh_list, mAP_list)
    plt.title('mAP@ovthresh')
    plt.xlabel('Ovthresh')
    plt.ylabel('mAP')
    plt.xticks(np.arange(0, 1.1, step=0.2))  # xticks>xlim
    plt.yticks(np.arange(0, 1.1, step=0.2))  # yticks>ylim
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    img_mAPxovthresh_path = os.path.join(result_dir, "img_mAPxovthresh.png")
    plt.savefig(img_mAPxovthresh_path)
    # plt.show()

    res_dict = {
        "img_classboxcount": img_classboxcount_path,
        "img_PRcurvexclass": img_PRcurvexclass_path,
        "img_AP50table": img_AP50table_path,
        "mPrecise50": str(round(mPrecise50, 3)),
        "mRecalls50": str(round(mRecalls50, 3)),
        "mAP50": str(round(mAP50, 3)),
        "img_mAPxovthresh": img_mAPxovthresh_path,
    }
    end_time = time.time()
    print("calculation metrics time use {:.3f} S".format(end_time-start_time))
    return res_dict

if __name__ == '__main__':
    gt_dir = "gt_voc"
    pd_dir = "pd_xml_voc"
    result_dir = "results"
    res_dict = calculation(gt_dir, pd_dir, result_dir, name_classes=name_classes)
    print("res_dict\n", res_dict)