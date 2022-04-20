# -*- coding: utf-8 -*-
"""
fun: 统计各类别数量
ref:
    https://blog.csdn.net/qq_35153620/article/details/101757464
"""
import xml.etree.ElementTree as ET
import os

def Countobject(xml_dir):
    """
    将 “按照图片名逐个文件保存预测结果的方式” 转换成 “按照类别名统计样本数量”
    """
    # 建立空字典，用于存储
    count = {}
    files = os.listdir(xml_dir)
    for file_name in files:
        if not file_name.endswith('.xml'):
            continue
        file_name_prefix = file_name.split('.')[0]
        file_path = os.path.join(xml_dir, file_name)
        tree = ET.parse(file_path)
        for obj in tree.findall('object'):
            class_name = obj.find('name').text
            # score = float(obj.find('score').text)
            # if score < 0 or score > 1 :
            #     raise ValueError("score value error!")
            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            if int(xmin) < 0 or int(ymin) < 0 or int(xmax) < 0 or int(ymax) < 0 \
                or int(xmax) < int(xmin) or int(ymax) < int(ymin) :
                raise ValueError("bbox value error!")
            # 检测标注类别是否第一次出现，若第一次出现则设置其数目为0；
            # 否则计数逐次加1
            if class_name in count:
                count[class_name] += 1
            else:
                count[class_name] = 1
    return count

if __name__ == '__main__':
    xml_dir = "gt_val_voc" # 模型预测结果
    count_dict = Countobject(xml_dir)
    print("count_dict:\n",count_dict)
