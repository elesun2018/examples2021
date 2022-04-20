# -*- coding: utf-8 -*-
"""
fun: 绘制字典型数据，画类别数量统计图
ref:
    https://blog.csdn.net/leokingszx/article/details/101456624
    https://blog.csdn.net/m0_46515351/article/details/106980698
"""
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

count_dict = {'dog': 16, 'person': 90, 'train': 12, 'sofa': 5, 'chair': 29, 'car': 28, 'pottedplant': 11, 'diningtable': 4, 'horse': 5, 'cat': 12, 'cow': 5, 'bus': 2, 'bicycle': 7, 'aeroplane': 6, 'motorbike': 9, 'tvmonitor': 5, 'bird': 6, 'bottle': 4, 'boat': 13, 'sheep': 1}

print("count_dict.items\n", count_dict.items())
plt.figure()
plt.barh(list(count_dict.keys()), list(count_dict.values()))
plt.title('Class Count')
plt.xlabel('Count Number')
plt.ylabel('Class Label')
plt.savefig('img_dict.png')
plt.show()
