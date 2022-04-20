# -*- coding: utf-8 -*-
"""
fun: 绘制矩阵热度图
ref:
    072 Python画图seaborn matplotlib
    说明seaborn热度图.doc
    06 seaborn生成热度图
    https://www.cnblogs.com/yexionglin/p/11432180.html
    https://blog.csdn.net/diyong3604/article/details/101184214?utm_medium=distribute.wap_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-7.essearch_wap_relevant
    https://blog.csdn.net/qq_41645987/article/details/109146503
    https://ai-exception.blog.csdn.net/article/details/80038380?utm_medium=distribute.wap_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-4.essearch_wap_relevant&depth_1-utm_source=distribute.wap_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-4.essearch_wap_relevant
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(3)
# M = np.random.random((10, 10)) # 生成指定形状的0 - 1之间的随机数
M = np.random.randint(0, 10, size=(10, 10)) # 生成指定数值范围内的随机整数
print("Matrix\n",M)
label_list = [0,1,2,3,4,5,6,7,8,9]
plt.figure()
sns.heatmap(M, fmt='g', cmap='Blues', annot=True, cbar=False, xticklabels=label_list,
            yticklabels=label_list)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
plt.title('Matrix')
plt.ylabel('Y')
plt.xlabel('X')
plt.savefig('img_matrix.png')
plt.show()
