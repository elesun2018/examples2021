# -*- coding: utf-8 -*-
"""
fun: 画混淆矩阵(confusion matrix)
ref:
    072 Python画图seaborn matplotlib
    说明seaborn热度图.doc
    06 seaborn生成热度图
    https://www.cnblogs.com/yexionglin/p/11432180.html
    https://blog.csdn.net/diyong3604/article/details/101184214?utm_medium=distribute.wap_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-7.essearch_wap_relevant
    https://blog.csdn.net/qq_41645987/article/details/109146503
    https://ai-exception.blog.csdn.net/article/details/80038380?utm_medium=distribute.wap_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-4.essearch_wap_relevant&depth_1-utm_source=distribute.wap_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-4.essearch_wap_relevant
"""
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

label_list = ['a', 'b', 'c']
y_true = [0, 0, 1, 2, 1, 2, 0, 2, 2, 0, 1, 1]
y_pred = [1, 0, 1, 2, 1, 0, 0, 2, 2, 0, 1, 1]
Confusion_Matrix = confusion_matrix(y_true, y_pred)
print("Confusion_Matrix\n",Confusion_Matrix)
# Plot ConfusionMatrix
plt.figure()
sns.heatmap(Confusion_Matrix, fmt='g', cmap='Blues', annot=True, cbar=False, xticklabels=label_list,
            yticklabels=label_list)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
plt.title('confusion_matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('img_ConfusionMatrix.png')
plt.show()
