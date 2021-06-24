#利用鸢尾花数据集绘制P-R曲线
# https://www.cnblogs.com/cxq1126/p/13018923.html
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score,auc,precision_score,recall_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier  #一对其余（每次将一个类作为正类，剩下的类作为负类）

# from sklearn.cross_validation import train_test_split  #适用于anaconda 3.6及以前版本
from sklearn.model_selection import train_test_split#适用于anaconda 3.7

#以iris数据为例，画出P-R曲线
iris = datasets.load_iris()
X = iris.data    #150*4
y = iris.target  #150*1

# 标签二值化,将三个类转为001, 010, 100的格式.因为这是个多类分类问题，后面将要采用
#OneVsRestClassifier策略转为二类分类问题
y = label_binarize(y, classes=[0, 1, 2])    #将150*1转化成150*3
n_classes = y.shape[1]                      #列的个数，等于3
# print("y\n",y)

# 训练集和测试集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# 一对其余，转换成两类，构建新的分类器
clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=0))
#训练集送给fit函数进行拟合训练，训练完后将测试集的样本特征注入，得到测试集中每个样本预测的分数
clf.fit(X_train, y_train)
# y_score = clf.decision_function(X_test)
y_score = clf.predict(X_test)

precision = dict()
recall = dict()
average_precision = dict()
################Compute micro-average PR and PR area##################################
# 'micro':通过先计算总体的TP，FN和FP的数量，再计算
# Compute micro-average curve and area. ravel()将多维数组降为一维
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),  y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score, average="micro") #This score corresponds to the area under the precision-recall curve.
################Compute macro-average PR and PR area##################################
# 'macro':分布计算每个类别的指标，然后做指标平均
for i in range(n_classes):
    #对于每一类，计算精确率和召回率的序列（:表示所有行，i表示第i列）
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],  y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])#切片，第i个类的分类结果性能
# precision["macro"], recall["macro"], _ = precision_recall_curve(y_test,  y_score)
average_precision["macro"] = average_precision_score(y_test, y_score, average="macro")
################Plot all PR curves##################################################
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label='micro-average PR curve (area/microAP = {0:0.2f})'
               ''.format(average_precision["micro"]),
         color='deeppink', linestyle='solid', linewidth=2)
# plt.plot(recall["macro"], precision["macro"],
#          label='macro-average PR curve (area/macroAP = {0:0.2f})'
#                ''.format(average_precision["macro"]),
#          color='navy', linestyle='solid', linewidth=2)
# Plot Precision-Recall curve for each class
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, linestyle='dotted', linewidth=2,
             label='PR curve of class {0} (area/AP = {1:0.2f})'
             ''.format(i, average_precision[i]))
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05]) #xlim、ylim：分别设置X、Y轴的显示范围。
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")#legend 是用于设置图例的函数
plt.savefig("img_PR_iris.png") #保存图片
# plt.show()
