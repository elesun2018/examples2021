#利用鸢尾花数据集绘制P-R曲线
#以iris数据为例，画出P-R曲线
# https://www.cnblogs.com/cxq1126/p/13018923.html
# https://blog.csdn.net/liujh845633242/article/details/102938143?utm_medium=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.wap_blog_relevant_pic&depth_1-utm_source=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.wap_blog_relevant_pic
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm,datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

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

fpr = dict()
tpr = dict()
roc_auc = dict()
################Compute micro-average ROC and ROC area##################################
# 'micro':通过先计算总体的TP，FN和FP的数量，再计算
fpr["micro"],tpr["micro"],_ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"],tpr["micro"])
################Compute macro-average ROC and ROC area##################################
# 'macro':分布计算每个类别的指标，然后做指标平均
# Compute ROC curve and ROC area for each class
for i in range(n_classes):
    # 取出来的是各个类的测试值和预测值
    fpr[i], tpr[i],_ = roc_curve(y_test[:, i],y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
################Plot all ROC curves##################################################
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area/microAUC = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle='solid', linewidth=2)
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area/macroAUC = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle='solid', linewidth=2)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, linestyle='dotted', linewidth=2,
             label='ROC curve of class {0} (area/AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.savefig("img_ROC_iris.png") #保存图片
# plt.show()
