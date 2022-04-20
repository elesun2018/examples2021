# -*- coding: utf-8 -*-
"""
nam:
    sklearn实现鸢尾花数据分类模型训练测试评估指标metrics
fun:

env:

ref:
    https://www.pianshen.com/article/3086301452/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,label_binarize
from sklearn.multiclass import OneVsRestClassifier  #一对其余（每次将一个类作为正类，剩下的类作为负类）
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score,classification_report,precision_recall_curve,average_precision_score

# 加载本地数据
iris = pd.read_csv('iris.csv', usecols=[1, 2, 3, 4, 5])
# iris.info()
# print("iris.head\n",iris.head())
# print("iris.describe\n",iris.describe())
# 载入特征和标签集
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']
target_names=iris['Species'].unique()
n_classes=target_names.shape[0]     #多少类

# 对标签集进行编码
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# print("y\n",y)

# iris = load_iris()
# print("iris.DESCR",iris.DESCR)
# X = iris.data
# y = iris.target
# print("X.shape",X.shape)
# print("X\n",X)
# print("y.shape",y.shape)
# print("y\n",y)
# target_names=iris.target_names
# n_classes=target_names.shape[0]     #多少类

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("train_X.shape:",train_X.shape, "train_y.shape:",train_y.shape, "test_X.shape:",test_X.shape, "test_y.shape:",test_y.shape)

# 标准化特征值
sc = StandardScaler()
sc.fit(train_X)
train_X_std = sc.transform(train_X)
test_X_std = sc.transform(test_X)
# 模型创建训练保存
# Decision Tree
print("Decision Tree")
clf=DecisionTreeClassifier()
clf.fit(train_X_std, train_y)
score = clf.score(test_X_std, test_y)
print("score",round(score,3)) # accuracy_score

print("Feature importances:\n{}".format(clf.feature_importances_))
plt.figure()
n_features = X.shape[1] # iris.data.shape[1]
plt.barh(range(n_features), clf.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X.columns.values,fontsize=5,rotation=45)
# 指标显示不全，对坐标的大小方向设定 iris.feature_names
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title('Feature Importance')
plt.savefig("img_feature_importance_iris.png") #保存图片
# plt.show()
# del clf
# 预测推理
pred_y = clf.predict(test_X_std)
# print("pred_y\n",pred_y)
pred_y_proba = clf.predict_proba(test_X_std)
# print("pred_y_proba\n",pred_y_proba)

# 指标评估
# print("classification_report\n",classification_report(test_y,pred_y,target_names=target_names))
print("confusion_matrix\n",confusion_matrix(test_y,pred_y,labels=range(n_classes)))#
# 准确率
metric_accuracy = accuracy_score(test_y,pred_y)
print("metric_accuracy",round(metric_accuracy,3))
# 精准率
metric_precision = precision_score(test_y,pred_y,average='macro')
# 'micro':通过先计算总体的TP，FN和FP的数量，再计算F1 'macro':分布计算每个类别的F1，然后做平均（各类别F1的权重相同）
print("metric_precision",round(metric_precision,3))
# 召回率
metric_recall = recall_score(test_y,pred_y,average='macro')
print("metric_recall",round(metric_recall,3))
# F1(精准率与召回率的平衡)
metric_f1 = f1_score(test_y,pred_y,average='macro')
print("metric_f1",round(metric_f1,3))

