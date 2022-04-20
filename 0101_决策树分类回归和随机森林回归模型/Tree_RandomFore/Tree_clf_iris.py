# -*- coding: utf-8 -*-
""" 
    DecisionTreeClassifier
fun:
    决策树分类实例-鸢尾花数据
env:
    scikit-learn0.23.2;graphviz0.15
ref:
    https://www.baidu.com/link?url=ZL0WV0MqydCFdP-hQkzM3gy4jX8Ltfa8PjRH1RMIYiBGrWFiaUgFGbxvo5SLaNpQN3XTG4UNfuuqVE4U-3lsoa&wd=&eqid=e1f8d42f0000152e000000065fd1d059
"""
from __future__ import print_function
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,export_graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
import graphviz
# 加载你的数据
iris = load_iris()   # 载入鸢尾花数据集
data=iris.data
target = iris.target
# print("data\n",data),print("target\n",target)
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.2,random_state=0)
# print("X_train\n",X_train),print("X_test\n",X_test),print("y_train\n",y_train),print("y_test\n",y_test)

# 决策树建模
clf = DecisionTreeClassifier(
        criterion="gini", # 标准有gini or entropy可以选择
        splitter="best", # best or random 前者是在所有特征中找最好的切分点 后者是在部分特征中（数据量大的时候）
        max_depth=None, # 数据少或者特征少的时候可以不管这个值，如果模型样本量多，特征也多的情况下，可以尝试限制下
        min_samples_split=2, # 如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值
        min_samples_leaf=1, # 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝，如果样本量不大，不需要管这个值，大些如10W可是尝试下5
        min_weight_fraction_leaf=0., # 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了
        max_features=None, # None（所有），log2，sqrt，N 特征小于50的时候一般使用所有的
        random_state=0,
        max_leaf_nodes=None, # 通过限制最大叶子节点数，可以防止过拟合，默认是”None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制具体的值可以通过交叉验证得到
        min_impurity_decrease=0.,
        min_impurity_split=None,
        class_weight=None, # 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多导致训练的决策树过于偏向这些类别。
        presort='deprecated',
        ccp_alpha=0.0,
        )
# max_depth = 2
clf.fit(X_train, y_train)
print("score : ",clf.score(X_train, y_train))
# score : the mean accuracy on the given test data and labels
print('Save model...')
import joblib
joblib.dump(clf, 'model_iris.pkl')
del clf
clf = joblib.load('model_iris.pkl')

# 对测试集进行预测
y_pred = clf.predict(X_test)
# model.predict_proba
#计算准确率
accuracy = accuracy_score(y_test,y_pred)
print('predict acc : %.3f'%accuracy)

# 可视化显示
dot_data_string = export_graphviz(clf, out_file=None, feature_names=iris.feature_names, filled=True, impurity=False, rounded=True)
# dot_data_string = dot_data_string.replace('helvetica','"Microsoft Yahei"')
#因为标签是中文所以需要将参数设置成支持微软雅黑的格式

graph = graphviz.Source(dot_data_string, filename="dot_clf_iris")#选择要可视化的dot数据
graph.render(view=True)#将可视化结果输出至指定位置