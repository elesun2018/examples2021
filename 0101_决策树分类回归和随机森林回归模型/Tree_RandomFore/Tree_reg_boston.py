# -*- coding: utf-8 -*-
""" 
    DecisionTreeRegressor
fun:
    决策树回归实例-加州房价数据
env:
    scikit-learn0.23.2;graphviz0.15
ref:
    https://blog.csdn.net/sanjianjixiang/article/details/102789339
"""
from sklearn.datasets.california_housing import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
import graphviz
# 读取加州房价数据
housing = fetch_california_housing()
print("housing.DESCR",housing.DESCR)
print("housing.data.shape",housing.data.shape)
# print("housing.data[0]",housing.data[0])
# print("housing.data\n",housing.data),print("housing.target\n",housing.target)
X_train,X_test,y_train,y_test =train_test_split(housing.data,housing.target,test_size=0.2,random_state=0)
# print("X_train\n",X_train),print("X_test\n",X_test),print("y_train\n",y_train),print("y_test\n",y_test)

# 决策树建模
reg = DecisionTreeRegressor(
        # criterion="mse", # 特征选择标准criterion 可以使用"mse"或者"mae"，前者是均方差，后者是和均值之差的绝对值之和。推荐使用默认的"mse"。一般来说"mse"比"mae"更加精确。
        # splitter="best", # 特征划分点选择标准splitter 可以使用"best"或者"random"。前者在特征的所有划分点中找出最优的划分点。后者是随机的在部分划分点中找局部最优的划分点。默认的"best"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"random"
        # max_depth=None, #决策树的最大深度，默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。
        # min_samples_split=2, # 内部节点再划分所需最小样本数 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。
        # min_samples_leaf=1, # 叶子节点最少样本数 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。之前的10万样本项目使用min_samples_leaf的值为5，仅供参考。
        # min_weight_fraction_leaf=0., # 叶子节点最小的样本权重和 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。
        # max_features=None, # 划分时考虑的最大特征数 一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。
        random_state=0,
        # max_leaf_nodes=None, # 最大叶子节点数  通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
        # min_impurity_decrease=0.,
        # min_impurity_split=None, # 节点划分最小不纯度 这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点。即为叶子节点
        # presort='deprecated', # 数据是否预排序presort 这个值是布尔值，默认是False不排序。一般来说，如果样本量少或者限制了一个深度很小的决策树，设置为true可以让划分点选择更加快，决策树建立的更加快。如果样本量太大的话，反而没有什么好处。问题是样本量少的时候，我速度本来就不慢。所以这个值一般懒得理它就可以了。
        # ccp_alpha=0.0
)
reg.fit(X_train, y_train)
print("score : ",reg.score(X_train, y_train))
# score : coefficient of determination R^2 of the prediction.
print('Save model...')
import joblib
joblib.dump(reg, 'model_boston.pkl')
del reg
reg = joblib.load('model_boston.pkl')

# 对测试集进行预测
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print('predict mse : %.3f'%mse)

# 可视化显示
dot_data_string = export_graphviz(reg, out_file=None, feature_names=housing.feature_names, filled=True, impurity=False, rounded=True)
# dot_data_string = dot_data_string.replace('helvetica','"Microsoft Yahei"')
#因为标签是中文所以需要将参数设置成支持微软雅黑的格式

graph = graphviz.Source(dot_data_string, filename="dot_reg_boston")#选择要可视化的dot数据
graph.render(view=True)#将可视化结果输出至指定位置
