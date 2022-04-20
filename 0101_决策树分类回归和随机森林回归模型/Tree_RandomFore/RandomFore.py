# -*- coding: utf-8 -*-
"""
    RandomForestRegressor
fun:
    随机森林回归实例-加州房价数据
env:
    scikit-learn0.23.2;graphviz0.15
ref:
    https://blog.csdn.net/sanjianjixiang/article/details/102789339
"""
import os
from sklearn.datasets.california_housing import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
import graphviz
# 读取加州房价数据
housing = fetch_california_housing()
# print("housing.DESCR",housing.DESCR)
print("housing.data.shape",housing.data.shape)
# print("housing.data[0]",housing.data[0])
# print("housing.data\n",housing.data),print("housing.target\n",housing.target)
X_train,X_test,y_train,y_test =train_test_split(housing.data,housing.target,test_size=0.2,random_state=0)
# print("X_train\n",X_train),print("X_test\n",X_test),print("y_train\n",y_train),print("y_test\n",y_test)

# 决策树建模
reg = RandomForestRegressor(
            # n_estimators=100, # 子模型的数量 100：默认值 在利用最大投票数或平均值来预测之前，你想要建立子树的数量。 较多的子树可以让模型有更好的性能，但同时让你的代码变慢。 你应该选择尽可能高的值，只要你的处理器能够承受的住，因为这使你的预测更好更稳定。
            # criterion="mse", # 判断节点是否继续分裂采用的计算方法 mse
            # max_depth=None, # 最大深度，如果max_leaf_nodes参数指定，则忽略
            # min_samples_split=2, # 分裂所需的最小样本数
            # min_samples_leaf=1, # 叶节点最小样本数
            # min_weight_fraction_leaf=0., # 叶节点最小样本权重总值 0
            # max_features="auto", # 节点分裂时参与判断的最大特征数 auto：所有特征数的开方 None：等于所有特征数
            # max_leaf_nodes=None, # 最大叶节点数 none 不限制叶节点数
            n_jobs=-1, # 并行数 这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，而“1”值意味着它只能使用一个处理器。
            random_state=0,
            verbose=2, # 日志冗长度 0：不输出训练过程  1：偶尔输出 >1：对每个子模型都输出
)
reg.fit(X_train, y_train)
print("score : ",reg.score(X_train, y_train))
# score : coefficient of determination R^2 of the prediction

# 对测试集进行预测
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print('predict mse : %.3f'%mse)

#网络搜索
from sklearn.model_selection import GridSearchCV
print("reg.get_params().keys()",reg.get_params().keys())
#dict_keys(['bootstrap', 'ccp_alpha', 'criterion', 'max_depth', 'max_features', 'max_leaf_nodes', 'max_samples', 'min_impurity_decrease', 'min_impurity_split', 'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 'n_estimators', 'n_jobs', 'oob_score', 'random_state', 'verbose', 'warm_start'])
param_grid = {
            # 'max_depth':[3,6,9],
            # 'min_sample_split': [3,6,9],
            'n_estimators': [10,30,50,80,100]
              }
# min_samples_split:分裂内部节点需要的最少样例数.int(具体数目),float(数目的百分比)
# n_estimators:森林中数的个数。这个属性是典型的模型表现与模型效率成反比的影响因子,即便如此,你还是应该尽可能提高这个数字,以让你的模型更准确更稳定。
grid = GridSearchCV(reg, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
# grid.cv_results_['mean_test_score']
# grid.cv_results_['std_test_score']
print("grid.cv_results_",grid.cv_results_)
print("grid.best_params_",grid.best_params_)
print("grid.best_score_",grid.best_score_)

# 根据以上输出设定好最优参数,再做随机森林回归
del reg
print("采用最优随机森林参数建模")
# 决策树建模
reg = RandomForestRegressor(
            n_estimators=10, # 子模型的数量 100：默认值 在利用最大投票数或平均值来预测之前，你想要建立子树的数量。 较多的子树可以让模型有更好的性能，但同时让你的代码变慢。 你应该选择尽可能高的值，只要你的处理器能够承受的住，因为这使你的预测更好更稳定。
            criterion="mse", # 判断节点是否继续分裂采用的计算方法 mse
            max_depth=5, # 最大深度，如果max_leaf_nodes参数指定，则忽略
            min_samples_split=20, # 分裂所需的最小样本数
            min_samples_leaf=10, # 叶节点最小样本数
            # min_weight_fraction_leaf=0., # 叶节点最小样本权重总值 0
            # max_features="auto", # 节点分裂时参与判断的最大特征数 auto：所有特征数的开方 None：等于所有特征数
            # max_leaf_nodes=None, # 最大叶节点数 none 不限制叶节点数
            n_jobs=-1, # 并行数 这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，而“1”值意味着它只能使用一个处理器。
            random_state=0,
            verbose=2, # 日志冗长度 0：不输出训练过程  1：偶尔输出 >1：对每个子模型都输出
)
reg.fit(X_train, y_train)
print("score : ",reg.score(X_train, y_train))
# score : coefficient of determination R^2 of the prediction
# 对测试集进行预测
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test,y_pred)

print('Save model...')
import joblib
joblib.dump(reg, 'model_rf_boston.pkl')
del reg
reg = joblib.load('model_rf_boston.pkl')

# 特征重要性
print("特征重要性排序：\n",pd.Series(reg.feature_importances_, index=housing.feature_names).sort_values(ascending=False))
# # 可视化显示
# 循环打印每棵树
for idx, estimator in enumerate(reg.estimators_):
    print("drawing tree",idx)
    # 导出dot文件
    export_graphviz(estimator,
                    out_file='rf_tree{}.dot'.format(idx),
                    feature_names=housing.feature_names,
                    class_names=housing.target_names,
                    rounded=True,
                    proportion=False,
                    precision=2,
                    filled=True)
    # 转换为png文件
    os.system('dot -Tpng rf_tree{}.dot -o rf_tree{}.png'.format(idx, idx))