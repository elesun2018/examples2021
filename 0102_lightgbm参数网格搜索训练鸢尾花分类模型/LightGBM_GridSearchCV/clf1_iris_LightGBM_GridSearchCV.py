# coding: utf-8
# lightgbm参数网格搜索 训练鸢尾花分类模型
# LightGBM 的 sklearn 风格接口 LGBMClassifier
# https://blog.csdn.net/huacha__/article/details/81057150
import os
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sklearn
print(sorted(sklearn.metrics.SCORERS.keys()))
# 加载数据
print('Load data...')
iris = load_iris()   # 载入鸢尾花数据集
data = iris.data
target = iris.target
X_train,X_valid,y_train,y_valid =train_test_split( data, target, test_size=0.2, random_state=0)
print("X_train.shape",X_train.shape)
print("y_train.shape",y_train.shape)
print("X_valid.shape",X_valid.shape)
print("y_valid.shape",y_valid.shape)
# 数据标准化
ss = StandardScaler()
X_train_s = ss.fit_transform(X_train)
X_valid_s = ss.transform(X_valid)
# 输出下原数据的标准差和平均数
print("ss.scale_",ss.scale_)
print("ss.mean_",ss.mean_)
# print("y_train\n",y_train)
# print("y_valid\n",y_valid)
print('Start training...')
# 构建分类器
# Construct a gradient boosting model
clf = lgb.LGBMClassifier(
    objective =  "multiclass",
    boosting_type = "gbdt" ,
    # num_leaves = 50 ,
    # max_depth = 6 ,
    # learning_rate = 0.01,
    #n_estimators = 10,
    # min_data_in_leaf = 10,
    silent=True , # 是否静默模式，True不打印警告和信息
    )
# 参数说明
# boosting_type='gbdt', num_leaves=31, max_depth=-1,
# learning_rate=0.1, n_estimators=100,
# subsample_for_bin=200000, objective=None,
# reg_alpha=0., reg_lambda=0., random_state=None,
# n_jobs=-1, silent=True
#设定网格搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param_dist = {
    # Step1. 学习率和估计器及其数目
    #     'n_estimators': range(1,1000,10), # best21
    #     'learning_rate': np.linspace(0.01,1,50), # best0.111
    # Step2. 提高精确度的最重要的参数 max_depth 和 num_leaves num_leaves = 2^(max_depth)
        'max_depth': range(2,8,1), # best2
        # 'num_leaves': range(10,200,10),   # 叶子节点数 best10
    # Step3: 降低过拟合 min_data_in_leaf 和 min_sum_hessian_in_leaf 没有这个参数???
    #     'min_child_samples': [18, 19, 20, 21, 22], # min_data_in_leaf（min_child_samples ）：叶子节点中最小的数据量，调大可以防止过拟合。
    #     'min_child_weight':[0.001, 0.002], # min_sum_hessian_in_leaf（min_child_weight）：叶子节点的最小权重和，调大可以防止过拟合。
    # Step4: 降低过拟合 feature_fraction 和 bagging_fraction
    #     'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9], # feature_fraction（colsample_bytree）：列采样比例，同 XGBoost，调小可以防止过拟合，加快运算速度。
    #     'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0], # bagging_fraction（subsample ）：样本采样比例，同 XGBoost ，调小可以防止过拟合，加快运算速度。
        # bagging_freq（subsample_freq）：bagging 的频率，0 表示禁止 bagging，正整数表示每隔多少个迭代进行 bagging。
    # 降低过拟合 正则化参数  没有这个参数???
    #     'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5], # lambda_l1（reg_alpha）：L1 正则化项 best0.5
    #     'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5], # lambda_l2（reg_lambda）：L2 正则化项 best0
        }
grid_search = GridSearchCV(clf,param_dist,
                           scoring = "accuracy",
                           n_jobs = -1,cv = 5)
# GridSearchCV参数说明
# estimator：选择使用的分类器，并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法：如estimator = RandomForestClassifier(min_sample_split=100,min_samples_leaf = 20,max_depth = 8,max_features = 'sqrt' , random_state =10),
# scoring = None ：模型评价标准，默认为None，这时需要使用score函数；或者如scoring = 'roc_auc'，根据所选模型不同，评价准则不同，字符串（函数名），或是可调用对象，需要其函数签名，形如：scorer(estimator，X，y）；如果是None，则使用estimator的误差估计函数。
# verbose = 0 ,verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
#scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
print("grid_search\n",grid_search)
grid_search.fit(X_train_s,y_train)
#grid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合
#best_score_：成员提供优化过程期间观察到的最好的评分
print("best_estimator_",grid_search.best_estimator_)
print("best_params_",grid_search.best_params_)
print("best_score_",grid_search.best_score_)


# 用最优的参数构建分类器
# Construct a gradient boosting model
del clf
clf = lgb.LGBMClassifier(
    objective =  "multiclass",
    boosting_type = "gbdt" ,
    # num_leaves = 50 ,
    max_depth = 2 ,
    # learning_rate = 0.01,
    #n_estimators = 10,
    # min_data_in_leaf = 10,
    silent=False , # 是否静默模式，True不打印警告和信息
    )
print("clf\n",clf)
clf.fit(X_train_s,y_train)
results_train_score = clf.score(X_train_s, y_train)
print("results_train_score",results_train_score)

print('Save model...')
import joblib
joblib.dump(clf, 'model_iris.pkl')
del clf
clf = joblib.load('model_iris.pkl')
# 预测数据集
print('Start predicting...')
results_proba = clf.predict_proba(X_valid_s)
# print("results_proba\n",results_proba)
y_predict = clf.predict(X_valid_s)
# 评估模型
print('Start eval...')
# print("y_valid\n",y_valid)
# print("y_predict\n",y_predict)
#  精准率
from sklearn.metrics import precision_score
metric_precision = precision_score(y_valid,y_predict, average='micro')
print("metric_precision",round(metric_precision,3))
# 召回率
from sklearn.metrics import recall_score
metric_recall = recall_score(y_valid,y_predict, average='micro')
print("metric_recall",round(metric_recall,3))
# F1(精准率与召回率的平衡)
from sklearn.metrics import f1_score
metric_f1 = f1_score(y_valid,y_predict, average='micro')
print("metric_f1",round(metric_f1,3))

from sklearn.metrics import roc_auc_score
# metric_roc = roc_auc_score(y_valid,y_predict), average="micro", multi_class="ovr")
# print("metric_roc",round(metric_roc,3))


print('predict...')
clf = joblib.load('model_iris.pkl')
# 加载数据
print('Load test data...')
X_train,X_test,y_train,y_test =train_test_split( data, target, test_size=0.5, random_state=1)
print("X_test.shape",X_test.shape)
print("y_test.shape",y_test.shape)
print("y_test\n",y_test)
X_test_s = ss.transform(X_test)
y_predict_test = clf.predict(X_test_s)
print("y_predict_test\n",y_predict_test)
result_file = 'results.csv'
if os.path.isfile(result_file) :
    os.remove(result_file)
df_dict = {
        "target": y_test,
        "predict": y_predict_test
    }
df = pd.DataFrame(data=df_dict)
df.to_csv(result_file, header=True, index=False) # 有表头，无索引

metric_precision = precision_score(y_test,y_predict_test, average='micro')
print("metric_precision",round(metric_precision,3))
metric_recall = recall_score(y_test,y_predict_test, average='micro')
print("metric_recall",round(metric_recall,3))
metric_f1 = f1_score(y_test,y_predict_test, average='micro')
print("metric_f1",round(metric_f1,3))