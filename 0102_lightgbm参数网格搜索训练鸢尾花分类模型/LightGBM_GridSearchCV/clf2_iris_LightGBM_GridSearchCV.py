# coding: utf-8
# lightgbm参数网格搜索 训练鸢尾花分类模型
# LightGBM 的 sklearn 风格接口 LGBMClassifier
# https://blog.csdn.net/variablex/article/details/107256149
from sklearn.datasets import load_iris
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, LGBMClassifier

# 加载样本数据集
iris = load_iris()
X,y = iris.data,iris.target
train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=0)   # 分训练集和验证集
train = lgb.Dataset(train_x, train_y)
valid = lgb.Dataset(valid_x, valid_y, reference=train)


parameters = {
              'max_depth': [5, 6, 7],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              # 'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              # 'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              # 'bagging_freq': [2, 4, 5, 6, 8],
              # 'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              # 'lambda_l2': [0, 10, 15, 35, 40],
              # 'cat_smooth': [1, 10, 15, 20, 35]
}
gbm = LGBMClassifier(
                    n_estimators=200, # 使用多少个弱分类器
                    objective='multiclass',
                    num_class=3,
                    num_leaves = 15,
                    # min_child_weight=2,
                    # subsample=0.8,
                    # colsample_bytree=0.8,
                    # reg_alpha=0,
                    # reg_lambda=1,
                    seed=0, # 随机数种子
                    silent=True , # 是否静默模式，True不打印警告和信息
                )
# 有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
gsearch.fit(train_x, train_y, verbose=False)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
