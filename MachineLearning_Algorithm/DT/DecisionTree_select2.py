'''
用于决策树选择参数min_impurity_split
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#方法一
# def cv_score(val):
#     with open('E:\\VgIndex2.py\\数据\\pca_data', 'rb') as f1:
#         pca_data = pickle.load(f1)
#     pca_data=pca_data.sample(n=10000,axis=0)
#     X = pca_data.drop('label', axis=1)
#     y = pca_data['label']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#     clf = DecisionTreeClassifier(criterion='gini',min_impurity_split=val)
#     clf.fit(X_train,y_train)
#     tr_score = clf.score(X_train,y_train)
#     cv_score = clf.score(X_test,y_test)
#     return (tr_score,cv_score)
#
# #指定参数范围，分别训练模型并计算评分
# values = np.linspace(0,0.5,50)
# scores = [cv_score(v) for v in values]
# tr_scores = [s[0] for s in scores]
# cv_scores = [s[1] for s in scores]
#
# # 找出评分最高的模型参数
# best_score_index = np.argmax(cv_scores)
# best_score = cv_scores[best_score_index]
# best_param = values[best_score_index]
# print('best param: {0};best scores: {1}'.format(best_param,best_score))
#
# # 画出模型参数与模型评分的关系
# plt.figure(figsize=(6,4), dpi=144)
# plt.grid()
# plt.xlabel('threashold of entropy')
# plt.ylabel('score')
# plt.plot(values,cv_scores,'.g--',label='cross-validation score')
# plt.plot(values, tr_scores, '.r--', label='training score')
# plt.legend()
# plt.show()

#方法二，gridsearch画出平均值
def plot_curve(train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']

    plt.figure(figsize=(6,4),dpi=144)
    # plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,
                     train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,
                     alpha=0.1,color="g")
    plt.plot(train_sizes,train_scores_mean,'.--',color="r",
             label="Training score")
    plt.plot(train_sizes,test_scores_mean,'.-',color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    with open("E:\\modifiedversion\\Datasets\\all_original_pca_auto_shuffle_tune",'rb') as f:
        all_data = pickle.load(f)
    all_data=all_data.sample(n=50000,axis=0)
    X = all_data.drop('label', axis=1)
    y = all_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    thresholds =np.linspace(0,0.5,50)
    param_grid ={'min_impurity_split':thresholds}
    clf = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
    clf.fit(X_train,y_train)
    print("best param:{0}\nbest_score:{1}".format(clf.best_params_,clf.best_score_))
    plot_curve(thresholds,clf.cv_results_,xlabel='Gini_thresholds')