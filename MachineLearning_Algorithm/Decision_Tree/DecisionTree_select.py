'''
决策树选择参数max_depth
'''

import numpy as np
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#方法一
# def std_PCA(**argv):
#     scaler = StandardScaler()
#     pca = PCA(**argv)
#     pipeline = Pipeline([('scaler',scaler),
#                          ('pca',pca)])
#     return pipeline
#
# def cv_score(d):
#     # all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
#     #                          names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
#     #                                 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
#     #                                 'norm_green', 'cvi', 'green_red_ndvi', 'label'])
#     #
#     # all_dataselect = all_data.drop(
#     #     labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(
#     #     n=20000,
#     #     axis=0)
#     # pca = std_PCA(n_components=2)  # PCA降维为两个特征值
#     # y = np.array(all_dataselect['label']).ravel()
#     # X = all_dataselect.drop('label', axis=1)
#     # X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
#     with open('E:\\VgIndex2.py\\数据\\pca_data', 'rb') as f1:
#         pca_data = pickle.load(f1)
#     pca_data=pca_data.sample(n=10000,axis=0)
#     X = pca_data.drop('label', axis=1)
#     y = pca_data['label']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#     clf = DecisionTreeClassifier(max_depth=d)
#     clf.fit(X_train,y_train)
#     tr_score = clf.score(X_train,y_train)
#     cv_score = clf.score(X_test,y_test)
#     return (tr_score, cv_score)
#
# if __name__ == "__main__":
#     depths = range(2, 150)
#     scores = [cv_score(d) for d in depths]
#     tr_scores = [s[0] for s in scores]
#     cv_scores = [s[1] for s in scores]
#
#     # 找出交叉验证数据集评分最高的索引
#     best_score_index = np.argmax(cv_scores)
#     best_score = cv_scores[best_score_index]
#     best_param = depths[best_score_index]
#     print('best param: {0}; best_scores: {1}'.format(best_param, best_score))
#
#     # 绘制模型参数图
#     plt.figure()
#     plt.grid()
#     plt.xlabel('max depth of decision tree')
#     plt.ylabel('score')
#     plt.plot(depths,cv_scores,'.g-',label='cross validation score')
#     plt.plot(depths,tr_scores,'.r--',label='training score')
#     plt.legend()
#     plt.show()

#方法二，gridsearch画出平均值
def plot_curve(train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']

    plt.figure(figsize=(6,4),dpi=144)
    plt.title('parameters turning')
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
    with open('E:\\VgIndex2.py\\数据\\pca_data', 'rb') as f1:
        pca_data = pickle.load(f1)
    pca_data=pca_data.sample(n=20000,axis=0)
    X = pca_data.drop('label', axis=1)
    y = pca_data['label']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    depths = range(2, 150)
    param_grid ={'max_depth':depths}
    clf = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
    clf.fit(X,y)
    print("best param:{0}\nbest_score:{1}".format(clf.best_params_,clf.best_score_))
    plot_curve(depths,clf.cv_results_,xlabel='max depth of decision tree')