'''
用于决策树选择多个参数
'''

import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# 画出模型参数与模型评分的关系图
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
    plt.plot(train_sizes,test_scores_mean,'.--',color="r",
             label="Training score")
    plt.plot(train_sizes,test_scores_mean,'.-',color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    with open("E:\\modifiedversion\\Datasets\\all_original_pca_auto_shuffle_tune",'rb') as f:
        all_data = pickle.load(f)
    all_data = all_data.sample(n=50000, axis=0)
    X = all_data.drop('label', axis=1)
    y = all_data['label']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # 设置参数矩阵
    entropy_thresholds = np.linspace(0,1,50)
    gini_thresholds = np.linspace(0,0.4,40)
    param_grid = [{'criterion': ['entropy'],'min_impurity_decrease': entropy_thresholds, 'max_depth': range(8, 20),
            'min_samples_split': range(2, 30, 2),'min_samples_leaf':range(1,4)},
          {'criterion': ['gini'],'min_impurity_decrease': gini_thresholds, 'max_depth': range(8, 20),
            'min_samples_split': range(2, 30, 2),'min_samples_leaf':range(1,4)}]
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    clf.fit(X, y)
    print("best param:{0}\nbest_score:{1}".format(clf.best_params_, clf.best_score_))
        # plot_curve(depths, clf.cv_results_, xlabel='max depth of decision tree')