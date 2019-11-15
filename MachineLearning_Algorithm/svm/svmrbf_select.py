'''
svm高斯核函数选择gamma值
'''
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle

# train_sizes=np.linspace(.1,1.0,5)表示把训练样本数量从0.1~1分成五等分，生成[0.1,0.325,0.55,0.775,1]
def plot_param_curve(plt, train_sizes, cv_results, xlabel):
    train_scores_mean = (cv_results['mean_train_score'])
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '.--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '.-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == "__main__":
    with open("E:\\modifiedversion\\Datasets\\all_original_pca_auto_shuffle_tune",'rb') as f:
        all_data = pickle.load(f)
    y = np.array(all_data['label']).ravel()
    X = all_data.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.997, random_state = 42) # 选部分样本进行计算
    gammas = np.linspace(0, 0.1,20)
    clf_C = np.linspace(0.01, 10, 20)
    param_grid = {'C':clf_C,'gamma':gammas,'kernel':['rbf']}
    clf = GridSearchCV(SVC(), param_grid, cv=5,n_jobs=-1)
    clf.fit(X_train,y_train)
    print("best param: {0}\n best score: {1}".format(clf.best_params_,clf.best_score_))
    #plt.figure(figsize=(10,4),dpi=144)
    #plot_param_curve(plt,gammas,clf.cv_results_,xlabel='gamma')
    #plt.show()