'''
比较svm各核函数并绘制二维图
'''

import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

def plot_hyperplance(clf, X, y, h=0.02, draw_sv=True, title='hyperplan'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) # meshgrid(x,y) 用两个坐标轴上的点在平面上画格
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    makers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y) #np.unique()是去除数组中的重复数字，并进行排序之后输出
    for label in labels:
        plt.scatter(X[y == label][:, 0], X[y == label][:, 1], c=colors[label], marker=makers[label])
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')


if __name__ == '__main__':
    all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
                             names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
                                    'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
                                    'norm_green', 'cvi', 'green_red_ndvi', 'label'])

    all_dataselect = all_data.drop(labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(n=20000,axis=0)
    pca = std_PCA(n_components=2) #PCA降维为两个特征值
    y = np.array(all_dataselect['label']).ravel()
    X = pca.fit_transform(all_dataselect.drop('label', axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    clf_linear = SVC(C=1,kernel='linear')
    clf_poly = SVC(C=1,kernel='poly',degree=3)
    clf_rbf = SVC(C=6.32,kernel='rbf',gamma=0.0947)
    clfs = [clf_linear,clf_poly,clf_rbf]
    titles = ['Linear Kernel',
              'Polynomial Kernel with Degree=3',
              'Gaussian Kernel with gamma=0.0947']
    results = []
    for clf,i in zip(clfs, range(len(clfs))):
        print('开始循环{}'.format(titles[i]))
        begin=time.time()
        clf.fit(X_train,y_train)
        plt.subplot(1,1,1)
        plot_hyperplance(clf,X_train,y_train,title=titles[i])
        # 做预测
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fpr, tpr)
        plt.show()
        print("{}循环一次时间".format(titles[i]),time.time()-begin)
        #
        kfold = KFold(n_splits=10)
        cv_result = cross_val_score(clf, X_train, y_train, cv=kfold)
        results.append((titles[i], cv_result))
    for i in range(len(results)):
        print("name:{};cross val score:{}".format(results[i][0],results[i][1].mean()))
