from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.1,1.0,5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores=learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.grid()
    # 颜色填充，alpha为透明度
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="r")
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="g")
    plt.plot(train_sizes,test_scores_mean,'o-',color="r",label="Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color="g",label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features",polynomial_features),
                         ("linear_regresssion",linear_regression)])
    return pipeline

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

if __name__ == "__main__":

    all_data = pd.read_table("E:\\combinate_newdata\\all_index.txt", sep=" ",
                            names=['normalizeddiff_veg_index', 'simple_rat_index', 'diff_veg_index',
                                   'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'norm_red', 'norm_nir',
                                   'norm_green', 'cvi', 'green_red_ndvi', 'label'])
    with open("all_data", 'wb') as f:
        pickle.dump(all_data, f)
    with open("all_data", 'rb') as f:
        all_data = pickle.load(f)

    all_dataselect = all_data.drop(
        labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(
        n=20000, axis=0)
    pca = std_PCA(n_components=3)  # PCA降维为两个特征值
    y = np.array(all_dataselect['label']).ravel()
    X = all_dataselect.drop('label', axis=1)

    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    X_trans = pca.fit_transform(X_train)

    cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=42)
    plt.figure(figsize=(18,4),dpi=200)
    title = 'Learning Curves (degree={0})'
    degrees = [1,2,3]

    plt.figure(figsize=(18,4),dpi=200)
    for i in range(len(degrees)):
        print('start:{}LinearRegression'.format(degrees[i]))
        start=time.clock()
        model = polynomial_model(degrees[i])
        model.fit(X_trans,Y_train)
        train_score = model.score(X_trans,Y_train)
        Y_pred = model.predict(pca.fit_transform(X_val))
        cv_score = model.score(pca.fit_transform(X_val),Y_val)
        print('train_socre:{0:0.6f};cv_score:{1:0.6f}'.format(train_score,cv_score))
        print(confusion_matrix(Y_val, Y_pred))
        print(classification_report(Y_val, Y_pred))
        fpr, tpr, thresholds = metrics.roc_curve(Y_val, Y_pred)
        auc = metrics.auc(fpr, tpr)
        plt.subplot(1,3,i+1)
        plot_learning_curve(model,title.format(degrees[i]),X,y,ylim=(0.01, 1.01),cv=cv,n_jobs=-1)
        print('elaspe:{0:.6f}'.format(time.clock()-start))
    plt.show()