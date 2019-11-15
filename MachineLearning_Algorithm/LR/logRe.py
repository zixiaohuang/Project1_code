import time
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

# 增加多项式预处理
def polynomial_model(degree=1,**kwargs):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    logistic_regression = LogisticRegression(**kwargs)
    pipline = Pipeline([("polynomial_features",polynomial_features),
                        ("logistic_regression",logistic_regression)])
    return pipline

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

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
    plt.plot(train_sizes,train_scores_mean,'o-',color="r",label="Training score")
    plt.plot(train_sizes,test_scores_mean,'o-',color="g",label="Cross-validation score")
    plt.legend(loc="best")
    return plt

if __name__ == "__main__":
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
    title ='Learning Curves(degree={0}, penalty={1})'
    degrees = [5]
    penaltys=['l1','l2']


    for j in range(len(penaltys)):
        for i in range(len(degrees)):
            print('start:degree={0},penalty={1}LogisticRegression'.format(degrees[i],penaltys[j]))
            start = time.clock()
            model = polynomial_model(degree=degrees[i],penalty=penaltys[j])
            model.fit(X_trans, Y_train)
            train_score = model.score(X_trans, Y_train)
            Y_pred = model.predict(pca.fit_transform(X_val))
            cv_score = model.score(pca.fit_transform(X_val), Y_val)
            print('train_socre:{0:0.6f};cv_score:{1:0.6f}'.format(train_score, cv_score))
            print(confusion_matrix(Y_val, Y_pred))
            print(classification_report(Y_val, Y_pred))
            # plt.figure(figsize=(10, 10), dpi=144)
            fpr, tpr, thresholds = metrics.roc_curve(Y_val, Y_pred)
            auc = metrics.auc(fpr, tpr)
            plt.subplot(111)
            plot_learning_curve(polynomial_model(degree=degrees[i],penalty=penaltys[j]),
                            title.format(degrees[i],penaltys[j]),
                            X_trans,Y_train,ylim=(0.8,1.01),cv=cv,n_jobs=-1)
            print('elaspe:{0:.6f}'.format(time.clock() - start))
            plt.show()





